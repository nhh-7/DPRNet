import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from inspect import isfunction
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import trunc_normal_

def exists(val):
    return val is not None

def is_empty(t):
    return t.nelement() == 0

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def default(x, d):
    if not exists(x):
        return d if not isfunction(d) else d()
    return x

def ema(old, new, decay):
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)

def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha= (1 - decay))


def init_logit(prob: float) -> torch.Tensor:
    prob = min(max(prob, 1e-4), 1.0 - 1e-4)
    return torch.logit(torch.tensor(prob, dtype=torch.float32))


def similarity(x, means):
    return torch.einsum('bld,cd->blc', x, means)

def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets

def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out

def center_iter(x, means, buckets = None):
    b, l, d, dtype, num_tokens = *x.shape, x.dtype, means.shape[0]

    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, num_tokens).sum(0, keepdim=True)
    zero_mask = bins.long() == 0

    means_ = buckets.new_zeros(b, num_tokens, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)
    return means
    
class IASA(nn.Module):
    def __init__(self, dim, qk_dim, heads, group_size):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.prototype_to_k = nn.Linear(dim, qk_dim, bias=False)
        self.prototype_to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.group_size = group_size
        
    
    def forward(self, sorted_x, idx_last, prototypes):
        x = sorted_x
        _, N, _ = x.shape
       
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
   
        gs = min(N, self.group_size)  # group size
        ng = (N + gs - 1) // gs
        pad_n = ng * gs - N
        
        paded_q = torch.cat((q, torch.flip(q[:,N-pad_n:N, :], dims=[-2])), dim=-2)
        paded_q = rearrange(paded_q, "b (ng gs) (h d) -> b ng h gs d",ng=ng,h=self.heads)
        paded_k = torch.cat((k, torch.flip(k[:,N-pad_n-gs:N, :], dims=[-2])), dim=-2)
        paded_k = paded_k.unfold(-2,2*gs,gs)
        paded_k = rearrange(paded_k, "b ng (h d) gs -> b ng h gs d",h=self.heads)
        paded_v = torch.cat((v, torch.flip(v[:,N-pad_n-gs:N, :], dims=[-2])), dim=-2)
        paded_v = paded_v.unfold(-2,2*gs,gs)
        paded_v = rearrange(paded_v, "b ng (h d) gs -> b ng h gs d",h=self.heads)
        out1 = F.scaled_dot_product_attention(paded_q,paded_k,paded_v)
        
        
        k_global = self.prototype_to_k(prototypes)
        v_global = self.prototype_to_v(prototypes)
        k_global = rearrange(k_global, "b n (h d) -> b h n d", h=self.heads)
        v_global = rearrange(v_global, "b n (h d) -> b h n d", h=self.heads)
        k_global = k_global.unsqueeze(1).expand(-1,ng,-1,-1,-1)
        v_global = v_global.unsqueeze(1).expand(-1,ng,-1,-1,-1)
       
        out2 = F.scaled_dot_product_attention(paded_q,k_global,v_global)
        out = out1 + out2
        out = rearrange(out, "b ng h gs d -> b (ng gs) (h d)")[:, :N, :]
 
        out = out.scatter(dim=-2, index=idx_last.expand_as(out), src=out)
        out = self.proj(out)
    
        return out
    
class DPR(nn.Module):
    def __init__(self, dim, router_dim, num_prototypes,
                 use_prototype_query_refine=True, refine_init=0.25):
        super().__init__()
        self.dim = dim
        self.router_dim = router_dim
        self.num_prototypes = num_prototypes
        self.use_prototype_query_refine = use_prototype_query_refine

        self.embed = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.assign = nn.Linear(dim, num_prototypes, bias=False)
        self.prototype_queries = nn.Parameter(torch.randn(num_prototypes, dim) * 0.02)

        self.refine_q = nn.Linear(dim, router_dim, bias=False)
        self.refine_k = nn.Linear(dim, router_dim, bias=False)
        self.refine_v = nn.Linear(dim, dim, bias=False)

        self.token_proj = nn.Linear(dim, router_dim, bias=False)
        self.prototype_proj = nn.Linear(dim, router_dim, bias=False)

        self.prototype_norm = nn.LayerNorm(dim)
        self.refine_gate = nn.Parameter(init_logit(refine_init))
        self.scale = router_dim ** -0.5

    def forward(self, x):
        b, n, c = x.shape
        embed = self.embed(x)

        assignment = F.softmax(self.assign(embed), dim=-1)
        proto_content = torch.einsum("bnm,bnc->bmc", assignment, x)
        proto_weight = assignment.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        prototypes = proto_content / proto_weight
        prototypes = F.normalize(self.prototype_norm(prototypes), dim=-1)

        if self.use_prototype_query_refine:
            query_seed = prototypes + self.prototype_queries.unsqueeze(0)
            q_proto = self.refine_q(query_seed)
            k_tokens = self.refine_k(embed)
            v_tokens = self.refine_v(x)
            refine_logits = torch.matmul(q_proto, k_tokens.transpose(-2, -1)) * self.scale
            refine_attn = F.softmax(refine_logits, dim=-1)
            proto_refine = torch.matmul(refine_attn, v_tokens)

            gamma = torch.sigmoid(self.refine_gate)
            prototypes = F.normalize(self.prototype_norm(prototypes + gamma * proto_refine), dim=-1)

        token_features = F.normalize(self.token_proj(embed), dim=-1)
        prototype_features = F.normalize(self.prototype_proj(prototypes), dim=-1)
        score_logits = torch.matmul(token_features, prototype_features.transpose(-2, -1)) * self.scale
        scores = F.softmax(score_logits, dim=-1)

        x_scores, belong_idx = torch.max(scores, dim=-1)
        sort_key = belong_idx.to(scores.dtype) + 0.5 * (1.0 - x_scores)
        sorted_idx = torch.argsort(sort_key, dim=-1)

        gather_idx = sorted_idx.unsqueeze(-1).expand(b, n, c)
        sorted_x = torch.gather(x, dim=1, index=gather_idx)
        sorted_belong_idx = torch.gather(belong_idx, dim=1, index=sorted_idx)
        sorted_scores = torch.gather(x_scores, dim=1, index=sorted_idx)
        idx_last = sorted_idx.unsqueeze(-1)

        return sorted_x, idx_last, sorted_belong_idx, sorted_scores, prototypes
    
    
class TAB(nn.Module):
    def __init__(self, dim, qk_dim, mlp_dim, heads, n_iter=3,
                 num_tokens=8, group_size=128,
                 ema_decay = 0.999):
        super().__init__()

        self.n_iter = n_iter
        self.ema_decay = ema_decay
        self.num_tokens = num_tokens
        
        
        self.norm = nn.LayerNorm(dim)
        self.mlp = PreNorm(dim, ConvFFN(dim,mlp_dim))
        self.dpr = DPR(dim, qk_dim, num_tokens)
        self.iasa_attn = IASA(dim,qk_dim,heads,group_size)
        self.conv1x1 = nn.Conv2d(dim,dim,1, bias=False)

    
    def forward(self, x):
        _,_,h, w = x.shape
        x = rearrange(x, 'b c h w->b (h w) c')
        residual = x
        x = self.norm(x)
        sorted_x, idx_last, _, _, prototypes = self.dpr(x)
        y = self.iasa_attn(sorted_x, idx_last, prototypes)
        y = rearrange(y,'b (h w) c->b c h w',h=h).contiguous()
        y = self.conv1x1(y)
        x = residual + rearrange(y, 'b c h w->b (h w) c')
        x = self.mlp(x, x_size=(h, w)) + x
    
        return rearrange(x, 'b (h w) c->b c h w',h=h)
        
        
        

def patch_divide(x, step, ps):
    """Crop image into patches.
    Args:
        x (Tensor): Input feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        crop_x (Tensor): Cropped patches.
        nh (int): Number of patches along the horizontal direction.
        nw (int): Number of patches along the vertical direction.
    """
    b, c, h, w = x.size()
    if h == ps and w == ps:
        step = ps
    crop_x = []
    nh = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        nh += 1
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh
    crop_x = torch.stack(crop_x, dim=0)  # (n, b, c, ps, ps)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()  # (b, n, c, ps, ps)
    return crop_x, nh, nw


def patch_reverse(crop_x, x, step, ps):
    """Reverse patches into image.
    Args:
        crop_x (Tensor): Cropped patches.
        x (Tensor): Feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        output (Tensor): Reversed image.
    """
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
    for i in range(step, h + step - ps, step):
        top = i
        down = i + ps - step
        if top + ps > h:
            top = h - ps
        output[:, :, top:down, :] /= 2
    for j in range(step, w + step - ps, step):
        left = j
        right = j + ps - step
        if left + ps > w:
            left = w - ps
        output[:, :, :, left:right] /= 2
    return output


class PreNorm(nn.Module):
    """Normalization layer.
    Args:
        dim (int): Base channels.
        fn (Module): Module after normalization.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        heads (int): Head numbers.
        qk_dim (int): Channels of query and key.
    """

    def __init__(self, dim, heads, qk_dim):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        
        

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
       
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
      
        out = F.scaled_dot_product_attention(q,k,v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


class LRSA(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        num (int): Number of blocks.
        qk_dim (int): Channels of query and key in Attention.
        mlp_dim (int): Channels of hidden mlp in Mlp.
        heads (int): Head numbers of Attention.
    """

    def __init__(self, dim, qk_dim, mlp_dim,heads=1):
        super().__init__()
     

        self.layer = nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, qk_dim)),
                PreNorm(dim, ConvFFN(dim, mlp_dim))])

    def forward(self, x, ps):
        step = ps - 2
        crop_x, nh, nw = patch_divide(x, step, ps)  # (b, n, c, ps, ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')

        attn, ff = self.layer
        crop_x = attn(crop_x) + crop_x
        crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)
        
        x = patch_reverse(crop_x, x, step, ps)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w-> b (h w) c')
        x = ff(x, x_size=(h, w)) + x
        x = rearrange(x, 'b (h w) c->b c h w', h=h)
        
        return x


    
@ARCH_REGISTRY.register()
class CATANet(nn.Module):
    setting = dict(dim=40, block_num=8, qk_dim=36, mlp_dim=96, heads=4, 
                     patch_size=[16, 20, 24, 28, 16, 20, 24, 28])

    def __init__(self,in_chans=3,n_iters=[5,5,5,5,5,5,5,5],
                 num_tokens=[16,32,64,128,16,32,64,128],
                 group_size=[256,128,64,32,256,128,64,32],
                 upscale: int = 4):
        super().__init__()
        
    
        self.dim = self.setting['dim']
        self.block_num = self.setting['block_num']
        self.patch_size = self.setting['patch_size']
        self.qk_dim = self.setting['qk_dim']
        self.mlp_dim = self.setting['mlp_dim']
        self.upscale = upscale
        self.heads = self.setting['heads']
        
        


        self.n_iters = n_iters
        self.num_tokens = num_tokens
        self.group_size = group_size
    
        #-----------1 shallow--------------
        self.first_conv = nn.Conv2d(in_chans, self.dim, 3, 1, 1)

        #----------2 deep--------------
        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
   
        for i in range(self.block_num):
          
            self.blocks.append(nn.ModuleList([TAB(self.dim, self.qk_dim, self.mlp_dim,
                                                                 self.heads, self.n_iters[i], 
                                                                 self.num_tokens[i],self.group_size[i]), 
                                              LRSA(self.dim, self.qk_dim, 
                                                             self.mlp_dim,self.heads)]))
            self.mid_convs.append(nn.Conv2d(self.dim, self.dim,3,1,1))
            
        #----------3 reconstruction---------
        
      
     
        if upscale == 4:
            self.upconv1 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif upscale == 2 or upscale == 3:
            self.upconv = nn.Conv2d(self.dim, self.dim * (upscale ** 2), 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(upscale)
    
        self.last_conv = nn.Conv2d(self.dim, in_chans, 3, 1, 1)
        if upscale != 1:
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        for i in range(self.block_num):
            residual = x
      
            global_attn,local_attn = self.blocks[i]
            
            x = global_attn(x)
            
            x = local_attn(x, self.patch_size[i])
            
            x = residual + self.mid_convs[i](x)
        return x
        
    def forward(self, x):
        
        if self.upscale != 1: 
            base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        else: 
            base = x
        x = self.first_conv(x)
        
   
        x = self.forward_features(x) + x
    
        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 1:
            out = x
        else:
            out = self.lrelu(self.pixel_shuffle(self.upconv(x)))
        out = self.last_conv(out) + base
       
    
        return out
    
    
    def __repr__(self):
        num_parameters = sum(map(lambda x: x.numel(), self.parameters()))
        return '#Params of {}: {:<.4f} [K]'.format(self._get_name(),
                                                      num_parameters / 10 ** 3) 
  
  


if __name__ == '__main__':


    model = CATANet(upscale=3).cuda()
    x = torch.randn(2, 3, 128, 128).cuda()
    print(model)
 
  
