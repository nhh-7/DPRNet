# 当前 DPR 超越 CATANet 的优化方案

## 1. 文档目标

本文面向当前仓库中的实际实现，目标不是重新定义 DPR，而是基于 `CATANet/basicsr/archs/catanet_arch.py` 中已经落地的 DPR 替换版，分析它为什么未必能稳定超过原 CATANet，并给出一套更有可能提升最终 PSNR/SSIM 的最小优化方案。

当前 DPR 已完成以下主链：

```text
TAB 输入特征
-> flatten + LayerNorm
-> DPR 动态 prototype 生成
-> token-to-prototype 匹配
-> argmax + argsort 内容重排
-> IASA 在排序后 token 上聚合
-> scatter 回原空间顺序
-> FFN
```

这个方向有潜力，但当前实现仍偏“结构替换可运行版”。要更有可能超过原 CATANet，需要优先解决路由置信度、硬路由风险、prototype 塌缩、全局注入过强和验证可视化错误这几个问题。

---

## 2. 当前 DPR 的关键不足

### 2.1 路由置信度被固定 scale 压低

当前最终路由代码为：

```python
token_features = F.normalize(self.token_proj(embed), dim=-1)
prototype_features = F.normalize(self.prototype_proj(prototypes), dim=-1)
score_logits = torch.matmul(token_features, prototype_features.transpose(-2, -1)) * self.scale
scores = F.softmax(score_logits, dim=-1)
```

这里 token/prototype 已经做了 L2 normalize，点积本身是 cosine 相似度。继续乘 `router_dim ** -0.5` 会让 logits 过小，softmax 接近均匀。结果是：

- `x_scores` 区分度弱，不能可靠表示归属置信度。
- 组内 confidence sort 作用有限。
- argmax 路由容易被微小噪声影响。

这会直接降低 DPR 构造高质量候选集合的能力。

### 2.2 hard routing 没有低置信保护

当前 DPR 使用：

```python
x_scores, belong_idx = torch.max(scores, dim=-1)
sorted_idx = torch.argsort(sort_key, dim=-1)
sorted_x = torch.gather(x, dim=1, index=gather_idx)
```

`argmax + argsort` 是硬离散路径。训练早期 prototype 尚不稳定时，错误分组会直接影响 IASA 的 attention 邻域。对于 SR 中的边缘、细线、纹理过渡区域，低置信 token 被强行归入某一组，很容易造成细节模糊或纹理错位。

### 2.3 `sorted_scores` 没有参与聚合

当前 DPR 返回了 `sorted_scores`，但 `TAB.forward()` 中直接丢弃：

```python
sorted_x, idx_last, belong_idx, _, prototypes = self.dpr(x)
y = self.iasa_attn(sorted_x, idx_last, prototypes)
```

这意味着高置信 token 和低置信 token 接受完全相同的 IASA 聚合强度。DPR 的置信度信息没有转化为重建收益。

### 2.4 IASA 的 prototype global attention 直接相加

当前 IASA 中：

```python
out1 = F.scaled_dot_product_attention(paded_q, paded_k, paded_v)
out2 = F.scaled_dot_product_attention(paded_q, k_global, v_global)
out = out1 + out2
```

`out2` 来自 DPR prototypes，但它没有门控。若训练早期 prototypes 不准，或者某些 token 的路由置信度低，全局 prototype 注入仍然强制生效，可能伤害局部重建。

### 2.5 prototype 使用率没有约束

当前 `assignment = softmax(assign(embed))` 后直接加权聚合。没有任何约束保证 prototype 使用均衡，可能出现：

- 少数 prototype 吸收大量 token。
- 某些 prototype 近似空槽。
- 多个 prototype 表达重复内容。

这会让排序后的 token 分组质量不稳定，尤其影响不同随机种子下的结果方差。

### 2.6 routing map 可视化存在空间错位和潜在报错

当前 `TAB.last_routing_map` 保存的是 `sorted_belong_idx`，不是原始空间顺序的 `belong_idx`。直接 reshape 成 `H x W` 会错位。

此外，`CATANetModel.nondist_validation()` 中先执行：

```python
del self.lq
```

后面保存 cluster map 时又访问：

```python
_, _, h_lq, w_lq = self.lq.shape
```

这在 `save_img=True` 时有潜在报错风险。

---

## 3. 推荐保留的核心优化

为了避免方案过散，本文只保留 4 个最可能影响最终指标的优化。它们按优先级排序如下：

1. **可学习路由温度 + DPR 输出接口修正**。
2. **confidence-aware IASA 门控**。
3. **低置信 token 的 soft prototype fallback**。
4. **prototype 使用率均衡辅助损失**。

其中 1 是基础，2 和 3 是最有可能带来指标提升的核心，4 主要提高稳定性和可复现性。

---

## 4. 优化一：可学习路由温度与输出接口修正

### 4.1 目标

让 token-to-prototype scores 具备真实区分度，并让 DPR 返回原始顺序的路由信息，供后续 gate、fallback、可视化和诊断使用。

### 4.2 代码改动位置

文件：`CATANet/basicsr/archs/catanet_arch.py`

类：`DPR`

### 4.3 修改 `DPR.__init__()`

当前：

```python
self.prototype_norm = nn.LayerNorm(dim)
self.refine_gate = nn.Parameter(init_logit(refine_init))
self.scale = router_dim ** -0.5
```

建议改为：

```python
self.prototype_norm = nn.LayerNorm(dim)
self.refine_gate = nn.Parameter(init_logit(refine_init))

# refine attention 仍使用标准 qk scale
self.scale = router_dim ** -0.5

# 最终 token-to-prototype 路由使用可学习 cosine logit scale
self.router_logit_scale = nn.Parameter(torch.ones([]) * 2.302585093)  # log(10)
self.max_router_logit_scale = 50.0
```

### 4.4 修改最终路由 logits

当前：

```python
score_logits = torch.matmul(token_features, prototype_features.transpose(-2, -1)) * self.scale
scores = F.softmax(score_logits, dim=-1)
```

改为：

```python
router_scale = self.router_logit_scale.exp().clamp(max=self.max_router_logit_scale)
score_logits = torch.matmul(token_features, prototype_features.transpose(-2, -1)) * router_scale
scores = F.softmax(score_logits, dim=-1)
```

### 4.5 修改 DPR 返回值

当前：

```python
return sorted_x, idx_last, sorted_belong_idx, sorted_scores, prototypes
```

建议改为：

```python
route_info = {
    'belong_idx': belong_idx,                  # 原始顺序 B x N
    'x_scores': x_scores,                      # 原始顺序 B x N
    'scores': scores,                          # 原始顺序 B x N x M
    'sorted_belong_idx': sorted_belong_idx,    # 排序后 B x N
    'sorted_scores': sorted_scores,            # 排序后 B x N
    'sorted_idx': sorted_idx,
}
return sorted_x, idx_last, prototypes, route_info
```

### 4.6 修改 `TAB.forward()` 接口

当前：

```python
sorted_x, idx_last, belong_idx, _, prototypes = self.dpr(x)
self.last_routing_map = belong_idx
```

改为：

```python
sorted_x, idx_last, prototypes, route_info = self.dpr(x)
self.last_routing_map = route_info['belong_idx'].detach()
self.last_sorted_routing_map = route_info['sorted_belong_idx'].detach()
```

### 4.7 预期收益

- `x_scores` 从接近均匀变为有明显区分度。
- 排序质量提升。
- 后续 confidence gate 和 soft fallback 有可靠输入。
- routing map 可视化恢复到原始空间顺序。

---

## 5. 优化二：confidence-aware IASA 门控

### 5.1 目标

控制 DPR prototypes 对 IASA 的全局注入强度。高置信 token 可以更多使用 prototype global attention，低置信 token 减弱全局注入，避免错误 prototype 影响局部重建。

### 5.2 代码改动位置

文件：`CATANet/basicsr/archs/catanet_arch.py`

类：`IASA`

### 5.3 修改 `IASA.__init__()`

新增：

```python
self.global_gate = nn.Parameter(torch.tensor(-1.0))
```

### 5.4 修改 `IASA.forward()` 签名

当前：

```python
def forward(self, sorted_x, idx_last, prototypes):
```

改为：

```python
def forward(self, sorted_x, idx_last, prototypes, sorted_scores=None):
```

### 5.5 替换 out1/out2 融合方式

当前：

```python
out2 = F.scaled_dot_product_attention(paded_q,k_global,v_global)
out = out1 + out2
```

改为：

```python
out2 = F.scaled_dot_product_attention(paded_q, k_global, v_global)
global_gate = torch.sigmoid(self.global_gate)

if sorted_scores is not None:
    if pad_n > 0:
        paded_scores = torch.cat(
            (sorted_scores, torch.flip(sorted_scores[:, N-pad_n:N], dims=[-1])),
            dim=-1
        )
    else:
        paded_scores = sorted_scores
    score_gate = rearrange(paded_scores, 'b (ng gs) -> b ng 1 gs 1', ng=ng, gs=gs)
    out = out1 + global_gate * score_gate * out2
else:
    out = out1 + global_gate * out2
```

### 5.6 修改 `TAB.forward()` 调用

```python
hard_y = self.iasa_attn(
    sorted_x,
    idx_last,
    prototypes,
    route_info['sorted_scores']
)
```

### 5.7 预期收益

- 训练早期 prototype 不稳定时，降低全局错误注入。
- 路由高置信区域能获得更强长程内容聚合。
- 边缘和纹理过渡区域更安全。

---

## 6. 优化三：低置信 token 的 soft prototype fallback

### 6.1 目标

DPR 的 hard sort 有利于构建语义候选集合，但低置信 token 不应完全依赖 hard routing。增加 soft prototype fallback，使低置信 token 能从连续可微的 prototype mixture 中获得补偿。

### 6.2 代码改动位置

文件：`CATANet/basicsr/archs/catanet_arch.py`

类：`TAB`

### 6.3 修改 `TAB.__init__()`

新增：

```python
self.soft_context_proj = nn.Linear(dim, dim, bias=False)
self.soft_fallback_gate = nn.Parameter(torch.tensor(-2.0))
```

### 6.4 修改 `TAB.forward()`

在 IASA 得到 `hard_y` 后加入：

```python
soft_context = torch.matmul(route_info['scores'], prototypes)  # B x N x C, 原始顺序
soft_context = self.soft_context_proj(soft_context)
low_conf = (1.0 - route_info['x_scores'].detach()).unsqueeze(-1)
y = hard_y + torch.sigmoid(self.soft_fallback_gate) * low_conf * soft_context
```

这里建议先对 `x_scores` 使用 `.detach()`，避免模型通过人为降低置信度来过度使用 fallback。

### 6.5 推荐的 `TAB.forward()` 主体

```python
def forward(self, x):
    _, _, h, w = x.shape
    x = rearrange(x, 'b c h w->b (h w) c')
    residual = x
    x = self.norm(x)

    sorted_x, idx_last, prototypes, route_info = self.dpr(x)
    self.last_routing_map = route_info['belong_idx'].detach()
    self.last_sorted_routing_map = route_info['sorted_belong_idx'].detach()

    hard_y = self.iasa_attn(sorted_x, idx_last, prototypes, route_info['sorted_scores'])

    soft_context = torch.matmul(route_info['scores'], prototypes)
    soft_context = self.soft_context_proj(soft_context)
    low_conf = (1.0 - route_info['x_scores'].detach()).unsqueeze(-1)
    y = hard_y + torch.sigmoid(self.soft_fallback_gate) * low_conf * soft_context

    y = rearrange(y, 'b (h w) c->b c h w', h=h).contiguous()
    y = self.conv1x1(y)
    x = residual + rearrange(y, 'b c h w->b (h w) c')
    x = self.mlp(x, x_size=(h, w)) + x
    return rearrange(x, 'b (h w) c->b c h w', h=h)
```

### 6.6 预期收益

- 降低 hard routing 错误造成的细节破坏。
- 让边界、细线、混合纹理区域更稳定。
- 提高 Urban100 / Manga109 上超过原 CATANet 的概率。

---

## 7. 优化四：prototype 使用率均衡辅助损失

### 7.1 目标

防止 prototype 空槽或塌缩，提高 DPR 分组稳定性。该优化的主要价值是降低方差和坏 seed 风险，不一定带来最大的单次涨点。

### 7.2 代码改动位置

- `CATANet/basicsr/archs/catanet_arch.py`
- `CATANet/basicsr/models/catanet_model.py`
- `CATANet/options/train/*.yml`

### 7.3 修改 `DPR.__init__()`

建议添加参数：

```python
def __init__(self, dim, router_dim, num_prototypes,
             use_prototype_query_refine=True, refine_init=0.25,
             balance_loss_weight=0.0):
```

并设置：

```python
self.balance_loss_weight = balance_loss_weight
self.aux_loss = None
self.last_usage = None
```

### 7.4 在 `DPR.forward()` 中计算 balance loss

在 `assignment` 后加入：

```python
usage = assignment.mean(dim=1)  # B x M
self.last_usage = usage.detach().mean(dim=0)

if self.training and self.balance_loss_weight > 0:
    target = torch.full_like(usage, 1.0 / self.num_prototypes)
    loss_balance = ((usage - target) ** 2).mean()
    self.aux_loss = self.balance_loss_weight * loss_balance
else:
    self.aux_loss = None
```

### 7.5 修改 `TAB` 和 `CATANet` 参数传递

`TAB.__init__()` 增加：

```python
route_balance_weight=0.0
```

并修改：

```python
self.dpr = DPR(dim, qk_dim, num_tokens, balance_loss_weight=route_balance_weight)
```

`CATANet.__init__()` 增加：

```python
route_balance_weight=0.0
```

构造 TAB 时使用关键字传入：

```python
TAB(
    self.dim, self.qk_dim, self.mlp_dim,
    self.heads,
    n_iter=self.n_iters[i],
    num_tokens=self.num_tokens[i],
    group_size=self.group_size[i],
    route_balance_weight=route_balance_weight
)
```

### 7.6 在训练 loss 中汇总 aux_loss

文件：`CATANet/basicsr/models/catanet_model.py`

在 `optimize_parameters()` 中，`l_total.backward()` 前加入：

```python
l_route = None
for module in self.net_g.modules():
    aux_loss = getattr(module, 'aux_loss', None)
    if aux_loss is not None:
        l_route = aux_loss if l_route is None else l_route + aux_loss

if l_route is not None:
    l_total += l_route
    loss_dict['l_route'] = l_route
```

### 7.7 修改训练配置

在训练 yml 中：

```yaml
network_g:
  type: CATANet
  upscale: 2
  route_balance_weight: 0.001
```

建议初始值只用 `0.001`。如果指标下降，先改为 `0.0005` 或关闭。

---

## 8. 必须修复的验证与可视化问题

这些修复不直接提升模型能力，但会避免错误判断 DPR 是否有效。

### 8.1 cluster map 使用原始顺序 belong_idx

通过优化一，`self.last_routing_map` 应保存原始顺序：

```python
self.last_routing_map = route_info['belong_idx'].detach()
```

不要用 `sorted_belong_idx` 直接 reshape 成图。

### 8.2 保存 cluster map 前保留输入尺寸

当前代码先删除 `self.lq` 再访问 shape。建议在 `del self.lq` 前保存：

```python
lq_shape = self.lq.shape
```

后面使用：

```python
_, _, h_lq, w_lq = lq_shape
```

或者把 cluster map 保存逻辑移动到 `del self.lq` 之前。

### 8.3 修复 self-ensemble 非 EMA 分支

当前非 EMA 分支中错误调用了 `self.net_g_ema`：

```python
out_list = [self.net_g_ema(aug) for aug in lq_list]
```

应改为：

```python
out_list = [self.net_g(aug) for aug in lq_list]
```

---

## 9. 推荐实验顺序

### Stage 0：建立当前 DPR baseline

先不改结构，完整记录当前 DPR 在相同训练设置下的指标：

- Set5
- Set14
- B100
- Urban100
- Manga109

同时记录每个 TAB block 的：

```text
x_scores mean / std / min / max
prototype usage
空槽比例
```

### Stage 1：只做优化一

只加入 learnable router scale 和 route_info 输出修正。

判断标准：

- `x_scores.std()` 是否变大。
- `x_scores.mean()` 是否不再接近 `1 / num_prototypes`。
- PSNR 是否至少不下降。

### Stage 2：加入优化二

加入 IASA global gate 和 sorted score gate。

判断标准：

- Urban100 / Manga109 是否改善。
- B100 是否不明显下降。
- 训练初期 loss 是否更平稳。

### Stage 3：加入优化三

加入 soft prototype fallback。

判断标准：

- 边缘和重复纹理可视化是否更稳。
- Urban100 / Manga109 是否进一步提升。
- 若图像变平滑，则降低 `soft_fallback_gate` 初始值。

### Stage 4：加入优化四

加入 route balance loss。

判断标准：

- prototype usage 是否更均衡。
- 多 seed 方差是否下降。
- 若 PSNR 下降，降低权重或只在 warmup 阶段启用。

---

## 10. 预期结果判断

更可能成功的结果形态是：

```text
Urban100 / Manga109 明显提升
Set5 / Set14 基本持平
B100 不下降或轻微提升
```

如果只在 Urban100 提升但 B100 明显下降，说明 DPR 的内容路由过强，泛化或空间结构保护不足。此时应降低 `global_gate` 或 `soft_fallback_gate`，不要继续增加 prototype 数量。

如果所有数据集都下降，优先检查：

1. `router_logit_scale` 是否过大。
2. `x_scores` 是否过度尖锐。
3. balance loss 权重是否过高。
4. `soft_context` 是否导致结果过平滑。

---

## 11. 最终推荐版本

最推荐的 DPR-CATANet 目标形态是：

```text
DPR 动态 prototype
+ 可学习 routing temperature
+ confidence-aware IASA global injection
+ low-confidence soft prototype fallback
+ weak prototype balance loss
```

这不是单纯“用 DPR hard sort 替换 TAB”，而是把 DPR 改成一个更适合 SR 的稳定内容组织模块：

```text
高置信区域：充分利用内容路由和长程相似 token
低置信区域：保留 soft 原型上下文，避免错误硬分组
全局注入：由 gate 控制，避免 prototype 不稳定时伤害重建
prototype 槽位：用弱 balance 防止塌缩
```

这套优化比继续堆复杂结构更有可能稳定超过原 CATANet，因为它直接处理了当前 DPR 最核心的失败模式：置信度不准、硬路由脆弱、全局注入过强和 prototype 不均衡。
