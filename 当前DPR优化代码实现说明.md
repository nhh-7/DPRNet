# 当前 DPR 优化代码实现说明

## 1. 修改范围

本次实际代码实现了 `当前DPR超越CATANet优化方案.md` 中的前三项：

1. 可学习路由温度 + DPR 输出接口修正。
2. confidence-aware IASA 门控。
3. 低置信 token 的 soft prototype fallback。

修改文件：

```text
CATANet/basicsr/archs/catanet_arch.py
```

未改动训练配置、loss 汇总和 prototype balance loss；该部分属于后续第 4 项优化。

---

## 2. DPR：可学习路由温度与输出接口修正

### 2.1 新增最终路由专用温度

在 `DPR.__init__()` 中新增：

```python
self.router_logit_scale = nn.Parameter(torch.ones([]) * 2.302585093)  # log(10)
self.max_router_logit_scale = 50.0
```

原来的 `self.scale = router_dim ** -0.5` 继续保留，只用于 prototype refine attention。

### 2.2 替换最终 token-to-prototype logits

原实现：

```python
score_logits = torch.matmul(token_features, prototype_features.transpose(-2, -1)) * self.scale
```

现实现：

```python
router_scale = self.router_logit_scale.exp().clamp(max=self.max_router_logit_scale)
score_logits = torch.matmul(token_features, prototype_features.transpose(-2, -1)) * router_scale
```

这样避免 normalized cosine similarity 再被 `1 / sqrt(router_dim)` 压得过小，使 `x_scores` 更可能成为有效的路由置信度。

### 2.3 DPR 返回 route_info 字典

原返回值：

```python
return sorted_x, idx_last, sorted_belong_idx, sorted_scores, prototypes
```

现返回值：

```python
route_info = {
    'belong_idx': belong_idx,
    'x_scores': x_scores,
    'scores': scores,
    'sorted_belong_idx': sorted_belong_idx,
    'sorted_scores': sorted_scores,
    'sorted_idx': sorted_idx,
}
return sorted_x, idx_last, prototypes, route_info
```

其中：

- `belong_idx` / `x_scores` / `scores` 是原始 token 顺序。
- `sorted_belong_idx` / `sorted_scores` 是排序后顺序。
- `sorted_idx` 保留排序索引，方便后续诊断。

---

## 3. IASA：confidence-aware prototype global attention

### 3.1 新增全局门控参数

在 `IASA.__init__()` 中新增：

```python
self.global_gate = nn.Parameter(torch.tensor(-1.0))
```

前向时使用 `sigmoid`，初始值约为 `0.27`，避免训练初期 prototype global attention 注入过强。

### 3.2 forward 接收 sorted_scores

原签名：

```python
def forward(self, sorted_x, idx_last, prototypes):
```

现签名：

```python
def forward(self, sorted_x, idx_last, prototypes, sorted_scores=None):
```

### 3.3 使用置信度调制 out2

原融合：

```python
out = out1 + out2
```

现融合：

```python
global_gate = torch.sigmoid(self.global_gate)
if sorted_scores is not None:
    ...
    score_gate = rearrange(paded_scores, "b (ng gs) -> b ng 1 gs 1", ng=ng, gs=gs)
    out = out1 + global_gate * score_gate * out2
else:
    out = out1 + global_gate * out2
```

这样高置信 token 可以更多使用 prototype global attention，低置信 token 会降低全局 prototype 注入强度。

---

## 4. TAB：低置信 token 的 soft prototype fallback

### 4.1 新增 soft context 分支参数

在 `TAB.__init__()` 中新增：

```python
self.soft_context_proj = nn.Linear(dim, dim, bias=False)
self.soft_fallback_gate = nn.Parameter(torch.tensor(-2.0))
```

`sigmoid(-2.0)` 初始约为 `0.12`，表示 fallback 初始较弱。

### 4.2 TAB.forward() 使用 route_info

原调用：

```python
sorted_x, idx_last, belong_idx, _, prototypes = self.dpr(x)
self.last_routing_map = belong_idx
Y = self.iasa_attn(sorted_x, idx_last, prototypes)
```

现调用：

```python
sorted_x, idx_last, prototypes, route_info = self.dpr(x)
self.last_routing_map = route_info['belong_idx'].detach()
self.last_sorted_routing_map = route_info['sorted_belong_idx'].detach()
hard_y = self.iasa_attn(sorted_x, idx_last, prototypes, route_info['sorted_scores'])
```

`last_routing_map` 现在保存原始空间顺序的路由标签，后续可视化不会再使用排序后的错位标签。

### 4.3 低置信 token soft fallback

新增逻辑：

```python
soft_context = torch.matmul(route_info['scores'], prototypes)
soft_context = self.soft_context_proj(soft_context)
low_conf = (1.0 - route_info['x_scores'].detach()).unsqueeze(-1)
y = hard_y + torch.sigmoid(self.soft_fallback_gate) * low_conf * soft_context
```

含义：

- `hard_y`：DPR hard routing 后经 IASA 的主路径输出。
- `soft_context`：原始顺序下的 soft prototype mixture。
- `low_conf`：低置信 token 获得更强 fallback。
- `.detach()`：防止模型通过刻意降低 `x_scores` 来过度使用 fallback。

---

## 5. 验证情况

已执行语法检查：

```bash
PYTHONPYCACHEPREFIX=/private/tmp/pycache_dpr python3 -m py_compile CATANet/basicsr/archs/catanet_arch.py
```

结果：通过。

尝试执行随机前向验证时，当前环境缺少 `torch`：

```text
ModuleNotFoundError: No module named 'torch'
```

因此本次只完成了静态语法验证。建议在安装好项目依赖的训练环境中继续运行：

```bash
cd CATANet
python - <<'PY'
import torch
from basicsr.archs.catanet_arch import CATANet
model = CATANet(upscale=2).cuda().eval()
x = torch.randn(1, 3, 64, 64).cuda()
with torch.no_grad():
    y = model(x)
print(y.shape)
PY
```

预期输出：

```text
torch.Size([1, 3, 128, 128])
```

---

## 6. 后续建议

1. 先跑短训或小 batch 前向，确认显存和速度变化。
2. 记录每个 TAB block 的 `route_info['x_scores']` 分布，确认可学习路由温度是否让置信度拉开。
3. 对比三组消融：
   - 只开 router temperature。
   - router temperature + IASA gate。
   - router temperature + IASA gate + soft fallback。
4. 若输出过平滑，可把 `soft_fallback_gate` 初始值从 `-2.0` 降到 `-3.0`。
5. 若训练不稳定，可把 `router_logit_scale` 初始值从 `log(10)` 降到 `log(6)`。

---

## 7. 二次协调性检查补充

再次检查全网调用后，补充修复了两个与整体网络协调相关的问题，修改文件：

```text
CATANet/basicsr/models/catanet_model.py
```

### 7.1 修复 self-ensemble 非 EMA 分支

原代码在没有 `net_g_ema` 时仍调用 `self.net_g_ema`：

```python
out_list = [self.net_g_ema(aug) for aug in lq_list]
```

已修复为：

```python
out_list = [self.net_g(aug) for aug in lq_list]
```

### 7.2 修复 cluster map 保存时访问已删除 self.lq 的问题

原验证流程先删除 `self.lq`，后续保存 DPR cluster map 时又访问 `self.lq.shape`。已在删除前缓存 shape：

```python
lq_shape = self.lq.shape
```

后续改用：

```python
_, _, h_lq, w_lq = lq_shape
```

### 7.3 当前协调性结论

已检查 `DPR -> TAB -> IASA -> CATANet -> CATANetModel` 的接口调用链：

```text
DPR.forward()
-> returns sorted_x, idx_last, prototypes, route_info
TAB.forward()
-> consumes route_info['sorted_scores'], route_info['scores'], route_info['x_scores']
IASA.forward()
-> accepts sorted_scores=None as backward-compatible optional input
CATANet.forward_features()
-> TAB output remains B x C x H x W, so block-level接口不变
CATANetModel validation
-> last_routing_map now uses original token order for visualization
```

静态语法检查通过：

```bash
PYTHONPYCACHEPREFIX=/private/tmp/pycache_dpr python3 -m py_compile \
  CATANet/basicsr/archs/catanet_arch.py \
  CATANet/basicsr/models/catanet_model.py
```

注意：由于新增了 `router_logit_scale`、`global_gate`、`soft_context_proj`、`soft_fallback_gate` 等参数，旧 CATANet/DPR checkpoint 使用 `strict_load_g: true` 直接加载会出现 missing keys。若要基于旧权重 finetune，需要将对应配置改为：

```yaml
strict_load_g: false
```

或编写 checkpoint 兼容加载逻辑。

---

## 8. x_scores mean/std 训练日志

已新增每个 TAB block 的路由置信度统计，用于观察可学习路由温度是否把 token-to-prototype 置信度拉开。

### 8.1 TAB 中缓存统计

在 `TAB.forward()` 中，每次 DPR 前向后缓存：

```python
with torch.no_grad():
    x_scores = route_info['x_scores'].detach()
    self.last_x_scores_mean = x_scores.mean()
    self.last_x_scores_std = x_scores.std(unbiased=False)
```

### 8.2 训练日志中输出

在 `CATANetModel.optimize_parameters()` 中，将每个 TAB 的统计加入 `loss_dict`：

```python
loss_dict[f'route_b{i}_xscore_mean'] = tab.last_x_scores_mean
loss_dict[f'route_b{i}_xscore_std'] = tab.last_x_scores_std
```

训练时会随已有 logger 输出，例如：

```text
route_b0_xscore_mean
route_b0_xscore_std
...
route_b7_xscore_mean
route_b7_xscore_std
```

### 8.3 如何判断

当前默认 `num_tokens` 为：

```text
[16, 32, 64, 128, 16, 32, 64, 128]
```

均匀 softmax 的基线约为：

```text
block 0/4: 1/16  = 0.0625
block 1/5: 1/32  = 0.03125
block 2/6: 1/64  = 0.015625
block 3/7: 1/128 = 0.0078125
```

如果 `route_b{i}_xscore_mean` 明显高于对应 `1/M`，且 `route_b{i}_xscore_std` 不接近 0，说明路由置信度被拉开。若 mean 长期接近 `1/M` 且 std 很小，说明 routing 仍接近均匀，需要调整 `router_logit_scale` 或训练策略。
