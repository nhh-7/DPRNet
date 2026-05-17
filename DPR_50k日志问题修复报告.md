# DPR 50k 训练日志问题修复报告

## 背景

本次修改基于 `CATANet/experiments/train_CATANet_x2_scratch/` 下 0~50k 训练日志暴露的问题进行：

- 若干 TAB/DPR block 出现路由塌缩，典型表现为 `usage_active` 很低、`usage_max` 接近 1、`entropy_norm` 接近 0。
- `router_scale` 在训练中持续增大，使 token-to-prototype softmax 越来越尖锐。
- DPR 缺少 prototype 使用率/路由熵约束，长训 800k 时塌缩风险会被放大。
- `router_logit_scale` 与普通网络参数使用相同学习率，温度参数更新过快。
- 多验证集 warning 对当前 `CATANetModel` 造成不必要的日志干扰。

## 修改目标

1. 降低 DPR router 初始尖锐程度。
2. 限制 router scale 最大值，防止长训中无约束放大。
3. 增加弱路由熵正则，抑制 prototype/route 塌缩。
4. 为 `router_logit_scale` 单独使用更小学习率。
5. 保持 800k 正式训练 scheduler 不变，仅修正网络与训练稳定性问题。
6. 修正当前模型已支持多验证集但仍输出 warning 的工程问题。

## 代码改动

### 1. `CATANet/basicsr/archs/catanet_arch.py`

#### 1.1 DPR 增加可配置 router scale

新增参数：

```python
router_scale_init=6.0
max_router_logit_scale=10.0
```

将原始初始化：

```python
self.router_logit_scale = nn.Parameter(torch.ones([]) * 2.302585093)  # log(10)
self.max_router_logit_scale = 50.0
```

改为：

```python
self.router_logit_scale = nn.Parameter(torch.log(torch.tensor(float(router_scale_init))))
self.max_router_logit_scale = float(max_router_logit_scale)
```

含义：

- 初始 scale 从 10 降到 6。
- 最大 scale 从 50 限制到 10。
- 减少 softmax 过早/过强尖锐化。

#### 1.2 DPR 增加路由熵辅助损失

新增参数：

```python
balance_loss_weight=0.0
```

在最终 token-to-prototype `scores = softmax(...)` 后统计使用率并计算熵正则：

```python
usage = scores.mean(dim=1).clamp_min(1e-8)
self.last_usage = usage.detach().mean(dim=0)
if self.training and self.balance_loss_weight > 0:
    entropy = -(usage * usage.log()).sum(dim=-1) / math.log(self.num_prototypes)
    self.aux_loss = self.balance_loss_weight * (1.0 - entropy.mean())
else:
    self.aux_loss = None
```

选择 entropy loss 而不是强制 uniform MSE 的原因：

- 目标是防止 `entropy_norm -> 0` 的塌缩，而不是强迫每个 prototype 完全均匀使用。
- 对超分任务更温和，降低正则损害重建细节的风险。

#### 1.3 TAB/CATANet 透传新参数

`TAB` 新增并传递：

```python
router_scale_init=6.0
max_router_logit_scale=10.0
route_balance_weight=0.0
```

`CATANet` 新增：

```python
router_scale_init=6.0
max_router_logit_scale=10.0
route_balance_weight=0.0
```

`route_balance_weight` 支持：

- 单个 float：所有 block 使用同一权重。
- 长度等于 `block_num` 的 list：不同 block 使用不同权重。

### 2. `CATANet/basicsr/models/catanet_model.py`

#### 2.1 汇总 DPR aux loss

在 `optimize_parameters()` 中，反向传播前收集所有模块的 `aux_loss`：

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

训练日志中会新增 `l_route`，用于确认路由正则是否生效。

#### 2.2 `router_logit_scale` 单独降学习率

新增训练参数：

```yaml
router_scale_lr_mult: 0.1
```

优化器参数分组：

- 普通参数使用原学习率。
- 参数名包含 `router_logit_scale` 的参数使用 `lr * router_scale_lr_mult`。

默认即：

```text
router_logit_scale lr = 2e-4 * 0.1 = 2e-5
```

目的：防止路由温度参数在长训中过快增大。

#### 2.3 标记 CATANetModel 支持多验证集

新增：

```python
self.support_multi_val = True
```

用于配合训练入口避免错误 warning。

### 3. `CATANet/basicsr/train.py`

将多验证集 warning 修改为仅在模型不声明支持时输出：

```python
if len(val_loaders) > 1 and not getattr(model, 'support_multi_val', False):
    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
```

### 4. `CATANet/options/train/train_CATANet_x2_scratch.yml`

新增网络参数：

```yaml
network_g:
  type: CATANet
  upscale: 2
  router_scale_init: 6.0
  max_router_logit_scale: 10.0
  route_balance_weight: 0.0005
```

新增训练参数：

```yaml
train:
  router_scale_lr_mult: 0.1
```

当前没有改动 800k scheduler；如果正式长训，仍可使用原 800k milestones。已将 `resume_state` 改为 `~`，避免结构新增参数后误 resume 旧 optimizer/state。若要从旧权重迁移，请使用 `pretrain_network_g` 并视 checkpoint 兼容情况设置 `strict_load_g: false`。

## 预期效果

修改后应重点观察：

1. `route_b{i}_router_scale` 不应长期顶到 10。
2. b4/b5/b6 的 `entropy_norm` 不应长期为 0。
3. b4/b5/b6 的 `usage_max` 不应长期接近 1。
4. b4 的 `usage_active` 不应长期保持 1。
5. 训练日志应出现 `l_route`，数值应较小且稳定。

## 建议的后续验证

### 快速 sanity run

建议先跑 2.5k~10k，确认：

- 训练能正常反向传播。
- 日志出现 `l_route`。
- 验证时不再出现多验证集 warning。
- route stats 正常输出。

### 50k/100k 对比实验

建议至少对比：

1. 当前修改版：`router_scale_init=6, max=10, route_balance_weight=0.0005`。
2. 若仍塌缩：`route_balance_weight=0.001`。
3. 若 PSNR 下降：`route_balance_weight=0.00025` 或关闭正则，仅保留 router scale 限制。

### 800k 正式训练监控阈值

若连续多个验证点出现以下情况，应提前停止并调整：

```text
usage_active <= 2
usage_max >= 0.95
entropy_norm <= 0.02
router_scale 接近 max_router_logit_scale
```

优先调整顺序：

1. `route_balance_weight: 0.0005 -> 0.001`
2. `max_router_logit_scale: 10 -> 8`
3. `router_scale_lr_mult: 0.1 -> 0.05`
4. 仍不稳定时再考虑减少后半段 `num_tokens`

## 已完成检查

已完成 Python 语法编译检查：

```bash
PYTHONPYCACHEPREFIX=/private/tmp/dprnet_pycache python3 -m compileall \
  CATANet/basicsr/archs/catanet_arch.py \
  CATANet/basicsr/models/catanet_model.py \
  CATANet/basicsr/train.py
```

本地环境未安装 `torch`，因此没有执行真实模型实例化和训练前向检查。另已完成一次复查修正：路由熵正则现在作用于最终 router `scores`，而不是初始 prototype aggregation 的 `assignment`，更贴近日志中 `belong_idx/usage` 暴露的问题。
