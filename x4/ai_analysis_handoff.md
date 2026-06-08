# CATANet x4 finetune 日志分析交接记录（供后续 AI 继续追加）

实验目录：`CATANet/experiments/train_CATANet_x4_finetune`

> 用途：
> - 这份文件用于给后续 AI 快速接手 x4 finetune 实验。
> - 默认按“追加记录”方式维护，不要覆盖历史结论。
> - 后续如果有新日志分析，直接在文末追加一个新小节即可。

---

## 1. 实验背景与总原则

这是 DPRNet / CATANet x4 超分实验，路线是**从 x2 长程 scratch 的锁定权重 finetune 到 x4**，而不是从头训练。

- 实验目录：`CATANet/experiments/train_CATANet_x4_finetune`
- 预训练来源：`experiments/train_CATANet_x2_scratch/models/net_g_792500.pth`
- `strict_load_g=false`：因 x4 的 `upconv` 通道与 x2 不同，加载时仅忽略 `upconv.weight / upconv.bias`，其余权重全部迁移。

总原则（延续 x2）：

1. 路由超参与 x2 完全对齐，不在 x4 阶段再调路由配置；
2. 重点盯深层 `b6 / b7`，确认 finetune 没有引发新的 router collapse；
3. 先观察 milestone 之后的真实 post-decay 行为，再讨论是否要改 scheduler；
4. 报告 checkpoint 必须用全 5 基准统一对比后定单点，禁止逐数据集挑最优。

---

## 2. 关键配置（来自训练日志头）

- scale: 4, gt_size: 256, crop_border: 4, test_y_channel: True
- network_g: CATANet, upscale=4, router_scale_init=6.0, max_router_logit_scale=10.0,
  route_balance_weight=[0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0007, 0.0007]
- train: optim Adam lr=2e-4, router_scale_lr_mult=0.1, MultiStepLR milestones=[125000, 200000, 225000, 237500], gamma=0.5
- total_iter: 250000, val_freq: 2500, batch_size_per_gpu: 16, num_gpu: 4
- #Params of CATANet: 659.707 K

---

## 3. 主时间线（0 -> 250k，单段完整跑完）

本实验为一次性完整训练（无中途 resume），日志：`train_train_CATANet_x4_finetune_20260607_001632.log`，
耗时约 1 天 12 小时（起于 2026-06-07 00:16）。

### 3.1 起步（2500）

- finetune 立即生效：Set5 `32.4194 / 0.8983`，Set14 `28.7987 / 0.7858`，从高位起步。
- 起点路由（2500）：
  - Set5 `b6`: active 45.4, usage_max 0.2329, entropy 0.6028, scale 5.8178
  - Set5 `b7`: active 89.8, usage_max 0.2025, entropy 0.6109, scale 6.1276
- 判断：x2 权重正确迁移，`upconv` 重新初始化后快速适配 x4。

### 3.2 高 LR 阶段（2.5k -> 125k，lr_g=2e-4）

- 段均值：Set5 `32.4911 / 0.8990`，Set14 `28.8445 / 0.7871`。
- Set5 在 `50000` 附近先冲高（32.54），随后高 LR 期有震荡，属正常。

### 3.3 第一次 decay 后（125k -> 200k，lr_g=1e-4）

- `127500` 起 lr_g 切到 1e-4（milestone 125000，打印滞后约 200 iter）。
- 段均值：Set5 `32.5552 / 0.8995`，Set14 `28.8719 / 0.7877`，整体抬升。
- 125000 路由：Set5 b6 active 44.2 / entropy 0.6468，b7 active 62.2 / entropy 0.5460。
  注意 x4 的 b7 active（约 60）明显低于 x3（约 95），属尺度差异下的路由分布特征，未触及塌缩。

### 3.4 后段连续 decay（200k -> 250k，lr_g 5e-5 -> 1.25e-5）

- decay 点：`202500`(5e-5) -> `227500`(2.5e-5) -> `240000`(1.25e-5)。
- 段均值：Set5 `32.5784 / 0.8997`，Set14 `28.8788 / 0.7878`，为三段最佳均值。
- Set5 PSNR 全局峰值 `32.5984 @ 195000`；Set14 PSNR 全局峰值 `28.8899 @ 240000`；Set14 SSIM 峰值 `0.7882 @ 175000`。
- 250000 路由：Set5 b6 active 46.8 / entropy 0.6695，b7 active 59.4 / entropy 0.5254，router_scale b6≈5.90 / b7≈6.40。

---

## 4. 总判断（截至 250k）

1. **x2 -> x4 finetune 成功**：权重迁移生效，全程指标单调改善，无 NaN / 发散。
2. **路由未塌缩**：深层 b6/b7 一路健康，router_scale 受控；x4 的 b7 active 数量偏低但 entropy 正常，属尺度特征而非塌缩。
3. **scheduler 正常**：四个 milestone decay 均按预期触发，post-decay 取得最佳均值。
4. **最优点分散**：Set5 在 `195000`、Set14 PSNR 在 `240000`、Set14 SSIM 在 `175000`，需全 5 基准统一定点。

---

## 5. 下一步建议

- **定报告 checkpoint**：在测试机用全 5 基准（Set5/Set14/BSD100/Urban100/Manga109）对若干末段 checkpoint 统一评测，锁定单一报告点。优先候选：
  1. `net_g_195000.pth`（Set5 PSNR 最强）
  2. `net_g_240000.pth`（Set14 PSNR 最强、后期点）
  3. `net_g_250000.pth`（最新稳定点，Set5 SSIM 并列最强）
- 效率指标用 `CATANet/scripts/measure_efficiency.py --scale 4` 测量后回填 Table II。
- 不建议再延长训练或改动路由 / scheduler 配置，边际收益已有限。

---

## 6. 后续追加模板（直接复制到文末）

```md
---

## Update YYYY-MM-DD / XXXk -> YYYk

### 日志 / checkpoint
- 日志：
- model：
- state：
- resume_state：
- total_iter：
- milestones：
- route_balance_weight：

### 关键观察
- LR 是否变化：
- Set5 best / final：
- Set14 best / final：
- b6 状态：
- b7 状态：
- 是否出现新问题：

### 结论
-

### 下一步建议
-
```
