# CATANet x3 finetune 日志分析交接记录（供后续 AI 继续追加）

实验目录：`CATANet/experiments/train_CATANet_x3_finetune`

> 用途：
> - 这份文件用于给后续 AI 快速接手 x3 finetune 实验。
> - 默认按“追加记录”方式维护，不要覆盖历史结论。
> - 后续如果有新日志分析，直接在文末追加一个新小节即可。

---

## 1. 实验背景与总原则

这是 DPRNet / CATANet x3 超分实验，路线是**从 x2 长程 scratch 的锁定权重 finetune 到 x3**，而不是从头训练。

- 实验目录：`CATANet/experiments/train_CATANet_x3_finetune`
- 预训练来源：`experiments/train_CATANet_x2_scratch/models/net_g_792500.pth`
- `strict_load_g=false`：因 x3 的 `upconv` 输出通道（360）与 x2（160）不同，加载时仅忽略 `upconv.weight / upconv.bias`，其余权重全部迁移。

总原则（延续 x2）：

1. 路由超参与 x2 完全对齐，不在 x3 阶段再调路由配置；
2. 重点盯深层 `b6 / b7`，确认 finetune 没有引发新的 router collapse；
3. 先观察 milestone 之后的真实 post-decay 行为，再讨论是否要改 scheduler；
4. 报告 checkpoint 必须用全 5 基准统一对比后定单点，禁止逐数据集挑最优。

---

## 2. 关键配置（来自训练日志头）

- scale: 3, gt_size: 192, crop_border: 3, test_y_channel: True
- network_g: CATANet, upscale=3, router_scale_init=6.0, max_router_logit_scale=10.0,
  route_balance_weight=[0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0007, 0.0007]
- train: optim Adam lr=2e-4, router_scale_lr_mult=0.1, MultiStepLR milestones=[125000, 200000, 225000, 237500], gamma=0.5
- total_iter: 250000, val_freq: 2500, batch_size_per_gpu: 16, num_gpu: 4
- #Params of CATANet: 674.147 K

---

## 3. 主时间线（0 -> 250k，单段完整跑完）

本实验为一次性完整训练（无中途 resume），日志：`train_train_CATANet_x3_finetune_20260607_000319.log`，
耗时约 1 天 12 小时（2026-06-07 00:03 -> 06-08 12:13）。

### 3.1 起步（2500）

- finetune 立即生效：Set5 `34.6203 / 0.9290`，Set14 `30.5774 / 0.8462`，从高位起步。
- 起点路由（2500）：
  - Set5 `b6`: active 49.2, usage_max 0.2638, entropy 0.5878, scale 5.8159
  - Set5 `b7`: active 76.4, usage_max 0.2190, entropy 0.5751, scale 6.1235
- 判断：x2 权重正确迁移，`upconv` 重新初始化后快速适配 x3。

### 3.2 高 LR 阶段（2.5k -> 125k，lr_g=2e-4）

- 段均值：Set5 `34.6577 / 0.9294`，Set14 `30.6001 / 0.8470`。
- Set5 PSNR 在 `82500` 出现早期高点，Set14 缓步抬升。

### 3.3 第一次 decay 后（125k -> 200k，lr_g=1e-4）

- `127500` 起 lr_g 切到 1e-4（milestone 125000，打印滞后约 200 iter）。
- 段均值：Set5 `34.6824 / 0.9295`，Set14 `30.6401 / 0.8477`，整体抬升。
- 125000 路由：Set5 b6 active 52.4 / entropy 0.6687，b7 active 85.6 / entropy 0.5733，健康。

### 3.4 后段连续 decay（200k -> 250k，lr_g 5e-5 -> 1.25e-5）

- decay 点：`202500`(5e-5) -> `227500`(2.5e-5) -> `240000`(1.25e-5)。
- 段均值：Set5 `34.7053 / 0.9297`，Set14 `30.6531 / 0.8480`，为三段最佳均值。
- Set5 PSNR 全局峰值 `34.7165 @ 210000`；Set14 PSNR 全局峰值 `30.6623 @ 250000`。
- 250000 路由：Set5 b6 active 50.6 / entropy 0.6772，b7 active 97.6 / entropy 0.6071，router_scale b6≈5.86 / b7≈6.19。

---

## 4. 总判断（截至 250k）

1. **x2 -> x3 finetune 成功**：权重迁移生效，全程指标单调改善，无 NaN / 发散。
2. **路由未塌缩**：深层 b6/b7 一路健康，router_scale 受控，沿用 x2 路由配置在 x3 依然成立。
3. **scheduler 正常**：四个 milestone decay 均按预期触发，post-decay 取得最佳均值。
4. **最优点分散**：Set5 在 `210000`、Set14 在 `250000`，需全 5 基准统一定点。

---

## 5. 下一步建议

- **定报告 checkpoint**：在测试机用全 5 基准（Set5/Set14/BSD100/Urban100/Manga109）对若干末段 checkpoint 统一评测，锁定单一报告点。优先候选：
  1. `net_g_210000.pth`（Set5 PSNR 最强）
  2. `net_g_250000.pth`（Set14 综合最强、最新稳定点）
  3. `net_g_232500.pth`（Set14 PSNR 后期高点 30.6608）
- 效率指标用 `CATANet/scripts/measure_efficiency.py --scale 3` 测量后回填 Table II。
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
