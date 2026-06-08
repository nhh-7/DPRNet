# CATANet x3 finetune 0-250k key metrics summary

- Source directory: `CATANet/experiments/train_CATANet_x3_finetune`
- Output CSV: `training_key_metrics.csv`
- Validation rows: 100 (scheduled checkpoints every 2500 iters from 2,500 to 250,000)
- Finetune setup: 从 x2 锁定权重 `net_g_792500.pth` 加载（`strict_load_g=false`，仅 `upconv` 因尺度不同而忽略），路由超参与 x2 完全对齐（`router_scale_init=6.0`、`max_router_logit_scale=10.0`、`route_balance_weight=[6×0.0005, 0.0007, 0.0007]`、`router_scale_lr_mult=0.1`）。
- Scale x3: scale=3, gt_size=192, crop_border=3, test_y_channel=True；val 集为 Set5 / Set14。
- Schedule: total_iter=250000, milestones=[125000, 200000, 225000, 237500], gamma=0.5。
- Duplicate overlap handling: 日志尾部 `Save the latest model` 后重复的一次 250000 validation（best iter 写成 `250001` 的 artifact）已忽略，不计入 CSV。
- Best-iter tie handling: 同一指标多个 iter 并列时，记录较大的 iter 号。

## Best current validation metrics in extracted rows

- Set5 PSNR: 34.7165 @ iter 210000
- Set5 SSIM: 0.9298 @ iter 250000（多点并列，按 latest-tie 记到 250000；首次出现于 82500）
- Set14 PSNR: 30.6623 @ iter 250000
- Set14 SSIM: 0.8482 @ iter 250000（与 230000 并列，按 latest-tie 记到 250000）

## Milestone table

| iter | l_pix | l_route | lr_g | Set5 PSNR | Set5 SSIM | Set5 best PSNR iter | Set5 best SSIM iter | Set14 PSNR | Set14 SSIM | Set14 best PSNR iter | Set14 best SSIM iter |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2500 | 0.0212 | 8.673e-07 | 2.000e-04 | 34.6203 | 0.9290 | 2500 | 2500 | 30.5774 | 0.8462 | 2500 | 2500 |
| 25000 | 0.0197 | 1.098e-06 | 2.000e-04 | 34.6150 | 0.9293 | 22500 | 5000 | 30.5631 | 0.8461 | 22500 | 22500 |
| 50000 | 0.0187 | 9.539e-07 | 2.000e-04 | 34.6452 | 0.9293 | 40000 | 40000 | 30.6212 | 0.8472 | 50000 | 37500 |
| 75000 | 0.0186 | 8.939e-07 | 2.000e-04 | 34.6501 | 0.9293 | 40000 | 40000 | 30.5983 | 0.8466 | 50000 | 65000 |
| 100000 | 0.0196 | 1.395e-06 | 2.000e-04 | 34.6813 | 0.9294 | 82500 | 82500 | 30.6180 | 0.8472 | 50000 | 65000 |
| 125000 | 0.0196 | 1.258e-06 | 2.000e-04 | 34.6567 | 0.9295 | 115000 | 82500 | 30.6160 | 0.8478 | 50000 | 125000 |
| 150000 | 0.0149 | 8.208e-07 | 1.000e-04 | 34.6754 | 0.9296 | 147500 | 82500 | 30.6302 | 0.8479 | 137500 | 145000 |
| 175000 | 0.0179 | 9.282e-07 | 1.000e-04 | 34.6901 | 0.9297 | 147500 | 82500 | 30.6313 | 0.8479 | 137500 | 145000 |
| 200000 | 0.0202 | 9.652e-07 | 1.000e-04 | 34.6934 | 0.9295 | 147500 | 82500 | 30.6459 | 0.8475 | 197500 | 145000 |
| 225000 | 0.0229 | 1.509e-06 | 5.000e-05 | 34.7130 | 0.9297 | 210000 | 82500 | 30.6553 | 0.8479 | 210000 | 145000 |
| 237500 | 0.0201 | 8.591e-07 | 2.500e-05 | 34.7009 | 0.9296 | 210000 | 82500 | 30.6585 | 0.8479 | 232500 | 230000 |
| 250000 | 0.0194 | 1.227e-06 | 1.250e-05 | 34.7138 | 0.9298 | 210000 | 82500 | 30.6623 | 0.8482 | 250000 | 230000 |

## Phase summary（按 LR 阶段切窗均值）

| range | lr_g | Set5 mean PSNR | Set5 mean SSIM | Set14 mean PSNR | Set14 mean SSIM |
|---|---|---:|---:|---:|---:|
| 2.5k-125k | 2.000e-04 | 34.6577 | 0.9294 | 30.6001 | 0.8470 |
| 125k-200k | 1.000e-04 | 34.6824 | 0.9295 | 30.6401 | 0.8477 |
| 200k-250k | 5e-05→1.25e-05 | 34.7053 | 0.9297 | 30.6531 | 0.8480 |

## Key takeaways

- **Finetune 立即生效**：首个 val 点（2500）即为 Set5 34.6203 / Set14 30.5774，从高位起步而非从零爬升，证明 x2 权重正确迁移、`upconv` 重新初始化后快速适配 x3。
- **LR decay 行为符合预期**：milestones 在 `127500 / 202500 / 227500 / 240000` 处依次触发（打印滞后约 200 iter），`lr_g` 从 2e-4 逐级衰减到 1.25e-5。
- **指标随训练单调改善**：三个 LR 阶段的 Set5/Set14 均值逐段抬升，post-decay 阶段（200k 后）取得段内最佳均值，没有出现 scheduler 负收益。
- **各数据集最优点分散**：Set5 PSNR 峰值 `34.7165@210000`，Set14 PSNR 峰值 `30.6623@250000`，需用全 5 基准统一对比后锁定单一报告 checkpoint，禁止逐集挑最优。
- **深层路由健康、无塌缩**：b6/b7 全程保持活跃（如 250000：Set5 b6 active 50.6 / entropy 0.6772，b7 active 97.6 / entropy 0.6071），router_scale 稳定在 b6≈5.86、b7≈6.19，无 NaN / 发散 / loss 异常。
- **训练耗时**：约 1 天 12 小时（2026-06-07 00:03 → 06-08 12:13）。

## 候选报告 checkpoint（待测试机全 5 基准复核）

1. `net_g_210000.pth`：Set5 PSNR 全局最强；
2. `net_g_250000.pth`：Set14 PSNR/SSIM 最强且为最新稳定点，综合候选；
3. `net_g_232500.pth`：Set14 PSNR 后期高点（30.6608），可作对照。
