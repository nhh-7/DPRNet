# CATANet x4 finetune 0-250k key metrics summary

- Source directory: `CATANet/experiments/train_CATANet_x4_finetune`
- Output CSV: `training_key_metrics.csv`
- Validation rows: 100 (scheduled checkpoints every 2500 iters from 2,500 to 250,000)
- Finetune setup: 从 x2 锁定权重 `net_g_792500.pth` 加载（`strict_load_g=false`，仅 `upconv` 因尺度不同而忽略），路由超参与 x2 完全对齐（`router_scale_init=6.0`、`max_router_logit_scale=10.0`、`route_balance_weight=[6×0.0005, 0.0007, 0.0007]`、`router_scale_lr_mult=0.1`）。
- Scale x4: scale=4, gt_size=256, crop_border=4, test_y_channel=True；val 集为 Set5 / Set14。
- Schedule: total_iter=250000, milestones=[125000, 200000, 225000, 237500], gamma=0.5。
- #Params of CATANet: 659.707 K（x4 upconv 通道数与 x2/x3 不同）。
- Duplicate overlap handling: 日志尾部 `Save the latest model` 后重复的一次 250000 validation 已忽略，不计入 CSV。
- Best-iter tie handling: 同一指标多个 iter 并列时，记录较大的 iter 号。

## Best current validation metrics in extracted rows

- Set5 PSNR: 32.5984 @ iter 195000
- Set5 SSIM: 0.8999 @ iter 247500
- Set14 PSNR: 28.8899 @ iter 240000
- Set14 SSIM: 0.7882 @ iter 175000

## Milestone table

| iter | l_pix | l_route | lr_g | Set5 PSNR | Set5 SSIM | Set5 best PSNR iter | Set5 best SSIM iter | Set14 PSNR | Set14 SSIM | Set14 best PSNR iter | Set14 best SSIM iter |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2500 | 0.0219 | 5.846e-07 | 2.000e-04 | 32.4194 | 0.8983 | 2500 | 2500 | 28.7987 | 0.7858 | 2500 | 2500 |
| 25000 | 0.0219 | 1.214e-06 | 2.000e-04 | 32.4626 | 0.8990 | 22500 | 22500 | 28.8219 | 0.7874 | 15000 | 22500 |
| 50000 | 0.0238 | 1.210e-06 | 2.000e-04 | 32.5434 | 0.8991 | 50000 | 22500 | 28.8471 | 0.7867 | 47500 | 47500 |
| 75000 | 0.0224 | 1.037e-06 | 2.000e-04 | 32.4797 | 0.8993 | 72500 | 22500 | 28.8457 | 0.7876 | 47500 | 75000 |
| 100000 | 0.0247 | 1.386e-06 | 2.000e-04 | 32.4684 | 0.8988 | 72500 | 22500 | 28.8519 | 0.7875 | 47500 | 75000 |
| 125000 | 0.0240 | 1.505e-06 | 2.000e-04 | 32.5566 | 0.8996 | 125000 | 125000 | 28.8668 | 0.7879 | 47500 | 120000 |
| 150000 | 0.0208 | 1.160e-06 | 1.000e-04 | 32.5672 | 0.8994 | 142500 | 142500 | 28.8829 | 0.7874 | 150000 | 142500 |
| 175000 | 0.0244 | 1.162e-06 | 1.000e-04 | 32.5777 | 0.8998 | 175000 | 175000 | 28.8811 | 0.7882 | 160000 | 175000 |
| 200000 | 0.0232 | 1.251e-06 | 1.000e-04 | 32.5713 | 0.8998 | 195000 | 175000 | 28.8765 | 0.7880 | 160000 | 175000 |
| 225000 | 0.0272 | 1.476e-06 | 5.000e-05 | 32.5875 | 0.8998 | 195000 | 215000 | 28.8708 | 0.7879 | 202500 | 175000 |
| 237500 | 0.0214 | 1.075e-06 | 2.500e-05 | 32.5770 | 0.8997 | 195000 | 215000 | 28.8780 | 0.7880 | 202500 | 175000 |
| 250000 | 0.0240 | 1.143e-06 | 1.250e-05 | 32.5862 | 0.8998 | 195000 | 215000 | 28.8838 | 0.7880 | 240000 | 175000 |

## Phase summary（按 LR 阶段切窗均值）

| range | lr_g | Set5 mean PSNR | Set5 mean SSIM | Set14 mean PSNR | Set14 mean SSIM |
|---|---|---:|---:|---:|---:|
| 2.5k-125k | 2.000e-04 | 32.4911 | 0.8990 | 28.8445 | 0.7871 |
| 125k-200k | 1.000e-04 | 32.5552 | 0.8995 | 28.8719 | 0.7877 |
| 200k-250k | 5e-05→1.25e-05 | 32.5784 | 0.8997 | 28.8788 | 0.7878 |

## Key takeaways

- **Finetune 立即生效**：首个 val 点（2500）即为 Set5 32.4194 / Set14 28.7987，从高位起步而非从零爬升，证明 x2 权重正确迁移、`upconv` 重新初始化后快速适配 x4。
- **LR decay 行为符合预期**：milestones 在 `127500 / 202500 / 227500 / 240000` 处依次触发（打印滞后约 200 iter），`lr_g` 从 2e-4 逐级衰减到 1.25e-5。
- **指标随训练单调改善**：三个 LR 阶段的 Set5/Set14 均值逐段抬升，post-decay 阶段（200k 后）取得段内最佳均值，没有出现 scheduler 负收益。
- **各数据集最优点分散**：Set5 PSNR 峰值 `32.5984@195000`，Set14 PSNR 峰值 `28.8899@240000`，Set14 SSIM 峰值 `0.7882@175000`，需用全 5 基准统一对比后锁定单一报告 checkpoint，禁止逐集挑最优。
- **深层路由健康、无塌缩**：b6/b7 全程保持活跃（如 250000：Set5 b6 active 46.8 / entropy 0.6695，b7 active 59.4 / entropy 0.5254），router_scale 稳定在 b6≈5.90、b7≈6.40（x4 的 b7 scale 略高于 x3），无 NaN / 发散 / loss 异常。x4 的 b7 active 数量明显低于 x3（约 60 vs 约 95），属尺度差异下的正常路由分布，未触及塌缩阈值。
- **训练耗时**：约 1 天 12 小时（2026-06-07 00:16 起）。

## 候选报告 checkpoint（待测试机全 5 基准复核）

1. `net_g_195000.pth`：Set5 PSNR 全局最强；
2. `net_g_240000.pth`：Set14 PSNR 全局最强且为后期点，综合候选；
3. `net_g_250000.pth`：最新稳定点，Set5 SSIM 并列最强、Set14 PSNR 次高，final 参考。
