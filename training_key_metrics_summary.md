# CATANet x2 scratch 0-550k key metrics summary

- Source directory: `CATANet/experiments/train_CATANet_x2_scratch`
- Output CSV: `training_key_metrics.csv`
- Validation rows: 220 (scheduled checkpoints every 2500 iters from 2,500 to 550,000)
- Duplicate overlap handling: later log files override earlier interrupted/restarted runs for the same iteration. Final `Save the latest model` duplicate validations are ignored.
- Best-iter tie handling: when multiple iterations have the same best metric value, the larger iteration number is recorded.
- File rename note: the old `training_key_metrics_0_400k_summary.md` / `training_key_metrics_0_400k.csv` pair has been superseded by `training_key_metrics_summary.md` / `training_key_metrics.csv` so future updates can append beyond 400k.

## Best current validation metrics in extracted rows

- Set5 PSNR: 38.2649 @ iter 530000
- Set5 SSIM: 0.9616 @ iter 550000
- Set14 PSNR: 34.0286 @ iter 527500
- Set14 SSIM: 0.9216 @ iter 500000

## Milestone table

| iter | l_pix | l_route | lr_g | Set5 PSNR | Set5 SSIM | Set5 best PSNR iter | Set5 best SSIM iter | Set14 PSNR | Set14 SSIM | Set14 best PSNR iter | Set14 best SSIM iter |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2500 | 0.0144 | 5.215e-11 | 0.0002 | 36.9586 | 0.9563 | 2500 | 2500 | 32.5673 | 0.9084 | 2500 | 2500 |
| 50000 | 0.0140 | 1.689e-07 | 0.0002 | 38.0297 | 0.9607 | 50000 | 50000 | 33.5698 | 0.9182 | 45000 | 50000 |
| 100000 | 0.0106 | 3.148e-07 | 0.0002 | 38.0455 | 0.9607 | 92500 | 80000 | 33.7393 | 0.9188 | 90000 | 90000 |
| 150000 | 0.0131 | 7.484e-07 | 0.0002 | 38.1670 | 0.9613 | 135000 | 150000 | 33.7492 | 0.9191 | 127500 | 127500 |
| 200000 | 0.0129 | 4.922e-07 | 0.0002 | 38.1724 | 0.9614 | 197500 | 197500 | 33.7916 | 0.9196 | 127500 | 195000 |
| 250000 | 0.0137 | 6.852e-07 | 0.0002 | 38.1888 | 0.9614 | 235000 | 242500 | 33.8566 | 0.9203 | 220000 | 220000 |
| 300000 | 0.0116 | 6.292e-07 | 0.0002 | 38.1723 | 0.9615 | 287500 | 280000 | 33.9140 | 0.9210 | 300000 | 255000 |
| 350000 | 0.0112 | 5.054e-07 | 0.0001 | 38.2214 | 0.9615 | 325000 | 325000 | 33.9282 | 0.9206 | 345000 | 345000 |
| 400000 | 0.0115 | 6.415e-07 | 0.0001 | 38.2424 | 0.9615 | 392500 | 397500 | 33.9650 | 0.9207 | 392500 | 345000 |
| 450000 | 0.0141 | 6.614e-07 | 1.000e-04 | 38.2045 | 0.9614 | 445000 | 447500 | 33.9521 | 0.9206 | 410000 | 410000 |
| 500000 | 0.0108 | 5.160e-07 | 1.000e-04 | 38.2285 | 0.9616 | 497500 | 500000 | 34.0229 | 0.9216 | 500000 | 500000 |
| 550000 | 0.0149 | 8.346e-07 | 5.000e-05 | 38.2582 | 0.9616 | 530000 | 547500 | 33.9606 | 0.9207 | 527500 | 500000 |

## 400k-550k phase summary

| range | rows | lr_g values | Set5 mean PSNR | Set5 mean SSIM | Set14 mean PSNR | Set14 mean SSIM | Set5 best PSNR | Set14 best PSNR |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 402.5k-447.5k | 19 | 1.000e-04 | 38.2338 | 0.9615 | 33.9467 | 0.9208 | 38.2525 | 34.0046 |
| 450k-497.5k | 20 | 1.000e-04 | 38.2376 | 0.9615 | 33.9478 | 0.9207 | 38.2580 | 33.9927 |
| 500k-550k | 21 | 1.000e-04 / 5.000e-05 | 38.2536 | 0.9616 | 33.9827 | 0.9211 | 38.2649 | 34.0286 |

## Key takeaways from the new 400k-550k segment

- The 500k milestone decay triggered as expected: training prints still show `1.000e-04 / 1.000e-05` at `500000`, then switch to `5.000e-05 / 5.000e-06` immediately after `500000`.
- Set5 still improves slightly beyond 400k, with the current best PSNR at `530000`; SSIM stays at `0.9616` across several checkpoints, and tie handling records the latest best iter as `550000`.
- Set14 is strongest around `500000-527500`, then remains high but oscillatory through `550000`, which looks more like a marginal-gain plateau than a failure mode.
- Deep routers stay healthy: `b6 / b7` remain in the same non-collapsed regime seen at 300k-400k, and router scales remain stable around `~5.83` (b6) / `~6.08` (b7).
- Practical checkpoint picks from this segment are `500000`, `527500`, and `530000` rather than the final `550000` checkpoint.

