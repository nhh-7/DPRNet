# CATANet x2 scratch 0-800k key metrics summary

- Source directory: `CATANet/experiments/train_CATANet_x2_scratch`
- Output CSV: `training_key_metrics.csv`
- Validation rows: 320 (scheduled checkpoints every 2500 iters from 2,500 to 800,000)
- Duplicate overlap handling: later log files override earlier interrupted/restarted runs for the same iteration. Final `Save the latest model` duplicate validations are ignored.
- Best-iter tie handling: when multiple iterations have the same best metric value, the larger iteration number is recorded.
- File rename note: the old `training_key_metrics_0_400k_summary.md` / `training_key_metrics_0_400k.csv` pair has been superseded by `training_key_metrics_summary.md` / `training_key_metrics.csv` so future updates can append beyond 400k.

## Best current validation metrics in extracted rows

- Set5 PSNR: 38.2763 @ iter 777500
- Set5 SSIM: 0.9617 @ iter 792500
- Set14 PSNR: 34.0425 @ iter 590000
- Set14 SSIM: 0.9216 @ iter 590000

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
| 600000 | 0.0136 | 7.250e-07 | 5.000e-05 | 38.2573 | 0.9615 | 585000 | 590000 | 34.0036 | 0.9212 | 590000 | 590000 |
| 650000 | 0.0113 | 4.379e-07 | 5.000e-05 | 38.2649 | 0.9617 | 585000 | 650000 | 33.9341 | 0.9209 | 590000 | 590000 |
| 675000 | 0.0138 | 6.979e-07 | 2.500e-05 | 38.2624 | 0.9616 | 585000 | 650000 | 33.9680 | 0.9208 | 590000 | 590000 |
| 700000 | 0.0131 | 5.445e-07 | 2.500e-05 | 38.2636 | 0.9616 | 585000 | 685000 | 33.9512 | 0.9209 | 590000 | 590000 |
| 750000 | 0.0136 | 5.570e-07 | 1.250e-05 | 38.2659 | 0.9617 | 585000 | 750000 | 33.9515 | 0.9209 | 590000 | 590000 |
| 800000 | 0.0111 | 4.821e-07 | 6.250e-06 | 38.2684 | 0.9616 | 777500 | 792500 | 33.9663 | 0.9209 | 590000 | 590000 |

## 750k-800k phase summary

| range | rows | lr_g values | Set5 mean PSNR | Set5 mean SSIM | Set14 mean PSNR | Set14 mean SSIM | Set5 best PSNR | Set14 best PSNR |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 752.5k-775k | 10 | 6.250e-06 | 38.2673 | 0.9616 | 33.9612 | 0.9209 | 38.2695 | 33.9747 |
| 777.5k-800k | 10 | 6.250e-06 | 38.2685 | 0.9616 | 33.9568 | 0.9209 | 38.2763 | 33.9804 |
| 752.5k-800k | 20 | 6.250e-06 | 38.2679 | 0.9616 | 33.9590 | 0.9209 | 38.2763 | 33.9804 |

## Key takeaways from the new 750k-800k segment

- Training resumed from `750000.state` / `net_g_750000.pth` and ran to `800000` (the originally planned long-range target) with the same `route_balance_weight` and milestones.
- The `750000` milestone decay behaved as expected: this run starts directly at `6.250e-06 / 6.250e-07` (i.e. the post-`750k` decayed LR), and that LR holds for the whole segment since `750000` is the last milestone.
- A new global PSNR best was reached in this segment: Set5 PSNR `38.2763 @ 777500`, surpassing the earlier global best `38.2742 @ 585000`. Set5 also produced a near-tie point `38.2742 @ 785000`.
- Set5 SSIM tied the global best `0.9617` again at `792500`; under latest-tie handling the global Set5 SSIM best iter now moves to `792500`.
- Set14 did not set new bests: segment-best Set14 PSNR is `33.9804 @ 797500` (below global `34.0425 @ 590000`), and Set14 SSIM peaked at `0.9210`, below the global best `0.9216 @ 590000`.
- The final `800000` checkpoint is stable but not the best point: Set5 `38.2684 / 0.9616`, Set14 `33.9663 / 0.9209`.
- Deep routers remain healthy. At `800000`, Set5 `b6/b7` are active `53.2 / 89.6` with entropy `0.6261 / 0.5874`; Set14 `b6/b7` are active `54.5 / 104.9286` with entropy `0.5491 / 0.5985`. Router scales stay bounded around `5.82` for `b6` and `6.13` for `b7`.
- Practical checkpoint picks remain `590000` for overall/Set14 balance, `777500` as the new global Set5 PSNR best, `792500` for Set5 SSIM, and `650000`/`750000` as earlier Set5 SSIM ties.

## 675k-750k phase summary

| range | rows | lr_g values | Set5 mean PSNR | Set5 mean SSIM | Set14 mean PSNR | Set14 mean SSIM | Set5 best PSNR | Set14 best PSNR |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 677.5k-697.5k | 9 | 2.500e-05 | 38.2618 | 0.9616 | 33.9636 | 0.9209 | 38.2711 | 33.9749 |
| 700k-750k | 21 | 2.500e-05 / 1.250e-05 | 38.2680 | 0.9616 | 33.9656 | 0.9209 | 38.2720 | 33.9839 |
| 702.5k-750k | 20 | 1.250e-05 | 38.2682 | 0.9616 | 33.9663 | 0.9209 | 38.2720 | 33.9839 |

## Key takeaways from the new 675k-750k segment

- Training resumed from `675000.state` / `net_g_675000.pth` and ran to `750000` with the same `route_balance_weight` and milestones.
- The `700000` milestone decay behaved as expected: `700000` itself still prints `2.500e-05 / 2.500e-06`, then `700200` and later rows switch to `1.250e-05 / 1.250e-06`.
- No new global PSNR best was reached in this segment. The best segment points are Set5 PSNR `38.2720 @ 735000` and Set14 PSNR `33.9839 @ 715000`, both below earlier global bests.
- Set5 SSIM tied the global best and, under latest-tie handling, now records `0.9617 @ 750000`. Set14 SSIM peaked at `0.9211 @ 710000`, below the earlier global best `0.9216 @ 590000`.
- The final `750000` checkpoint is stable but not the best PSNR point: Set5 `38.2659 / 0.9617`, Set14 `33.9515 / 0.9209`.
- Deep routers remain healthy. At `750000`, Set5 `b6/b7` are active `52.0 / 89.8` with entropy `0.6171 / 0.5958`; Set14 `b6/b7` are active `54.2857 / 106.1429` with entropy `0.5553 / 0.6025`. Router scales stay bounded around `5.82` for `b6` and `6.12` for `b7`.
- Practical checkpoint picks remain `590000` for overall/Set14 balance, `585000` for Set5 PSNR, `650000` or `750000` if prioritizing Set5 SSIM, and `735000` as the best late-stage Set5 PSNR point.

## 400k-550k phase summary

| range | rows | lr_g values | Set5 mean PSNR | Set5 mean SSIM | Set14 mean PSNR | Set14 mean SSIM | Set5 best PSNR | Set14 best PSNR |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 402.5k-447.5k | 19 | 1.000e-04 | 38.2338 | 0.9615 | 33.9467 | 0.9208 | 38.2525 | 34.0046 |
| 450k-497.5k | 20 | 1.000e-04 | 38.2376 | 0.9615 | 33.9478 | 0.9207 | 38.2580 | 33.9927 |
| 500k-550k | 21 | 1.000e-04 / 5.000e-05 | 38.2536 | 0.9616 | 33.9827 | 0.9211 | 38.2649 | 34.0286 |

## 550k-675k phase summary

| range | rows | lr_g values | Set5 mean PSNR | Set5 mean SSIM | Set14 mean PSNR | Set14 mean SSIM | Set5 best PSNR | Set14 best PSNR |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 552.5k-597.5k | 19 | 5.000e-05 | 38.2581 | 0.9616 | 33.9675 | 0.9209 | 38.2742 | 34.0425 |
| 600k-647.5k | 20 | 5.000e-05 | 38.2553 | 0.9616 | 33.9711 | 0.9210 | 38.2727 | 34.0262 |
| 650k-675k | 11 | 5.000e-05 / 2.500e-05 | 38.2641 | 0.9616 | 33.9689 | 0.9209 | 38.2712 | 33.9984 |

## Key takeaways from the new 550k-675k segment

- Training resumed from `550000.state` / `net_g_550000.pth` and ran to `675000` with the same `route_balance_weight` and milestones.
- The `650000` milestone decay behaved as expected: `650000` itself still prints `5.000e-05 / 5.000e-06`, then subsequent rows switch to `2.500e-05 / 2.500e-06`.
- New global bests were reached in this segment: Set5 PSNR `38.2742 @ 585000`, Set5 SSIM `0.9617 @ 650000`, Set14 PSNR/SSIM `34.0425 / 0.9216 @ 590000`.
- The final `675000` checkpoint is stable but not best: Set5 `38.2624 / 0.9616`, Set14 `33.9680 / 0.9208`.
- Deep routers remain healthy. At `675000`, Set5 `b6/b7` are active `52.4 / 87.6` with entropy `0.6131 / 0.5747`; Set14 `b6/b7` are active `54.7857 / 102.8571` with entropy `0.5381 / 0.5986`. Router scales stay bounded around `5.82` for `b6` and `6.12` for `b7`.
- Practical checkpoint picks from this segment are `590000` for Set14/general balance, `585000` for Set5 PSNR, and `650000` if prioritizing Set5 SSIM/post-milestone state.

## Key takeaways from the new 400k-550k segment

- The 500k milestone decay triggered as expected: training prints still show `1.000e-04 / 1.000e-05` at `500000`, then switch to `5.000e-05 / 5.000e-06` immediately after `500000`.
- Set5 still improves slightly beyond 400k in this segment, with the segment best PSNR at `530000`; SSIM stays at `0.9616` across several checkpoints, and segment-level tie handling recorded the latest best iter as `550000`.
- Set14 is strongest around `500000-527500`, then remains high but oscillatory through `550000`, which looks more like a marginal-gain plateau than a failure mode.
- Deep routers stay healthy: `b6 / b7` remain in the same non-collapsed regime seen at 300k-400k, and router scales remain stable around `~5.83` (b6) / `~6.08` (b7).
- Practical checkpoint picks from this segment are `500000`, `527500`, and `530000` rather than the final `550000` checkpoint.
