"""Sanity-check the numpy Y-PSNR implementation against the official test log.

The official x4 test (net_g_250000) reports dataset-MEAN PSNR. We recompute
per-image PSNR for all Urban100 images (GT vs DPRNet SR) and compare the mean
to the logged 26.8925 dB. A close match validates build_fig7_assets.py's metric.
"""
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("/Users/bytedance/WorkSpace/DPRNet/CATANet")
HR = ROOT / "datasets/TestDataSR/HR/Urban100/x4"
SR = ROOT / "results/CATANet/test_CATANet_x4-250000/visualization/Urban100"
BORDER = 4


def to_y(arr):
    a = arr.astype(np.float64)
    return 16.0 + (65.481 * a[..., 0] + 128.553 * a[..., 1] + 24.966 * a[..., 2]) / 255.0


def psnr_y(sr, gt, b):
    s = to_y(sr)[b:-b, b:-b]
    g = to_y(gt)[b:-b, b:-b]
    mse = np.mean((s - g) ** 2)
    return 10.0 * np.log10(255.0 ** 2 / mse)


vals = []
for gtp in sorted(HR.glob("*_x4.png")):
    sid = gtp.name.replace("_x4.png", "")
    srp = SR / f"{sid}_x4_CATANet.png"
    if not srp.exists():
        continue
    gt = np.asarray(Image.open(gtp).convert("RGB"))
    sr = np.asarray(Image.open(srp).convert("RGB"))
    if gt.shape != sr.shape:
        print(f"shape mismatch {sid}: {gt.shape} vs {sr.shape}")
        continue
    vals.append(psnr_y(sr, gt, BORDER))

print(f"Urban100 x4 mean Y-PSNR (recomputed): {np.mean(vals):.4f} dB  over {len(vals)} imgs")
print("Official logged value:                26.8925 dB")
