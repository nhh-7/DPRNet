"""Build Fig.7 competitor crops + metrics from staged competitor SR full images.

Run this AFTER you have produced each competitor's SR output for the three
Fig.7 hard samples and staged them under STAGE/<Method>/<dataset>_<sample>.png
(see paper/plan/fig7-competitor-inference-guide.md, step 3).

For each (Method, sample) it:
  1. loads the GT full image and the competitor SR full image,
  2. computes Y-channel PSNR/SSIM (crop_border=4) on the FULL image -- the SAME
     protocol used by build_fig7_assets.py for Bicubic/DPRNet, so the numbers are
     directly comparable,
  3. crops the FIXED region defined in metrics.csv (identical box for every
     column) and magnifies x3 with NEAREST, matching the existing crops,
  4. writes figures/fig7_assets/<tag>_<Method>_crop.png and a competitor CSV.

Only depends on numpy + PIL. Safe to run on the training machine or locally.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# CONFIG -- edit these three paths for the machine you run on.
# ---------------------------------------------------------------------------
# Where you staged competitor SR full images: STAGE/<Method>/<dataset>_<sample>.png
STAGE = Path("/hy-tmp/fig7_stage")
# GT (HR) root, e.g. <HR>/<dataset>/x4/<sample>_x4.png
HR = Path("/hy-tmp/TestDataSR/HR")
# Output dir for crops + competitor metrics CSV (copy these back to the repo).
OUT = Path("/hy-tmp/fig7_out")

SCALE = 4
BORDER = SCALE
MAG = 3  # NEAREST magnification of the crop (matches build_fig7_assets.py)

# Competitor display tags -> staging sub-folder name. Order = figure column order.
METHODS = ["IMDN", "RFDN", "SwinIR-light", "SRFormer-light", "CATANet"]

# (dataset, sample, crop_top, crop_left, crop_side) -- copied from metrics.csv.
# These MUST stay identical to the Bicubic/DPRNet/GT crops already in the figure.
SAMPLES = [
    ("Urban100", "img_092", 204, 756, 96),
    ("Urban100", "img_024", 408, 456, 96),
    ("Manga109", "ThatsIzumiko_000", 792, 648, 96),
]


def to_y(arr):
    """RGB uint8 -> Y (BT.601), matching BasicSR test_y_channel."""
    a = arr.astype(np.float64)
    return 16.0 + (65.481 * a[..., 0] + 128.553 * a[..., 1] + 24.966 * a[..., 2]) / 255.0


def psnr_y(sr, gt, border):
    s = to_y(sr)[border:-border, border:-border]
    g = to_y(gt)[border:-border, border:-border]
    mse = np.mean((s - g) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def _gaussian_window(size=11, sigma=1.5):
    coords = np.arange(size) - size // 2
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


def _filter2(img, win):
    from numpy.lib.stride_tricks import sliding_window_view
    k = win.shape[0]
    v = sliding_window_view(img, (k, k))
    return np.einsum("ijkl,kl->ij", v, win)


def ssim_y(sr, gt, border):
    s = to_y(sr)[border:-border, border:-border]
    g = to_y(gt)[border:-border, border:-border]
    win = _gaussian_window()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_s, mu_g = _filter2(s, win), _filter2(g, win)
    mu_s2, mu_g2, mu_sg = mu_s ** 2, mu_g ** 2, mu_s * mu_g
    sig_s = _filter2(s * s, win) - mu_s2
    sig_g = _filter2(g * g, win) - mu_g2
    sig_sg = _filter2(s * g, win) - mu_sg
    ssim_map = ((2 * mu_sg + c1) * (2 * sig_sg + c2)) / (
        (mu_s2 + mu_g2 + c1) * (sig_s + sig_g + c2)
    )
    return float(ssim_map.mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, sid, top, left, side in SAMPLES:
        tag = f"{ds}_{sid}"
        gt = Image.open(HR / ds / f"x{SCALE}" / f"{sid}_x{SCALE}.png").convert("RGB")
        gt_a = np.asarray(gt)
        for method in METHODS:
            src = STAGE / method / f"{ds}_{sid}.png"
            if not src.exists():
                print(f"[skip] missing {src}")
                continue
            sr = Image.open(src).convert("RGB")
            if sr.size != gt.size:
                # competitor SR must be full-res HR size; resize guard for safety
                print(f"[warn] {method} {tag}: size {sr.size} != GT {gt.size}; "
                      "check the SR output is full-resolution, not the crop.")
            sr_a = np.asarray(sr)
            p = psnr_y(sr_a, gt_a, BORDER)
            s = ssim_y(sr_a, gt_a, BORDER)

            crop = sr.crop((left, top, left + side, top + side))
            crop = crop.resize((side * MAG, side * MAG), Image.NEAREST)
            crop.save(OUT / f"{tag}_{method}_crop.png")

            rows.append([ds, sid, method, f"{p:.2f}", f"{s:.4f}"])
            print(f"{tag:30s} {method:15s} {p:.2f}/{s:.4f}")

    with open(OUT / "competitor_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "sample", "method", "psnr", "ssim"])
        w.writerows(rows)
    print(f"\nCrops + competitor_metrics.csv written to {OUT}")


if __name__ == "__main__":
    main()
