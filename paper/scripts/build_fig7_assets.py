"""Build Fig.7 visual-comparison assets locally (GT / Bicubic / DPRNet columns).

For each chosen x4 hard sample:
  1. Load GT (HR) and LR; produce Bicubic x4 baseline from LR.
  2. Auto-select a square crop where GT high-frequency energy (Laplacian var)
     is maximal -- the most texture-dense region, where content routing differs most.
  3. Save: per-method full image with a red crop box, and the magnified crop.
  4. Compute Y-channel PSNR/SSIM (crop_border=4) for Bicubic and DPRNet vs GT,
     on the FULL image (matching the paper's reported protocol), printed to stdout
     and dumped to a CSV for the figure caption / per-panel labels.

Competitor SR (original CATANet, SRFormer-light) is NOT available locally;
those columns are left as placeholders to fill from the training machine.

Only depends on numpy + PIL (no skimage/cv2).
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/Users/bytedance/WorkSpace/DPRNet/CATANet")
HR = ROOT / "datasets/TestDataSR/HR"
LR = ROOT / "datasets/TestDataSR/LR/LRBI"
SR = ROOT / "results/CATANet/test_CATANet_x4-250000/visualization"
OUT = Path("/Users/bytedance/WorkSpace/DPRNet/paper/figures/fig7_assets")

SCALE = 4
CROP = 96          # square crop side (in GT/SR pixels) before magnification
BORDER = SCALE     # crop_border for PSNR/SSIM, per SR convention at x4

# (dataset, sample_id)
SAMPLES = [
    ("Urban100", "img_092"),
    ("Urban100", "img_024"),
    ("Manga109", "ThatsIzumiko_000"),
]


def to_y(arr):
    """RGB uint8 -> Y (BT.601), float64, matching BasicSR test_y_channel."""
    a = arr.astype(np.float64)
    y = 16.0 + (65.481 * a[..., 0] + 128.553 * a[..., 1] + 24.966 * a[..., 2]) / 255.0
    return y


def psnr_y(sr, gt, border):
    s = to_y(sr)[border:-border, border:-border]
    g = to_y(gt)[border:-border, border:-border]
    mse = np.mean((s - g) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


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


def hf_map(gray):
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1]
        + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    out = np.zeros_like(gray)
    out[1:-1, 1:-1] = lap ** 2
    return out


def best_crop(gt_gray, crop):
    """Slide a (crop x crop) window with stride; return (top, left) of max HF energy."""
    H, W = gt_gray.shape
    crop = min(crop, H, W)
    energy = hf_map(gt_gray)
    integ = energy.cumsum(0).cumsum(1)
    integ = np.pad(integ, ((1, 0), (1, 0)))

    def box_sum(t, l):
        b, r = t + crop, l + crop
        return integ[b, r] - integ[t, r] - integ[b, l] + integ[t, l]

    best, bt, bl = -1.0, 0, 0
    stride = max(8, crop // 8)
    for t in range(0, H - crop + 1, stride):
        for l in range(0, W - crop + 1, stride):
            s = box_sum(t, l)
            if s > best:
                best, bt, bl = s, t, l
    return bt, bl, crop


def draw_box(im, top, left, side, width=4):
    im = im.copy()
    d = ImageDraw.Draw(im)
    d.rectangle([left, top, left + side, top + side], outline=(255, 0, 0), width=width)
    return im


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, sid in SAMPLES:
        gt = Image.open(HR / ds / f"x{SCALE}" / f"{sid}_x{SCALE}.png").convert("RGB")
        lr = Image.open(LR / ds / f"x{SCALE}" / f"{sid}_x{SCALE}.png").convert("RGB")
        dprnet = Image.open(SR / ds / f"{sid}_x{SCALE}_CATANet.png").convert("RGB")

        W, H = gt.size
        bicubic = lr.resize((W, H), Image.BICUBIC)

        gt_a = np.asarray(gt)
        bic_a = np.asarray(bicubic)
        dpr_a = np.asarray(dprnet)

        gt_gray = np.asarray(gt.convert("L"), dtype=np.float64)
        top, left, side = best_crop(gt_gray, CROP)

        # full PSNR/SSIM (Y, crop_border) per the paper protocol
        bic_psnr = psnr_y(bic_a, gt_a, BORDER)
        bic_ssim = ssim_y(bic_a, gt_a, BORDER)
        dpr_psnr = psnr_y(dpr_a, gt_a, BORDER)
        dpr_ssim = ssim_y(dpr_a, gt_a, BORDER)

        tag = f"{ds}_{sid}"
        # full image with red box (GT only, used as the locator panel)
        draw_box(gt, top, left, side).save(OUT / f"{tag}_GT_full_box.png")

        # magnified crops (nearest-neighbour upscale x3 so texture is visible in print)
        mag = 3
        for label, im in [("GT", gt), ("Bicubic", bicubic), ("DPRNet", dprnet)]:
            crop = im.crop((left, top, left + side, top + side))
            crop = crop.resize((side * mag, side * mag), Image.NEAREST)
            crop.save(OUT / f"{tag}_{label}_crop.png")

        rows.append([ds, sid, top, left, side,
                     f"{bic_psnr:.2f}", f"{bic_ssim:.4f}",
                     f"{dpr_psnr:.2f}", f"{dpr_ssim:.4f}"])
        print(f"{tag}: crop top={top} left={left} side={side} | "
              f"Bicubic {bic_psnr:.2f}/{bic_ssim:.4f}  DPRNet {dpr_psnr:.2f}/{dpr_ssim:.4f}")

    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "sample", "crop_top", "crop_left", "crop_side",
                    "bicubic_psnr", "bicubic_ssim", "dprnet_psnr", "dprnet_ssim"])
        w.writerows(rows)
    print(f"\nAssets + metrics.csv written to {OUT}")


if __name__ == "__main__":
    main()
