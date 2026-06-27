"""Rank SR outputs by high-frequency energy to pick visual-comparison candidates.

Content routing (DPR) benefits most on dense repetitive high-frequency texture.
We proxy "difficulty / texture richness" by the variance of a Laplacian-like
high-pass response on the luminance channel of DPRNet's own x4 SR outputs.
This is only a candidate-selection heuristic; final pick is by visual inspection.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

VIS = Path("/Users/bytedance/WorkSpace/DPRNet/CATANet/results/CATANet/"
           "test_CATANet_x4-250000/visualization")


def hf_energy(path: Path) -> float:
    img = Image.open(path).convert("L")
    a = np.asarray(img, dtype=np.float32)
    # 4-neighbour Laplacian high-pass
    lap = (
        -4.0 * a[1:-1, 1:-1]
        + a[:-2, 1:-1] + a[2:, 1:-1]
        + a[1:-1, :-2] + a[1:-1, 2:]
    )
    return float(lap.var())


def rank(dataset: str, topk: int = 12):
    d = VIS / dataset
    rows = []
    for p in sorted(d.glob("*_x4_CATANet.png")):
        if p.name.endswith("_cluster.png"):
            continue
        name = p.name.replace("_x4_CATANet.png", "")
        rows.append((name, hf_energy(p)))
    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"\n=== {dataset}: top {topk} by HF energy (of {len(rows)}) ===")
    for name, v in rows[:topk]:
        print(f"  {name:30s} {v:10.1f}")


if __name__ == "__main__":
    rank("Urban100")
    rank("Manga109")
