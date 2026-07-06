#!/usr/bin/env python3
"""Generate multi-seed copies of the from-scratch ablation configs.

Causal-attribution ablation redesign (2026-07-05).

The base configs `train_CATANet_x4_ablfs_*_s3407.yml` all carry `manual_seed: 3407`
and `name: ...__s3407`. This script clones each of them for the extra seeds so that
every ablation variant is trained under >= 3 seeds, enabling a mean +/- std / paired
significance statement (and the C4 seed-variance sub-claim).

It does a purely textual, line-oriented rewrite so it does NOT need PyYAML and cannot
reorder/lose any keys:
  - `manual_seed: <old>`      -> `manual_seed: <new_seed>`
  - `name: <stem>_s3407`      -> `name: <stem>_s<new_seed>`

Usage (from CATANet/):
    python options/train/gen_seed_variants.py                 # seeds 42, 1234
    python options/train/gen_seed_variants.py --seeds 42 1234 7

Output: sibling files `..._s42.yml`, `..._s1234.yml`, etc. Existing files are skipped
unless --overwrite is given.
"""
import argparse
import glob
import os
import re

BASE_GLOB = "train_CATANet_x4_ablfs_*_s3407.yml"


def rewrite(text: str, new_seed: int) -> str:
    text = re.sub(r"(?m)^(\s*manual_seed:\s*)\d+\s*$", rf"\g<1>{new_seed}", text)
    text = re.sub(r"(?m)^(\s*name:\s*\S+?)_s3407\s*$", rf"\g<1>_s{new_seed}", text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1234],
                    help="extra seeds to generate (base seed 3407 already exists)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    bases = sorted(glob.glob(os.path.join(here, BASE_GLOB))) 
    if not bases:
        raise SystemExit(f"No base configs matching {BASE_GLOB} in {here}")

    made = 0
    for base in bases:
        with open(base, "r", encoding="utf-8") as f:
            text = f.read()
        for seed in args.seeds:
            out = base.replace("_s3407.yml", f"_s{seed}.yml")
            if os.path.exists(out) and not args.overwrite:
                print(f"skip (exists): {os.path.basename(out)}")
                continue
            with open(out, "w", encoding="utf-8") as f:
                f.write(rewrite(text, seed))
            print(f"wrote: {os.path.basename(out)}")
            made += 1
    print(f"\nDone. {made} config(s) generated for seeds {args.seeds}.")


if __name__ == "__main__":
    main()
