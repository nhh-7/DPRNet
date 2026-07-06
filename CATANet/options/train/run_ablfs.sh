#!/usr/bin/env bash
# =============================================================================
# From-scratch, multi-seed ablation runner (causal-attribution redesign, 2026-07-05)
#
# WHY THIS EXISTS
#   The published ablations (train_CATANet_x4_abl_*.yml) finetune every variant from a
#   SINGLE converged full-model checkpoint for 80k iters under ONE seed. That shared
#   initialization means a disabled switch is removed from a representation already
#   adapted to it, so no isolated gain can be attributed, and there is no variance.
#
#   This script trains every variant FROM SCRATCH for the full 250k budget under
#   MULTIPLE seeds, which is what supports per-switch causal attribution and a
#   mean +/- std / significance statement.
#
# COST WARNING
#   6 variants x 3 seeds = 18 from-scratch x4 runs (each ~ a full 250k training).
#   Budget the compute before launching. If constrained, run PRIORITY_A3 only first:
#   it is the group whose C4 "seed-variance reduction" sub-claim strictly needs seeds.
#
# USAGE (from CATANet/)
#   1) Generate the extra-seed configs:
#        python options/train/gen_seed_variants.py           # -> _s42, _s1234
#   2) Launch everything (edit GPU / seeds / variants as needed):
#        bash options/train/run_ablfs.sh
#      Or just the A3 priority group:
#        RUN_SET=A3 bash options/train/run_ablfs.sh
#   3) After training, test each best checkpoint on Set5+Urban100 and export routing
#      diagnostics (see the tee pattern in data-collection-checklist.md A1).
# =============================================================================
set -euo pipefail

GPU="${GPU:-0}"
SEEDS="${SEEDS:-3407 42 1234}"
RUN_SET="${RUN_SET:-ALL}"          # ALL | A1 | A2 | A3 | FULL | C1
OPT_DIR="options/train"

# Variant stems (without _s<seed>.yml). FULL is the shared top-row reference.
FULL="train_CATANet_x4_ablfs_full"
A1="train_CATANet_x4_ablfs_A1_refine_off"
A2=( "train_CATANet_x4_ablfs_A2_v1_hardsort" \
     "train_CATANet_x4_ablfs_A2_v2_confsort" \
     "train_CATANet_x4_ablfs_A2_v3_scoregate" )
A3="train_CATANet_x4_ablfs_A3_balance_off"
# C1: dynamic prototypes (DPR) vs cross-batch EMA centers, same protocol/seeds.
C1=( "train_CATANet_x4_ablfs_c1_dpr" \
     "train_CATANet_x4_ablfs_c1_emacenter" )

case "$RUN_SET" in
  ALL)  STEMS=( "$FULL" "$A1" "${A2[@]}" "$A3" ) ;;
  FULL) STEMS=( "$FULL" ) ;;
  A1)   STEMS=( "$FULL" "$A1" ) ;;
  A2)   STEMS=( "$FULL" "${A2[@]}" ) ;;
  A3)   STEMS=( "$FULL" "$A3" ) ;;
  C1)   STEMS=( "${C1[@]}" ) ;;
  *)    echo "Unknown RUN_SET=$RUN_SET"; exit 1 ;;
esac

echo "GPU=$GPU  SEEDS=[$SEEDS]  RUN_SET=$RUN_SET"
echo "Variants: ${STEMS[*]}"
echo

for stem in "${STEMS[@]}"; do
  for seed in $SEEDS; do
    opt="${OPT_DIR}/${stem}_s${seed}.yml"
    if [[ ! -f "$opt" ]]; then
      echo "!! missing config: $opt  (run gen_seed_variants.py first)"; exit 1
    fi
    echo "==== TRAIN  $opt  ===="
    CUDA_VISIBLE_DEVICES="$GPU" torchrun --standalone --nnodes=1 --nproc_per_node=1 \
      basicsr/train.py -opt "$opt" --launcher pytorch
  done
done

echo
echo "All requested runs finished. Next: test best checkpoints + export routing CSVs."
