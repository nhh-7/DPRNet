#!/usr/bin/env bash
set -e

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  basicsr/train.py -opt options/train/train_CATANet_x4_abl_A2_v1_hardsort.yml --launcher pytorch

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  basicsr/train.py -opt options/train/train_CATANet_x4_abl_A2_v2_confsort.yml --launcher pytorch

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  basicsr/train.py -opt options/train/train_CATANet_x4_abl_A2_v3_scoregate.yml --launcher pytorch

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  basicsr/train.py -opt options/train/train_CATANet_x4_abl_A3_balance_off.yml --launcher pytorch

bash backup_result_and_shutdown.sh
