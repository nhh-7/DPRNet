# Repository Guidelines

## Project Structure & Module Organization

This repository centers on `CATANet/`, a BasicSR-based PyTorch implementation for lightweight image super-resolution. Core source lives in `CATANet/basicsr/`: model definitions in `archs/` and `models/`, datasets and loaders in `data/`, metrics in `metrics/`, losses in `losses/`, CUDA/C++ extensions in `ops/`, and shared helpers in `utils/`. Training and evaluation configs are YAML files under `CATANet/options/train/` and `CATANet/options/test/`. Put local datasets in `CATANet/datasets/` and downloaded weights in `CATANet/pretrained_models/`; keep generated outputs such as experiments, logs, and result images out of source control. Repository-level Chinese design notes and diagrams document DPR/CATANet refactor ideas.

## Build, Test, and Development Commands

Run commands from `CATANet/` unless noted:

```bash
pip install -r requirements.txt        # install runtime dependencies
python setup.py develop                # install basicsr in editable mode
BASICSR_EXT=True python setup.py develop  # optionally build CUDA extensions
python basicsr/test.py -opt options/test/test_CATANet_x2.yml
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 basicsr/train.py -opt options/train/train_CATANet_x2_scratch.yml --launcher pytorch
python -m pytest basicsr/metrics/test_metrics
```

Adjust GPU count, batch size, and dataset paths in the YAML configs before long runs.

## Coding Style & Naming Conventions

Use Python 3.9+ and follow the existing BasicSR style: 4-space indentation, snake_case for functions, files, and config keys, PascalCase for classes, and descriptive registry names. Keep architecture code in `basicsr/archs/*_arch.py` and model wrappers in `basicsr/models/*_model.py`. Prefer small, typed helper functions where practical, but match neighboring code over introducing unrelated formatting changes.

## Testing Guidelines

Existing tests use pytest, with examples under `basicsr/metrics/test_metrics/` and names like `test_psnr_ssim.py`. Add focused unit tests for new metrics, losses, transforms, or data utilities. For model changes, run the relevant `basicsr/test.py -opt ...` config and report PSNR/SSIM changes when applicable. Avoid committing large datasets, pretrained checkpoints, or generated visual results.

## Commit & Pull Request Guidelines

Recent history uses short, imperative Chinese summaries, for example `增加 将聚类索引转为彩色视觉图` and `删除 pycache 目录`. Keep commits concise and scoped to one change. Pull requests should include a brief purpose statement, changed configs or modules, commands run, key metrics, linked issues if any, and screenshots or sample outputs for visualization changes.
