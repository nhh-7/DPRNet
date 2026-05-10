# CATANet 安装、训练与测试流程

## 1. 文档目的

本文档用于说明当前 `CATANet` 项目的完整使用流程，包括：

1. 依赖安装
2. 环境准备
3. 数据集放置
4. 训练命令
5. 测试命令
6. 输出目录说明
7. 常见问题与注意事项

本文档基于当前仓库中的如下真实入口整理：

- `requirements.txt`
- `setup.py`
- `basicsr/train.py`
- `basicsr/test.py`
- `options/train/*.yml`
- `options/test/*.yml`
- `README.md`

## 2. 项目结构与执行根目录

以下命令默认在项目根目录执行：

```bash
cd /Users/hainan/WorkSpace/DPRNet/CATANet
```

后文所有相对路径均相对于该目录。

## 3. 环境要求

根据当前仓库内容，推荐使用以下基础环境：

- Python `3.9`
- PyTorch `>= 2.1`
- torchvision
- CUDA 可用的 GPU 环境

仓库 `README.md` 中给出的要求是：

- Python `3.9`
- PyTorch `>= 2.2`

而 `requirements.txt` 中约束为：

- `torch>=2.1`

因此实际建议为：

```text
Python 3.9 + PyTorch 2.1/2.2 + CUDA 对应版本
```

## 4. 创建环境

推荐使用 `conda` 创建独立环境：

```bash
conda create -n catanet python=3.9 -y
conda activate catanet
```

然后按你的 CUDA 版本安装 PyTorch。以下仅给出常见示例，具体请以 PyTorch 官方安装命令为准。

例如，CUDA 11.8：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

如果你使用 CPU 或其他 CUDA 版本，请替换成对应官方命令。

## 5. 安装项目依赖

### 5.1 安装 Python 依赖

先安装仓库依赖：

```bash
pip install -r requirements.txt
```

当前 `requirements.txt` 主要包括：

- `lmdb`
- `numpy>=1.17`
- `opencv-python`
- `pyyaml`
- `requests`
- `scipy`
- `tensorboard`
- `torch>=2.1`
- `torchvision`
- `tqdm`
- `einops`

### 5.2 安装项目包

执行：

```bash
python setup.py develop
```

该命令会把当前工程以开发模式安装为 `basicsr` 包，便于直接调用：

- `basicsr/train.py`
- `basicsr/test.py`

### 5.3 关于 CUDA 扩展

当前 `setup.py` 支持可选 CUDA 扩展编译，受环境变量 `BASICSR_EXT` 控制：

```bash
BASICSR_EXT=True python setup.py develop
```

如果不设置该变量，则默认不编译这些扩展。

对于先跑通训练/测试流程来说，建议先使用默认安装方式：

```bash
python setup.py develop
```

只有当你明确需要编译 `deform_conv`、`fused_act`、`upfirdn2d` 等扩展时，再启用 `BASICSR_EXT=True`。

## 6. 额外依赖注意事项

### 6.1 `deepspeed` 说明

当前代码已经修正为：

```text
不启用 deepspeed 时，不需要安装它；
只有当 ENABLE_DEEPSPEED=true 时，才要求环境中存在 deepspeed。
```

如果你需要启用它，请手动安装：

```bash
pip install deepspeed
```

如果你不使用 `deepspeed`，则无需额外安装。

### 6.2 多卡训练依赖

项目训练命令使用的是：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 basicsr/train.py ...
```

因此多卡训练依赖：

- `torchrun`
- 可用的多张 GPU
- `nccl` 后端

在 Linux 服务器环境中通常更稳妥；macOS 适合做阅读、轻量检查和非正式实验，不适合作为正式多卡训练环境。

## 7. 数据集准备

## 7.1 训练集

当前训练配置使用：

- `DIV2K_HR`
- `DIV2K_LR_bicubic/x2`
- `DIV2K_LR_bicubic/x3`
- `DIV2K_LR_bicubic/x4`

训练集推荐放置为：

```text
CATANet/
└── datasets/
    └── DIV2K/
        ├── DIV2K_HR/
        └── DIV2K_LR_bicubic/
            ├── x2/
            ├── x3/
            └── x4/
```

其中，`x2 / x3 / x4` 分别对应不同放大倍数的低分辨率输入。

## 7.2 测试集

当前测试配置覆盖以下数据集：

- `Set5`
- `Set14`
- `B100`
- `Urban100`
- `Manga109`

推荐目录结构如下：

```text
CATANet/
└── datasets/
    ├── Set5/
    │   ├── HR/
    │   └── LR_bicubic/
    │       ├── X2/
    │       ├── X3/
    │       └── X4/
    ├── Set14/
    │   ├── HR/
    │   └── LR_bicubic/
    │       ├── X2/
    │       ├── X3/
    │       └── X4/
    ├── B100/
    │   ├── HR/
    │   └── LR_bicubic/
    │       ├── X2/
    │       ├── X3/
    │       └── X4/
    ├── Urban100/
    │   ├── HR/
    │   └── LR_bicubic/
    │       ├── X2/
    │       ├── X3/
    │       └── X4/
    └── Manga109/
        ├── HR/
        └── LR_bicubic/
            ├── X2/
            ├── X3/
            └── X4/
```

### 7.3 当前配置中的默认路径

以 `x2` 训练配置为例，当前默认路径是：

```yaml
dataroot_gt: datasets/DIV2K/DIV2K_HR
dataroot_lq: datasets/DIV2K/DIV2K_LR_bicubic/x2
```

以 `x2` 测试配置为例，默认路径是：

```yaml
dataroot_gt: datasets/Set5/HR
dataroot_lq: datasets/Set5/LR_bicubic/X2
```

因此如果你的数据目录不同，需要先修改 `options/train/*.yml` 或 `options/test/*.yml`。

## 8. 预训练模型准备

测试时默认使用预训练权重，例如 `x2` 配置中：

```yaml
path:
  pretrain_network_g: pretrained_models/x2.pth
```

因此建议目录如下：

```text
CATANet/
└── pretrained_models/
    ├── x2.pth
    ├── x3.pth
    └── x4.pth
```

如果你更换了模型路径，也需要同步修改测试配置文件中的：

```yaml
path:
  pretrain_network_g: ...
```

## 9. 训练流程

## 9.1 查看训练配置

训练配置文件位于：

```text
options/train/
```

当前主要包括：

- `train_CATANet_x2_scratch.yml`
- `train_CATANet_x3_finetune.yml`
- `train_CATANet_x4_finetune.yml`

它们分别对应：

- x2 从头训练
- x3 微调训练
- x4 微调训练

## 9.2 官方多卡训练命令

### x2 从头训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
basicsr/train.py -opt options/train/train_CATANet_x2_scratch.yml --launcher pytorch
```

说明：

- 4 卡训练
- 每卡 `batch_size_per_gpu = 16`
- 总 batch size = `4 x 16 = 64`
- 总迭代数 `800000`

### x3 微调训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
basicsr/train.py -opt options/train/train_CATANet_x3_finetune.yml --launcher pytorch
```

说明：

- 4 卡训练
- 每卡 `batch_size_per_gpu = 16`
- 总迭代数 `250000`

### x4 微调训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
basicsr/train.py -opt options/train/train_CATANet_x4_finetune.yml --launcher pytorch
```

说明：

- 4 卡训练
- 每卡 `batch_size_per_gpu = 16`
- 总迭代数 `250000`

## 9.3 单卡训练命令

如果你只想先在单卡上跑通流程，可以使用：

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_CATANet_x2_scratch.yml --launcher none
```

同理，x3 与 x4 只需替换配置文件：

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_CATANet_x3_finetune.yml --launcher none
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_CATANet_x4_finetune.yml --launcher none
```

注意：

```text
单卡时建议把 batch_size_per_gpu、num_worker_per_gpu 等参数按显存与机器情况适当调小。
```

## 9.4 调试模式

训练脚本支持 `--debug`：

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_CATANet_x2_scratch.yml --launcher none --debug
```

调试模式下，代码会自动缩短部分验证/日志间隔，便于快速检查流程是否正常。

## 9.5 强制覆盖配置项

训练和测试入口都支持 `--force_yml` 覆盖配置项，例如：

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
  -opt options/train/train_CATANet_x2_scratch.yml \
  --launcher none \
  --force_yml train:ema_decay=0.999 logger:print_freq=20
```

适合做临时实验，而不必直接修改原始 `yml` 文件。

## 10. 训练输出目录

训练时，程序会自动在：

```text
experiments/<experiment_name>/
```

下生成实验目录。

例如 `train_CATANet_x2_scratch.yml` 中：

```yaml
name: train_CATANet_x2_scratch
```

则输出目录通常为：

```text
experiments/train_CATANet_x2_scratch/
```

其中一般包括：

- `models/`：模型权重
- `training_states/`：训练状态
- `visualization/`：可视化结果
- 日志文件

## 11. 测试流程

## 11.1 查看测试配置

测试配置位于：

```text
options/test/
```

当前主要包括：

- `test_CATANet_x2.yml`
- `test_CATANet_x3.yml`
- `test_CATANet_x4.yml`

## 11.2 测试命令

### x2 测试

```bash
python basicsr/test.py -opt options/test/test_CATANet_x2.yml
```

### x3 测试

```bash
python basicsr/test.py -opt options/test/test_CATANet_x3.yml
```

### x4 测试

```bash
python basicsr/test.py -opt options/test/test_CATANet_x4.yml
```

## 11.3 测试结果输出目录

测试时，程序会自动把结果写到：

```text
results/<experiment_name>/
```

例如 `test_CATANet_x2.yml` 中：

```yaml
name: test_CATANet_x2
path:
  results_root: results/CATANet
```

最终结果目录会进一步拼接实验名，通常形成：

```text
results/CATANet/test_CATANet_x2/
```

该目录下通常包含：

- 日志
- 指标结果
- `visualization/` 保存的测试图像

## 12. 一套推荐的完整执行顺序

如果你想从零开始完整跑一遍，建议按下面顺序执行。

### 步骤 1：进入项目目录

```bash
cd /Users/hainan/WorkSpace/DPRNet/CATANet
```

### 步骤 2：创建并激活环境

```bash
conda create -n catanet python=3.9 -y
conda activate catanet
```

### 步骤 3：安装 PyTorch

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 步骤 4：安装仓库依赖

```bash
pip install -r requirements.txt
python setup.py develop
```

如果你后续要启用 `deepspeed`，再额外执行：

```bash
pip install deepspeed
```

### 步骤 5：准备数据和权重

确保以下目录已经存在并放好数据：

```text
datasets/DIV2K/DIV2K_HR
datasets/DIV2K/DIV2K_LR_bicubic/x2
datasets/DIV2K/DIV2K_LR_bicubic/x3
datasets/DIV2K/DIV2K_LR_bicubic/x4

datasets/Set5/HR
datasets/Set5/LR_bicubic/X2
datasets/Set5/LR_bicubic/X3
datasets/Set5/LR_bicubic/X4
```

确保测试权重存在：

```text
pretrained_models/x2.pth
pretrained_models/x3.pth
pretrained_models/x4.pth
```

### 步骤 6：先做一次单卡冒烟训练

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_CATANet_x2_scratch.yml --launcher none --debug
```

### 步骤 7：正式训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
basicsr/train.py -opt options/train/train_CATANet_x2_scratch.yml --launcher pytorch
```

### 步骤 8：测试

```bash
python basicsr/test.py -opt options/test/test_CATANet_x2.yml
```

## 13. 常见问题

### 13.1 报 `No module named basicsr`

说明项目包未安装成功，重新执行：

```bash
python setup.py develop
```

### 13.2 报 `No module named deepspeed`

如果你明确启用了 `deepspeed`，请安装：

```bash
pip install deepspeed
```

如果你没有启用 `deepspeed`，则通常不需要安装它；请先检查是否误设置了：

```bash
ENABLE_DEEPSPEED=true
```

### 13.3 报数据路径不存在

请检查：

- `options/train/*.yml`
- `options/test/*.yml`

中的 `dataroot_gt` 与 `dataroot_lq` 是否与你本地目录一致。

### 13.4 报预训练权重找不到

请检查：

```yaml
path:
  pretrain_network_g: ...
```

是否指向实际存在的权重文件。

### 13.5 单卡显存不足

优先调整：

- `batch_size_per_gpu`
- `num_worker_per_gpu`
- `gt_size`

必要时先用 `--debug` 跑最小流程。

## 14. 总结

当前 `CATANet` 的标准使用路径可以概括为：

```text
创建 Python 3.9 环境
-> 安装 PyTorch
-> pip install -r requirements.txt
-> python setup.py develop
-> 如需 deepspeed 再单独安装
-> 准备 datasets 与 pretrained_models
-> 先单卡调试
-> 再多卡正式训练
-> 使用 options/test/*.yml 进行测试
```

如果后续你还会继续改动 `CATANet` 的结构或配置，建议同步维护本文件，保证仓库中始终保留一份可以直接执行的中文操作手册。
