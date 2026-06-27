# Fig.7 对手列推理操作手册（训练机自包含版）

> 目标：在训练机上为 Fig.7（×4 视觉对比，3 个难样本）补齐 **5 个对手列** 的 SR 图与
> Y-PSNR/SSIM，使其与已有的 Bicubic / DPRNet / GT 列严格对齐，可直接填进
> [fig7_visual.tex](file:///Users/bytedance/WorkSpace/DPRNet/paper/figures/fig7_visual.tex)。
>
> 本文档自包含：照着从上到下走即可完成全部对手推理，无需再查别处。

---

## 0. 全局约定（务必先读）

| 项 | 值 | 说明 |
|---|---|---|
| Scale | ×4 | Fig.7 只做 ×4 |
| 三个难样本 | `Urban100/img_092`、`Urban100/img_024`、`Manga109/ThatsIzumiko_000` | 已固定，不要换 |
| 评测口径 | Y 通道，`crop_border=4`，在**全图**上算 PSNR/SSIM | 与主表、与 DPRNet 列一致 |
| 裁剪框 | 见下表，**所有列共用同一框**，不许各列各裁 | 否则视觉对比无意义 |
| 测试数据根 | `/hy-tmp/TestDataSR/{HR,LR/LRBI}/<dataset>/x4/<sample>_x4.png` | 与 DPRNet 测试一致 |

固定裁剪框（来自 [metrics.csv](file:///Users/bytedance/WorkSpace/DPRNet/paper/figures/fig7_assets/metrics.csv)，单位为 HR 像素）：

| dataset | sample | crop_top | crop_left | crop_side |
|---|---|---|---|---|
| Urban100 | img_092 | 204 | 756 | 96 |
| Urban100 | img_024 | 408 | 456 | 96 |
| Manga109 | ThatsIzumiko_000 | 792 | 648 | 96 |

**核心思路**：每个对手只需要产出这 3 张图的 **SR 全图**（HR 分辨率，不是裁好的小块），
然后统一交给本仓库脚本 [build_fig7_competitor_crops.py](file:///Users/bytedance/WorkSpace/DPRNet/paper/scripts/build_fig7_competitor_crops.py)
去裁同一个框、算同一套指标。这样 5 个对手的环境差异（BasicSR / 独立框架 / 旧版 PyTorch）
都被隔离在“产出 SR 全图”这一步，后处理完全统一。

对手列在图中的顺序（弱→强）：
`Bicubic → IMDN → RFDN → SwinIR-light → SRFormer-light → CATANet → DPRNet(ours) → GT`

---

## 1. 准备：暂存目录与测试图

在训练机建一个暂存目录，按对手分子目录。每个对手跑完后，把它对这 3 个样本的 SR 全图
重命名成 `<dataset>_<sample>.png` 放进对应子目录：

```bash
mkdir -p /hy-tmp/fig7_stage/{IMDN,RFDN,SwinIR-light,SRFormer-light,CATANet}
mkdir -p /hy-tmp/fig7_out
```

最终期望的暂存结构（15 张全图）：

```
/hy-tmp/fig7_stage/
├── IMDN/            Urban100_img_092.png  Urban100_img_024.png  Manga109_ThatsIzumiko_000.png
├── RFDN/            （同上 3 张）
├── SwinIR-light/    （同上 3 张）
├── SRFormer-light/  （同上 3 张）
└── CATANet/         （同上 3 张）
```

> 提示：3 个样本的 LR / HR 已在 `/hy-tmp/TestDataSR/` 下。每个对手既可以只对这 3 张 LR 推理，
> 也可以跑完整 Urban100/Manga109 再挑出这 3 张——后者更省心（很多框架默认整目录跑）。

---

## 2. 五个对手：仓库、权重、环境、测试命令

> 所有命令默认在该对手仓库根目录执行。建议每个对手单独建一个 conda 环境，避免 PyTorch 版本互相污染。

### 2.1 CATANet（最强对手，BasicSR 框架）★必做

- **仓库**：https://github.com/EquationWalker/CATANet
- **预训练权重**：https://github.com/EquationWalker/CATANet/releases/tag/v0.0 （放入 `./pretrained_models`）
- **可选直接拿可视化结果**：作者发布了 visual results（夸克网盘 https://pan.quark.cn/s/f8ea09048957）。
  但**仍需自己裁我们的固定框**，所以即便下载，也只是省去推理、不能省去 step 3。
- **环境**：Python 3.9 / PyTorch ≥ 2.2

```bash
git clone https://github.com/EquationWalker/CATANet
cd CATANet
conda create -n catanet python=3.9 -y && conda activate catanet
pip install -r requirements.txt
python setup.py develop
# 把官方 x4 权重放到 ./pretrained_models/ 下
```

改 `options/test/test_CATANet_x4.yml` 的数据路径指向我们的测试集（与本仓库一致）：
```yaml
dataroot_gt: /hy-tmp/TestDataSR/HR/Urban100/x4
dataroot_lq: /hy-tmp/TestDataSR/LR/LRBI/Urban100/x4
# Manga109 同理；save_img: true
```

```bash
python basicsr/test.py -opt options/test/test_CATANet_x4.yml
```

- **SR 全图输出**：`results/<name>/visualization/<dataset>/<sample>_x4_<suffix>.png`
- **取图**：把 3 张对应 SR 全图复制到 `/hy-tmp/fig7_stage/CATANet/<dataset>_<sample>.png`

> ⚠️ 注意：**这是官方原版 CATANet**，与我们 DPRNet 不是同一个权重。
> 别和本仓库 `results/CATANet/...`（那是 DPRNet 的输出，arch 注册名恰好叫 CATANet）混淆。

### 2.2 SRFormer-light（Transformer 对手，BasicSR 框架）

- **仓库**：https://github.com/HVision-NKU/SRFormer
- **预训练权重**：Google Drive https://drive.google.com/drive/folders/1D5ER_HwYJoyZCcrKVstwE-iEl0hXulwd
  （取轻量 ×4 模型，放入 `./PretrainModel`）
- **环境**：Python 3.8 / PyTorch ≥ 1.7

```bash
git clone https://github.com/HVision-NKU/SRFormer
cd SRFormer
conda create -n srformer python=3.8 -y && conda activate srformer
pip install -r requirements.txt
python setup.py develop
# 把 light x4 权重放到 ./PretrainModel/ 下
```

改 `options/test/SRFormer/test_SRFormer_light_DIV2Ksrx4.yml`：
- `pretrain_network_g` 指向下载的 light x4 权重；
- 5 个/或仅需的数据集 `dataroot_gt/lq` 指向 `/hy-tmp/TestDataSR/...`；
- 确认 `val.save_img: true`。

```bash
python basicsr/test.py -opt options/test/SRFormer/test_SRFormer_light_DIV2Ksrx4.yml
```

- **SR 全图输出**：`results/<name>/visualization/<dataset>/<sample>_<suffix>.png`
- **取图**：复制 3 张到 `/hy-tmp/fig7_stage/SRFormer-light/<dataset>_<sample>.png`

> SRFormer 要求 HR 与 LR **同名**。我们的数据是 `<sample>_x4.png` 成对存在，已满足。

### 2.3 SwinIR-light（Transformer 对手，**独立框架**，非 BasicSR）

- **仓库**：https://github.com/JingyunLiang/SwinIR
- **预训练权重**：Releases / model_zoo，轻量 ×4 文件名
  `002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth`
  （下载页 https://github.com/JingyunLiang/SwinIR/releases ）
- **训练代码**（如需）：KAIR https://github.com/cszn/KAIR
- **环境**：Python 3.8+ / PyTorch ≥ 1.7（`pip install timm`）

```bash
git clone https://github.com/JingyunLiang/SwinIR
cd SwinIR
conda create -n swinir python=3.8 -y && conda activate swinir
pip install torch torchvision timm opencv-python numpy
mkdir -p model_zoo/swinir
# 下载 002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth 到 model_zoo/swinir/
```

**它用独立测试脚本，不是 `basicsr/test.py`**。直接对 LR / HR 目录跑：

```bash
# Urban100
python main_test_swinir.py --task lightweight_sr --scale 4 \
  --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth \
  --folder_lq /hy-tmp/TestDataSR/LR/LRBI/Urban100/x4 \
  --folder_gt /hy-tmp/TestDataSR/HR/Urban100/x4
# Manga109（把上面两个目录换成 Manga109/x4 再跑一次）
```

- **SR 全图输出**：`results/swinir_lightweight_sr_x4/<sample>_SwinIR.png`（脚本会同时打印 PSNR/SSIM）
- **取图**：复制 3 张到 `/hy-tmp/fig7_stage/SwinIR-light/<dataset>_<sample>.png`

> SwinIR 的文件命名 / 评测口径与 BasicSR 略有差异，**不要采信它脚本打印的 PSNR**，
> 统一以 step 3 我们脚本重算的为准（保证 5 列同口径）。

### 2.4 RFDN（CNN 轻量对手，独立框架，旧版 PyTorch）

- **仓库**：https://github.com/njulj/RFDN （ECCV2020 / arXiv:2009.11551）
- **预训练权重**：随仓库 `trained_model/`（如 `RFDN_AIM.pth`）；轻量 SR 权重见仓库说明
- **环境**：PyTorch 1.x（老仓库，建议单独环境）

RFDN 自带的 [test.py](https://github.com/njulj/RFDN/blob/master/test.py) 是写死 DIV2K 目录的 AIM 挑战脚本，
**需要改三处**：`model_path`、`L_folder`（LR 目录）、解开末尾保存结果的注释。最小改法：

```python
model_path = 'trained_model/<your_rfdn_x4>.pth'   # 指向 x4 权重
L_folder   = '/hy-tmp/TestDataSR/LR/LRBI/Urban100/x4'
E_folder   = '/hy-tmp/fig7_stage/RFDN'            # 直接存到暂存目录
# 取消 util.imsave(...) 那一行的注释，让它真正写出 SR 图
```

```bash
git clone https://github.com/njulj/RFDN && cd RFDN
conda create -n rfdn python=3.8 -y && conda activate rfdn
pip install torch torchvision numpy opencv-python
python test.py   # 改好路径后
# Manga109 改 L_folder 再跑一次
```

- **取图**：RFDN 输出文件名跟随输入名（`<sample>_x4.png`）。重命名为 `<dataset>_<sample>.png`
  放进 `/hy-tmp/fig7_stage/RFDN/`。

> 若官方权重只有 AIM 版（×4）与论文表数略有出入，没关系——Fig.7 是**定性**对比，
> 只要是 RFDN 官方权重产出的真实 SR 即可；step 3 会重算它的 PSNR/SSIM。

### 2.5 IMDN（CNN 轻量对手，独立框架，旧版 PyTorch）

- **仓库**：https://github.com/Zheng222/IMDN （ACM MM 2019 / arXiv:1909.11856）
- **预训练权重**：随仓库 `checkpoints/`（含 `IMDN_x4.pth`）
- **环境**：PyTorch 1.1（老仓库，单独环境）

IMDN 自带 `test_IMDN.py`，参数化程度好，直接用：

```bash
git clone https://github.com/Zheng222/IMDN && cd IMDN
conda create -n imdn python=3.7 -y && conda activate imdn
pip install torch torchvision numpy scikit-image opencv-python

# Urban100 x4
python test_IMDN.py \
  --test_hr_folder /hy-tmp/TestDataSR/HR/Urban100/x4/ \
  --test_lr_folder /hy-tmp/TestDataSR/LR/LRBI/Urban100/x4/ \
  --output_folder /hy-tmp/fig7_stage/IMDN/ \
  --checkpoint checkpoints/IMDN_x4.pth --upscale_factor 4
# Manga109 同理换两个 folder 再跑一次
```

- **取图**：输出到 `/hy-tmp/fig7_stage/IMDN/`，文件名跟随输入；重命名为 `<dataset>_<sample>.png`。

---

## 3. 统一后处理：裁同一框 + 算同口径指标

5 个对手的 15 张 SR 全图都到位（`/hy-tmp/fig7_stage/<Method>/<dataset>_<sample>.png`）后，
跑本仓库已备好的脚本：[build_fig7_competitor_crops.py](file:///Users/bytedance/WorkSpace/DPRNet/paper/scripts/build_fig7_competitor_crops.py)

```bash
# 该脚本顶部三个路径已默认为 /hy-tmp/...；如目录不同自行改 STAGE/HR/OUT
python paper/scripts/build_fig7_competitor_crops.py
```

它会：
1. 对每个 `(对手, 样本)` 读 SR 全图 + GT 全图，按 **Y 通道 / crop_border=4 / 全图** 算 PSNR/SSIM
   （与 [build_fig7_assets.py](file:///Users/bytedance/WorkSpace/DPRNet/paper/scripts/build_fig7_assets.py)
   完全相同的实现，保证 5 列与 Bicubic/DPRNet 同口径可比）；
2. 按固定框裁 96×96、NEAREST ×3 放大，存成 `<tag>_<Method>_crop.png`；
3. 写 `competitor_metrics.csv`（dataset, sample, method, psnr, ssim）。

输出在 `/hy-tmp/fig7_out/`：15 张 crop + 1 个 CSV。

**自检**：若脚本打印 `[warn] ... size ... != GT`，说明你放进暂存目录的是裁好的小块或下采样图，
必须换成 **HR 分辨率的 SR 全图** 重来。

---

## 4. 拷回本地 + 填图

把产出拷回论文仓库的资产目录：

```bash
# 在本地机执行（按你的训练机连接方式调整）
scp '<训练机>:/hy-tmp/fig7_out/*_crop.png' \
    /Users/bytedance/WorkSpace/DPRNet/paper/figures/fig7_assets/
scp '<训练机>:/hy-tmp/fig7_out/competitor_metrics.csv' \
    /Users/bytedance/WorkSpace/DPRNet/paper/figures/fig7_assets/
```

之后的 LaTeX 改图（把单占位列扩成 5 个对手列、填入 CSV 里的 PSNR/SSIM、并相应改窄 subfigure
宽度）属于回到论文仓库后的排版工作，本手册到此即完成了“训练机上的全部推理与后处理”。

---

## 5. 完成判定清单（Checklist）

- [ ] 5 个对手各自仓库已 clone、官方 ×4 权重已下载到位
- [ ] 5 个对手对 3 个难样本各产出 SR 全图（共 15 张），命名为 `<dataset>_<sample>.png`
- [ ] 15 张全图分别放入 `/hy-tmp/fig7_stage/<Method>/`
- [ ] `build_fig7_competitor_crops.py` 跑通，无 `[warn] size` / `[skip] missing`
- [ ] `/hy-tmp/fig7_out/` 下生成 15 张 `_crop.png` + `competitor_metrics.csv`
- [ ] crop 与 CSV 已拷回 `paper/figures/fig7_assets/`

---

## 附：仓库与权重速查表

| 方法 | 类型 | 仓库 | 权重位置 | 测试框架 |
|---|---|---|---|---|
| IMDN | CNN | https://github.com/Zheng222/IMDN | 仓库内 `checkpoints/IMDN_x4.pth` | 独立 `test_IMDN.py` |
| RFDN | CNN | https://github.com/njulj/RFDN | 仓库内 `trained_model/` | 独立 `test.py`（需改路径） |
| SwinIR-light | Transformer | https://github.com/JingyunLiang/SwinIR | Releases：`002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth` | 独立 `main_test_swinir.py` |
| SRFormer-light | Transformer | https://github.com/HVision-NKU/SRFormer | [Google Drive](https://drive.google.com/drive/folders/1D5ER_HwYJoyZCcrKVstwE-iEl0hXulwd) | BasicSR `basicsr/test.py` |
| CATANet | Transformer | https://github.com/EquationWalker/CATANet | [Release v0.0](https://github.com/EquationWalker/CATANet/releases/tag/v0.0) | BasicSR `basicsr/test.py` |
