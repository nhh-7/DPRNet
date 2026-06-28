# Fig.7 对手列推理操作手册（训练机自包含版）

> 目标：在训练机上为 Fig.7（×4 视觉对比）的 **5 个对手** 各产出 3 张难样本的 **SR 全图**
> （HR 分辨率，整张，不要裁剪、不要算指标）。全部跑完后把图回传，**裁剪与 PSNR/SSIM 计算由我在本地统一完成**。
>
> 你在训练机上只做一件事：照本文档把 15 张 SR 全图生成出来并回传。

---

## 0. 全局约定（务必先读）

| 项 | 值 |
|---|---|
| Scale | ×4（Fig.7 只做 ×4） |
| 需要的样本 | `Urban100/img_092`、`Urban100/img_024`、`Manga109/ThatsIzumiko_000`（共 3 张，2 城市 + 1 漫画） |
| 测试数据根 | `/hy-tmp/TestDataSR/HR/<dataset>/x4/`、`/hy-tmp/TestDataSR/LR/LRBI/<dataset>/x4/` |
| 你要产出的 | 每个对手 3 张 **SR 全图**，命名为 `<dataset>_<sample>.png`，放进 `/hy-tmp/fig7_stage/<Method>/` |
| 你**不需要**做的 | 裁剪、放大、算 PSNR/SSIM、对齐 crop——这些我本地做 |

3 张样本的确切源文件（HR 与 LR **同名**，都叫 `<sid>_x4.png`）：

```
Urban100/img_092            HR: /hy-tmp/TestDataSR/HR/Urban100/x4/img_092_x4.png
                            LR: /hy-tmp/TestDataSR/LR/LRBI/Urban100/x4/img_092_x4.png
Urban100/img_024            HR: /hy-tmp/TestDataSR/HR/Urban100/x4/img_024_x4.png
                            LR: /hy-tmp/TestDataSR/LR/LRBI/Urban100/x4/img_024_x4.png
Manga109/ThatsIzumiko_000   HR: /hy-tmp/TestDataSR/HR/Manga109/x4/ThatsIzumiko_000_x4.png
                            LR: /hy-tmp/TestDataSR/LR/LRBI/Manga109/x4/ThatsIzumiko_000_x4.png
```

### ⚠️ 关键：本数据集的命名特点，决定每个对手怎么改

本数据集里 **HR 和 LR 的文件名完全一样**，都是 `<sid>_x4.png`。这与多数对手脚本的默认假设冲突，
**必须按下表处理，否则脚本会找不到 LR、静默报错或读错图**：

| 对手 | 框架 | LR 名匹配方式 | 本数据集要怎么办 |
|---|---|---|---|
| CATANet | BasicSR | `filename_tmpl` 映射 GT→LR | 设 `filename_tmpl: '{}'`（GT、LR 同名即可，**天然匹配**） |
| SRFormer-light | BasicSR | 同上 | 同上，加 `filename_tmpl: '{}'` |
| SwinIR-light | 独立脚本 | GT 名后**追加** `x4` 当 LR 名 | GT=`<sid>_x4.png` 会去找 `<sid>_x4x4.png` ⇒ **必须做软链/改名暂存**（见 2.3） |
| IMDN | 独立脚本 | 同样追加 `x4` | 同上问题 ⇒ 本文档给**最小推理脚本**绕过（见 2.4） |
| RFDN | 独立脚本 | 写死 DIV2K、保存被注释 | 本文档给**最小推理脚本**直接读我们的 LR（见 2.5） |

---

## 1. 准备暂存目录

```bash
mkdir -p /hy-tmp/fig7_stage/{IMDN,RFDN,SwinIR-light,SRFormer-light,CATANet}
```

最终你要凑齐 15 张全图（每个对手 3 张），结构：

```
/hy-tmp/fig7_stage/
├── CATANet/         Urban100_img_092.png  Urban100_img_024.png  Manga109_ThatsIzumiko_000.png
├── SRFormer-light/  （同上 3 张）
├── SwinIR-light/    （同上 3 张）
├── IMDN/            （同上 3 张）
└── RFDN/            （同上 3 张）
```

---

## 2. 五个对手：逐个详细步骤

> 建议每个对手单独 conda 环境，避免 PyTorch 版本互相污染。命令默认在该对手仓库根目录执行。

---

### 2.1 CATANet（最强对手，BasicSR）★必做

- **仓库**：https://github.com/EquationWalker/CATANet
- **权重**：https://github.com/EquationWalker/CATANet/releases/tag/v0.0 下载 ×4 权重，放入 `./pretrained_models/`
  （文件名以 release 页为准，如 `CATANet_SRx4.pth` 之类；下一步 yml 里填它）
- **环境**：Python 3.9 / PyTorch ≥ 2.2

```bash
git clone https://github.com/EquationWalker/CATANet && cd CATANet
conda create -n catanet python=3.9 -y && conda activate catanet
pip install -r requirements.txt
python setup.py develop
# 把官方 x4 权重放到 ./pretrained_models/ 下
```

**编辑 `options/test/test_CATANet_x4.yml`**——只保留我们需要的两个数据集，并按下方逐行改：

```yaml
datasets:
  test_1:
    name: Urban100
    type: PairedImageDataset
    dataroot_gt: /hy-tmp/TestDataSR/HR/Urban100/x4
    dataroot_lq: /hy-tmp/TestDataSR/LR/LRBI/Urban100/x4
    filename_tmpl: '{}'          # 关键：GT/LR 同名，保持 '{}'
    io_backend:
      type: disk
  test_2:
    name: Manga109
    type: PairedImageDataset
    dataroot_gt: /hy-tmp/TestDataSR/HR/Manga109/x4
    dataroot_lq: /hy-tmp/TestDataSR/LR/LRBI/Manga109/x4
    filename_tmpl: '{}'
    io_backend:
      type: disk
  # 其余 Set5/Set14/B100 整段删掉或注释（Fig.7 用不到，省时间）

network_g:
  type: CATANet
  upscale: 4

path:
  pretrain_network_g: ./pretrained_models/<官方 x4 权重文件名>.pth   # 改成你下载的文件名
  strict_load_g: true

val:
  save_img: true               # 关键：必须 true，否则不存图
  suffix: CATANetOrig          # 给个能区分的后缀，避免和我们 DPRNet 混淆
```

```bash
python basicsr/test.py -opt options/test/test_CATANet_x4.yml
```

- **SR 全图输出**：`results/test_CATANet_x4/visualization/<dataset>/<sid>_x4_CATANetOrig.png`
- **挑图入库**（3 张）：
```bash
cp results/test_CATANet_x4/visualization/Urban100/img_092_x4_CATANetOrig.png        /hy-tmp/fig7_stage/CATANet/Urban100_img_092.png
cp results/test_CATANet_x4/visualization/Urban100/img_024_x4_CATANetOrig.png        /hy-tmp/fig7_stage/CATANet/Urban100_img_024.png
cp results/test_CATANet_x4/visualization/Manga109/ThatsIzumiko_000_x4_CATANetOrig.png /hy-tmp/fig7_stage/CATANet/Manga109_ThatsIzumiko_000.png
```

> ⚠️ 这是**官方原版 CATANet 权重**，与我们 DPRNet 不同。别和本仓库 `results/CATANet/...`（那是 DPRNet 的输出）混淆，所以这里用了 `CATANetOrig` 后缀。

---

### 2.2 SRFormer-light（Transformer 对手，BasicSR）

- **仓库**：https://github.com/HVision-NKU/SRFormer
- **权重**：Google Drive（repo README 内链接），取 **light ×4** 文件，文件名 `SRFormerLight_SRx4_DIV2K.pth`，放入 `./PretrainModel/`
- **环境**：Python 3.8 / PyTorch ≥ 1.7

```bash
git clone https://github.com/HVision-NKU/SRFormer && cd SRFormer
conda create -n srformer python=3.8 -y && conda activate srformer
pip install -r requirements.txt
python setup.py develop
# 把 SRFormerLight_SRx4_DIV2K.pth 放到 ./PretrainModel/ 下
```

**编辑 `options/test/SRFormer/test_SRFormer_light_DIV2Ksrx4.yml`**——只保留 Urban100/Manga109，逐行改：

```yaml
datasets:
  test_1:
    name: Urban100
    type: PairedImageDataset
    dataroot_gt: /hy-tmp/TestDataSR/HR/Urban100/x4
    dataroot_lq: /hy-tmp/TestDataSR/LR/LRBI/Urban100/x4
    filename_tmpl: '{}'          # 关键：补上这一行（原 yml 没有），保证 GT/LR 同名匹配
    io_backend:
      type: disk
  test_2:
    name: Manga109
    type: PairedImageDataset
    dataroot_gt: /hy-tmp/TestDataSR/HR/Manga109/x4
    dataroot_lq: /hy-tmp/TestDataSR/LR/LRBI/Manga109/x4
    filename_tmpl: '{}'
    io_backend:
      type: disk
  # Set5/Set14/B100 整段删掉

path:
  pretrain_network_g: PretrainModel/SRFormerLight_SRx4_DIV2K.pth
  strict_load_g: true

val:
  save_img: true               # 关键：原 yml 是 false，必须改成 true
  suffix: ~
```

> network_g 段（window_size: 16、embed_dim: 60 等）**保持原样不要动**，那是 light×4 的结构定义。

```bash
python basicsr/test.py -opt options/test/SRFormer/test_SRFormer_light_DIV2Ksrx4.yml
```

- **SR 全图输出**：`results/SRFormer_light_X4/visualization/<dataset>/<sid>_x4.png`
- **挑图入库**：
```bash
cp results/SRFormer_light_X4/visualization/Urban100/img_092_x4.png        /hy-tmp/fig7_stage/SRFormer-light/Urban100_img_092.png
cp results/SRFormer_light_X4/visualization/Urban100/img_024_x4.png        /hy-tmp/fig7_stage/SRFormer-light/Urban100_img_024.png
cp results/SRFormer_light_X4/visualization/Manga109/ThatsIzumiko_000_x4.png /hy-tmp/fig7_stage/SRFormer-light/Manga109_ThatsIzumiko_000.png
```

---

### 2.3 SwinIR-light（Transformer 对手，独立脚本 + **必须改名暂存**）

- **仓库**：https://github.com/JingyunLiang/SwinIR
- **权重**：`002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth`（https://github.com/JingyunLiang/SwinIR/releases ）
- **环境**：Python 3.8+ / PyTorch ≥ 1.7

```bash
git clone https://github.com/JingyunLiang/SwinIR && cd SwinIR
conda create -n swinir python=3.8 -y && conda activate swinir
pip install torch torchvision timm opencv-python numpy
mkdir -p model_zoo/swinir
# 下载 002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth 到 model_zoo/swinir/
```

**关键步骤：改名暂存**。SwinIR 用 GT 名后追加 `x4` 来找 LR（即 GT=`A.png`⇒LR=`Ax4.png`）。
我们的 GT 名是 `<sid>_x4.png`，会被去找 `<sid>_x4x4.png`（不存在）。所以先把 3 张样本改名暂存成它要的形式：

```bash
mkdir -p /hy-tmp/fig7_in/gt /hy-tmp/fig7_in/lr
for s in Urban100/img_092 Urban100/img_024 Manga109/ThatsIzumiko_000; do
  ds=${s%/*}; sid=${s#*/}
  cp /hy-tmp/TestDataSR/HR/$ds/x4/${sid}_x4.png    /hy-tmp/fig7_in/gt/${sid}.png      # GT 去掉 _x4
  cp /hy-tmp/TestDataSR/LR/LRBI/$ds/x4/${sid}_x4.png /hy-tmp/fig7_in/lr/${sid}x4.png  # LR 命名为 <sid>x4.png
done
```

跑（注意是 `main_test_swinir.py`，不是 basicsr）：

```bash
python main_test_swinir.py --task lightweight_sr --scale 4 \
  --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth \
  --folder_gt /hy-tmp/fig7_in/gt \
  --folder_lq /hy-tmp/fig7_in/lr
```

- **SR 全图输出**：`results/swinir_lightweight_sr_x4/<sid>_SwinIR.png`
- **挑图入库**（注意把 sid 归回它原本的 dataset）：
```bash
cp results/swinir_lightweight_sr_x4/img_092_SwinIR.png          /hy-tmp/fig7_stage/SwinIR-light/Urban100_img_092.png
cp results/swinir_lightweight_sr_x4/img_024_SwinIR.png          /hy-tmp/fig7_stage/SwinIR-light/Urban100_img_024.png
cp results/swinir_lightweight_sr_x4/ThatsIzumiko_000_SwinIR.png /hy-tmp/fig7_stage/SwinIR-light/Manga109_ThatsIzumiko_000.png
```

> 它脚本会打印 PSNR/SSIM，**忽略即可**，我本地会统一重算。

---

### 2.4 IMDN（CNN 对手，独立脚本 → 用最小推理脚本绕过命名坑）

- **仓库**：https://github.com/Zheng222/IMDN
- **权重**：仓库自带 `checkpoints/IMDN_x4.pth`
- **环境**：PyTorch 1.x

IMDN 自带 `test_IMDN.py` 既有“LR 名追加 x4”的坑，又会 `import utils` → `from skimage.measure import compare_psnr`，
而该函数在新版 skimage 已删除（装 skimage 反而要降级到 0.15、再下载一堆包）。
**正解：完全不 import IMDN 的 `utils`**，把它用到的两个小函数（加载权重、tensor 转图）内联进脚本，
这样只依赖 torch + opencv + numpy，零 skimage 依赖。在 IMDN 仓库根目录新建 `infer_fig7.py`：

```python
import os, cv2, numpy as np, torch
from model import architecture          # IMDN 仓库自带，不要 import utils

model = architecture.IMDN(upscale=4)
# 直接用 torch.load 读权重（IMDN_x4.pth 是纯 state_dict），并兼容可能的 module. 前缀
sd = torch.load('checkpoints/IMDN_x4.pth', map_location='cpu')
sd = sd.get('state_dict', sd)
sd = {k.replace('module.', ''): v for k, v in sd.items()}
model.load_state_dict(sd, strict=True)
model.eval().cuda()

PAIRS = [('Urban100', 'img_092'), ('Urban100', 'img_024'), ('Manga109', 'ThatsIzumiko_000')]
LR = '/hy-tmp/TestDataSR/LR/LRBI'
OUT = '/hy-tmp/fig7_stage/IMDN'
os.makedirs(OUT, exist_ok=True)

for ds, sid in PAIRS:
    p = f'{LR}/{ds}/x4/{sid}_x4.png'
    im = cv2.imread(p, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]          # BGR->RGB
    x = torch.from_numpy(np.transpose(im / 255.0, (2, 0, 1))[None]).float().cuda()
    with torch.no_grad():
        out = model(x)[0].detach().cpu().clamp_(0, 1).numpy()      # CHW, [0,1]
    out_img = (np.transpose(out, (1, 2, 0)) * 255.0).round().astype(np.uint8)  # HWC RGB
    cv2.imwrite(f'{OUT}/{ds}_{sid}.png', out_img[:, :, [2, 1, 0]]) # RGB->BGR 存盘
    print('saved', f'{OUT}/{ds}_{sid}.png')
```

```bash
git clone https://github.com/Zheng222/IMDN && cd IMDN
conda create -n imdn python=3.7 -y && conda activate imdn
pip install torch torchvision numpy opencv-python   # 注意：不装 scikit-image
# 新建上面的 infer_fig7.py
python infer_fig7.py
```

- **输出**：脚本已直接写成 `/hy-tmp/fig7_stage/IMDN/<dataset>_<sample>.png`，无需再挑图。

---

### 2.5 RFDN（CNN 对手，独立脚本 → 用最小推理脚本）

- **仓库**：https://github.com/njulj/RFDN
- **权重**：仓库自带 `trained_model/RFDN_AIM.pth`（×4）
- **环境**：PyTorch 1.x

RFDN 自带 `test.py` 写死了 DIV2K 目录、且保存结果那行被注释掉；它的 `utils.utils_image`
顶部还会 `import matplotlib`（推理根本用不到，却会因缺包直接崩）。
**正解：完全不 import RFDN 的 `utils`**，用 opencv 自己读写、内联归一化与还原（与上面 IMDN 脚本同一套）。
依赖只剩 torch + opencv + numpy，零 matplotlib/skimage。在 RFDN 仓库根目录新建 `infer_fig7.py`：

```python
import os, cv2, numpy as np, torch
from RFDN import RFDN                     # RFDN 仓库自带，不要 import utils

model = RFDN()                                                     # 默认 upscale=4
sd = torch.load('trained_model/RFDN_AIM.pth', map_location='cpu')
sd = sd.get('state_dict', sd)
sd = {k.replace('module.', ''): v for k, v in sd.items()}
model.load_state_dict(sd, strict=True)
model.eval().cuda()
for v in model.parameters():
    v.requires_grad = False

PAIRS = [('Urban100', 'img_092'), ('Urban100', 'img_024'), ('Manga109', 'ThatsIzumiko_000')]
LR = '/hy-tmp/TestDataSR/LR/LRBI'
OUT = '/hy-tmp/fig7_stage/RFDN'
os.makedirs(OUT, exist_ok=True)

for ds, sid in PAIRS:
    p = f'{LR}/{ds}/x4/{sid}_x4.png'
    im = cv2.imread(p, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]          # BGR->RGB
    x = torch.from_numpy(np.transpose(im / 255.0, (2, 0, 1))[None]).float().cuda()
    with torch.no_grad():
        out = model(x)[0].detach().cpu().clamp_(0, 1).numpy()      # CHW, [0,1]
    out_img = (np.transpose(out, (1, 2, 0)) * 255.0).round().astype(np.uint8)  # HWC RGB
    cv2.imwrite(f'{OUT}/{ds}_{sid}.png', out_img[:, :, [2, 1, 0]]) # RGB->BGR 存盘
    print('saved', f'{OUT}/{ds}_{sid}.png')
```

```bash
git clone https://github.com/njulj/RFDN && cd RFDN
conda create -n rfdn python=3.8 -y && conda activate rfdn
pip install torch torchvision numpy opencv-python   # 不装 matplotlib / scikit-image
# 新建上面的 infer_fig7.py
python infer_fig7.py
```

- **输出**：脚本已直接写成 `/hy-tmp/fig7_stage/RFDN/<dataset>_<sample>.png`，无需再挑图。

> RFDN_AIM 是 AIM2020 高效 SR 挑战的官方 ×4 权重；Fig.7 是定性对比，用官方权重的真实 SR 即可。

---

## 3. 自检 + 回传（训练机最后一步）

确认 15 张全图齐全：

```bash
ls -1 /hy-tmp/fig7_stage/*/*.png    # 期望 15 行
```

每个对手应有且仅有这 3 个文件名：
`Urban100_img_092.png`、`Urban100_img_024.png`、`Manga109_ThatsIzumiko_000.png`

**自检要点**：打开任意一张，确认是 **HR 分辨率的整张 SR 图**（不是裁好的小块、不是 LR 尺寸）。

回传到本地（在本地机执行，按你的连接方式调整）：

```bash
scp -r '<训练机>:/hy-tmp/fig7_stage' \
    /Users/bytedance/WorkSpace/DPRNet/paper/figures/fig7_assets/_raw_competitor
```

回传完成后告诉我一声即可——**裁剪到统一 crop 框、NEAREST ×3 放大、Y 通道 PSNR/SSIM（crop_border=4，全图）、与现有 Bicubic/DPRNet/GT 列对齐，全部由我在本地用同一套脚本完成**，保证口径一致。你在训练机上的工作到此结束。

---

## 4. 完成判定清单

- [ ] 5 个对手仓库已 clone、官方 ×4 权重到位
- [ ] CATANet / SRFormer：yml 已按上文改（数据路径 + `filename_tmpl: '{}'` + `save_img: true`），跑通
- [ ] SwinIR：已建 `/hy-tmp/fig7_in/{gt,lr}` 改名暂存，跑通
- [ ] IMDN / RFDN：已建 `infer_fig7.py`，跑通
- [ ] `/hy-tmp/fig7_stage/<Method>/` 下各 3 张，共 15 张，均为 HR 分辨率全图
- [ ] 已 scp 回本地 `_raw_competitor/`，并通知我接手

---

## 附：仓库与权重速查表

| 方法 | 类型 | 仓库 | 权重 | 推理方式 |
|---|---|---|---|---|
| CATANet | Transformer | https://github.com/EquationWalker/CATANet | [Release v0.0](https://github.com/EquationWalker/CATANet/releases/tag/v0.0) 的 ×4 | BasicSR，改 yml |
| SRFormer-light | Transformer | https://github.com/HVision-NKU/SRFormer | `SRFormerLight_SRx4_DIV2K.pth`（Google Drive） | BasicSR，改 yml |
| SwinIR-light | Transformer | https://github.com/JingyunLiang/SwinIR | `002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth`（Releases） | 独立脚本 + 改名暂存 |
| IMDN | CNN | https://github.com/Zheng222/IMDN | 仓库内 `checkpoints/IMDN_x4.pth` | 最小推理脚本 |
| RFDN | CNN | https://github.com/njulj/RFDN | 仓库内 `trained_model/RFDN_AIM.pth` | 最小推理脚本 |
