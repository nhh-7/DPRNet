# Fig.5 / Fig.8 路由可视化推理操作手册（训练机自包含版）

> 目标：在训练机上用 **DPRNet 自己的 x4 权重**（统一报告点 `net_g_250000`）跑一个脚本，
> 产出论文 §3.6 路由可解释性需要的两张图的资产：
> - **Fig.5**：token 置信度 `x^s` 直方图（CSV，本地用 pgfplots 出矢量图）。
> - **Fig.8**：路由聚类伪彩图（PNG，belong_idx 还原到 LR 的 H×W 上色）。
>
> 你在训练机上只做一件事：照本文档跑 2 条命令，把 `fig58_stage/` 整个目录回传。
> **后续写 .tex、串进 main.tex 由我在本地完成。**

---

## 0. 全局约定（务必先读）

| 项 | 值 |
|---|---|
| 模型 | DPRNet（= 本仓库改造后的 CATANet 默认结构），**不是官方原版 CATANet** |
| 权重 | `experiments/train_CATANet_x4_finetune/models/net_g_250000.pth`（x4 统一报告点） |
| Scale | ×4 |
| 脚本 | `CATANet/scripts/build_routing_figures.py`（已随仓库，无需下载，自包含） |
| 测试数据根 | `/hy-tmp/TestDataSR/LR/LRBI/<dataset>/x4/`（**只用 LR**，不需要 HR） |
| 运行目录 | **必须在 `CATANet/` 目录下跑**（脚本要 `import basicsr`） |
| 你要产出的 | `fig58_stage/` 目录（CSV + PNG），回传本地 |
| 你**不需要**做的 | 写 LaTeX、画矢量图、对齐排版——这些我本地做 |

**难样本与 block 选择**（与 Fig.7、§3.6 口径对齐）：
- Fig.8 聚类样本：`Urban100/img_092`、`Urban100/img_024`、`Manga109/ThatsIzumiko_000`。
- 可视化的 TAB block：`0,3,7`（首/中/末，覆盖 M=16/128/128 三种原型数，展示浅→深路由演化）。

**环境**：用你训练 DPRNet 的那个 conda 环境即可（已装 torch + basicsr）。
脚本额外只用到 `cv2`、`numpy`（BasicSR 依赖里都有）；`matplotlib` 可选，没有也不报错（只是不出 PNG 预览，CSV 照常生成）。

---

## 1. 前置检查（30 秒）

```bash
cd CATANet                              # 关键：必须在此目录，否则 import basicsr 失败
ls scripts/build_routing_figures.py    # 确认脚本在
ls experiments/train_CATANet_x4_finetune/models/net_g_250000.pth   # 确认权重在
ls /hy-tmp/TestDataSR/LR/LRBI/Urban100/x4 | head   # 确认 LR 数据在（应看到 img_001_x4.png ...）
```

三项都在就可以往下跑。若权重路径不同，把下面命令里的 `--weights` 改成实际路径即可。

---

## 2. 跑两条命令

> 脚本一次运行同时产出该数据集的 **Fig.5 直方图（全量 LR 图累加）** 和 **Fig.8 聚类图（指定样本）**。
> 直方图按数据集分别出，所以 Urban100、Manga109 各跑一次。

### 2.1 Urban100（全量直方图 + img_092 / img_024 聚类）

```bash
python scripts/build_routing_figures.py \
    --weights experiments/train_CATANet_x4_finetune/models/net_g_250000.pth \
    --scale 4 \
    --lr-dir /hy-tmp/TestDataSR/LR/LRBI/Urban100/x4 \
    --dataset-name Urban100 \
    --samples img_092,img_024 \
    --blocks 0,3,7 \
    --out-dir fig58_stage
```

### 2.2 Manga109（全量直方图 + ThatsIzumiko_000 聚类）

```bash
python scripts/build_routing_figures.py \
    --weights experiments/train_CATANet_x4_finetune/models/net_g_250000.pth \
    --scale 4 \
    --lr-dir /hy-tmp/TestDataSR/LR/LRBI/Manga109/x4 \
    --dataset-name Manga109 \
    --samples ThatsIzumiko_000 \
    --blocks 0,3,7 \
    --out-dir fig58_stage
```

两条命令写到同一个 `--out-dir fig58_stage`，文件名带数据集前缀，不会互相覆盖。

> 跑得慢/显存紧张？给 Fig.5 限张数：加 `--hist-limit 50`（只用前 50 张算直方图，趋势已足够稳）。
> 只想补聚类图、跳过直方图：加 `--skip-hist`。

---

## 3. 产物清单（跑完应有这些）

`fig58_stage/` 下：

```
fig58_stage/
├── fig5_Urban100_xscore_hist.csv          # 每 block 直方图：bin_left,bin_right,count_b0..count_b7
├── fig5_Urban100_xscore_stats.csv         # 每 block：num_tokens,uniform_floor,mean,std,median,p10,p90,frac_above_uniform,router_scale
├── fig5_Urban100_xscore_hist.png          # 预览（matplotlib 可用时才有）
├── fig5_Manga109_xscore_hist.csv
├── fig5_Manga109_xscore_stats.csv
├── fig5_Manga109_xscore_hist.png          # 同上，可选
├── fig8_Urban100_img_092_lr.png           # 对应 LR 输入（nearest 放大）
├── fig8_Urban100_img_092_b0_cluster.png   # block 0 聚类伪彩图
├── fig8_Urban100_img_092_b3_cluster.png
├── fig8_Urban100_img_092_b7_cluster.png
├── fig8_Urban100_img_024_lr.png
├── fig8_Urban100_img_024_b0_cluster.png
├── fig8_Urban100_img_024_b3_cluster.png
├── fig8_Urban100_img_024_b7_cluster.png
├── fig8_Manga109_ThatsIzumiko_000_lr.png
├── fig8_Manga109_ThatsIzumiko_000_b0_cluster.png
├── fig8_Manga109_ThatsIzumiko_000_b3_cluster.png
└── fig8_Manga109_ThatsIzumiko_000_b7_cluster.png
```

**自检要点**：
- 4 个 CSV 一定要有（PNG 预览有没有都行）。
- 每个聚类 PNG 打开应是**彩色分块**（不同原型不同颜色），且分块边界大致跟 LR 图里的纹理/结构对齐——
  说明路由确实在按内容分簇。若整张一个颜色，说明该 block 路由塌缩，记下来告诉我（可能要换 block）。
- `fig5_*_stats.csv` 里 `mean` 应明显高于 `uniform_floor`（=1/M），说明 softmax 分配有信息（§3.6 的 C3 证据）。

---

## 4. 回传（训练机最后一步）

在本地机执行（按你的连接方式调整主机名）：

```bash
scp -r '<训练机>:<CATANet 绝对路径>/fig58_stage' \
    /Users/bytedance/WorkSpace/DPRNet/paper/figures/fig58_stage
```

回传完告诉我一声即可——**把 CSV 写成 pgfplots 的 `fig5_xscore_hist.tex`、把聚类 PNG 排成 `fig8_cluster.tex`、
串进 main.tex、编译验证，全部由我在本地完成。** 你在训练机上的工作到此结束。

---

## 5. 完成判定清单

- [ ] 在 `CATANet/` 目录下、用 DPRNet x4 `net_g_250000` 权重跑通 2 条命令
- [ ] `fig58_stage/` 下 4 个 CSV 齐全（Urban100/Manga109 各 hist+stats）
- [ ] 12 张 Fig.8 PNG 齐全（3 样本 × [1 LR + 3 block 聚类]）
- [ ] 抽查聚类 PNG 为彩色分块且与纹理对齐；stats 的 mean > uniform_floor
- [ ] 已 scp 回本地 `paper/figures/fig58_stage/`，并通知我接手

---

## 附：脚本参数速查

| 参数 | 含义 | 默认 |
|---|---|---|
| `--weights` | checkpoint 路径（兼容 params / params_ema / 裸 state_dict） | 必填 |
| `--scale` | 放大倍数 | 4 |
| `--lr-dir` | 单个数据集的 LR 目录 | 必填 |
| `--dataset-name` | 文件名用的数据集标签 | 从 lr-dir 推断 |
| `--samples` | Fig.8 聚类样本 stem，逗号分隔（空=跳过 Fig.8） | 空 |
| `--blocks` | 要可视化的 TAB block 序号，逗号分隔 | `0,3,7` |
| `--out-dir` | 产物目录 | `fig58_stage` |
| `--bins` | Fig.5 直方图 bin 数（[0,1] 区间） | 50 |
| `--hist-limit` | Fig.5 最多用几张图（0=全量） | 0 |
| `--vis-scale` | 聚类/LR 预览图的 nearest 放大倍数 | 4 |
| `--skip-hist` | 只出 Fig.8、跳过 Fig.5 | 关 |
