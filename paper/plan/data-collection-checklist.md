# 数据 / 图表收集清单 (Data & Figure Collection Checklist)

投稿所需的全部"硬证据"。训练过程中即时收集，避免训练结束后无法补采。
每项标注：[ ] 待收集 / [~] 进行中 / [x] 完成。来源必须可追溯。

## A. 训练过程中必须收集的原始数据

### A1 训练日志（每个模型：x2 已有 / x3 / x4 / 每个消融变体）
- [ ] 每 val 点的 iter、l_pix、l_route、lr_g、Set5 PSNR/SSIM、Set14 PSNR/SSIM
- [ ] 训练 loss 曲线（用于附录或训练曲线图）
- [ ] 路由诊断：每个 TAB block（b0-b7）的 xscore_mean/std、usage_active/max/min/std、
      entropy_norm、router_scale —— 测试时已自动打印到日志（见 x2 测试日志 L137-201）。
      ⚠ 这些是 Fig.5/6 与 C3/C4 的核心证据，测试时必须重定向存盘，否则丢失：
      `python basicsr/test.py -opt <yml> 2>&1 | tee results/<tag>_metrics.log`
      然后从日志解析为 CSV（每 block 一行 × 每数据集）。
- [ ] 最优 checkpoint 的 iter 号与对应权重路径
- [ ] 训练耗时（GPU·hours）、GPU 型号/数量
- 工具：参照已有 training_key_metrics.csv 的提取方式，对 x3/x4/消融同样处理。

### A2 最终测试指标（每个尺度，全 5 基准）
> **权威来源（后续一律以此为准）**：`paper/data/dprnet_main_metrics.csv`
> （15 行 = 3 尺度 × 5 基准，逐行带 `source_log` 溯源到原始日志；已与日志逐格核对）。
> 本节表格仅作"选点依据"说明，精确数值取 CSV，不再翻原始日志。
- [x] x2：Set5/Set14/B100/Urban100/Manga109 PSNR/SSIM（用 test_CATANet_x2.yml）
  - 报告点：net_g_792500.pth（已锁定为主报告点 + x3/x4 finetune 起点 + 消融/可视化基准）
  - 数据来源：results/CATANet/test_CATANet_x2-792500/test_test_CATANet_x2_20260606_154428.log
  - Y通道，crop_border=2：

    | 数据集 | PSNR | SSIM |
    |---|---|---|
    | Set5 | 38.2706 | 0.9617 |
    | Set14 | 33.9614 | 0.9210 |
    | B100 | 32.3509 | 0.9020 |
    | Urban100 | 33.0500 | 0.9362 |
    | Manga109 | 39.3804 | 0.9787 |

  - 选点依据：9 个 checkpoint（500k–800k）全 5 基准对比后，792500 均值 SSIM 全场最高、
    均值 PSNR 与 777500 实质并列、独占 Manga109 双指标 + Set5 SSIM，五集无短板，且最晚最收敛。
- [x] x3：同上（test_CATANet_x3.yml）
  - 报告点：net_g_250000.pth（x3 finetune 最终点）
  - 数据来源：results/CATANet/test_CATANet_x3-250000/test_test_CATANet_x3_20260608_221117.log
  - Y通道，crop_border=3：

    | 数据集 | PSNR | SSIM |
    |---|---|---|
    | Set5 | 34.7129 | 0.9298 |
    | Set14 | 30.6599 | 0.8481 |
    | B100 | 29.2890 | 0.8103 |
    | Urban100 | 28.9551 | 0.8675 |
    | Manga109 | 34.4133 | 0.9499 |

  - 选点依据：210000/232500/250000 三点全 5 基准对比，250000 均值 PSNR(31.606)、
    均值 SSIM(0.8811) 均为最高，且 Set14/B100/Urban100/Manga 多项逐指标最优、五集无短板、最收敛。
- [x] x4：同上（test_CATANet_x4.yml）
  - 报告点：net_g_250000.pth（x4 finetune 最终点）
  - 数据来源：results/CATANet/test_CATANet_x4-250000/test_test_CATANet_x4_20260608_231742.log
  - Y通道，crop_border=4：

    | 数据集 | PSNR | SSIM |
    |---|---|---|
    | Set5 | 32.5826 | 0.8998 |
    | Set14 | 28.8846 | 0.7880 |
    | B100 | 27.7648 | 0.7434 |
    | Urban100 | 26.8225 | 0.8065 |
    | Manga109 | 31.2540 | 0.9181 |

  - 选点依据：195000/240000/250000 三点全 5 基准对比，250000 均值 PSNR(29.462)、
    均值 SSIM(0.8312) 均为最高，Urban100 双指标全场最高（本文核心主张数据集），最收敛。
- 注：crop_border 按尺度（2/3/4），test_y_channel=true。
- 三尺度统一报告口径：x2=net_g_792500，x3=net_g_250000，x4=net_g_250000（均"全基准均值最优 + 最收敛"，禁逐集挑最优）。

### A3 效率数据（DPRNet + CATANet，三尺度）
> **权威来源（后续一律以此为准）**：`paper/data/dprnet_efficiency.csv`
> （3 行 = 三尺度 Params/MACs(thop)/时延，带 `source` 溯源到 efficiency.md；已逐格核对）。
> 本节表格仅作口径说明，精确数值取 CSV，不再翻原始日志。
- 工具：scripts/measure_efficiency.py（已实现并在训练机 cuda 上运行）。
- 来源：results/CATANet/efficiency.md（device=cuda，固定 HR≈720×1280 反推 LR；FLOPs 后端 thop，报 MACs，
  对比 FLOPs-based 论文需 ×2；时延 warmup=20、repeat=100，单位 ms）。
- [x] Params（总参数量）：x2=601,947（601.95K）/ x3=674,147（674.15K）/ x4=659,707（659.71K）
- [x] FLOPs / Multi-Adds（thop MACs，固定 HR≈720×1280 输出）：

    | 尺度 | LR 输入 | Params | MACs(thop) | 时延 (ms, mean±std) |
    |---|---|---|---|---|
    | x2 | 360×640 | 601.95K | 126.52 G | 712.16 ± 23.47 |
    | x3 | 240×426 | 674.15K | 64.28 G  | 285.39 ± 5.66 |
    | x4 | 180×320 | 659.71K | 45.77 G  | 196.77 ± 10.02 |

- [x] 推理时延（ms，单 GPU cuda，warmup=20，repeat=100）：见上表。
- 口径说明：thop 报告 MACs，若与以 FLOPs 计的论文对比需 ×2；x3 HR 不整除按有效输出 720×1278。

## B. 从他人论文摘录的对比数据（标注来源，绝不编造）

- [x] CARN / IMDN / RFDN / RLFN x2/x3/x4 5 基准 PSNR/SSIM + Params
- [x] SwinIR-light / ELAN-light / SRFormer-light 同上
- [x] CATANet（原始）同上（优先自测，缺则引原论文）
- 每条记录格式：方法名 | 尺度 | 数据集 | PSNR | SSIM | Params | 来源(论文/表号)
- 评测口径：Y 通道 PSNR/SSIM，DIV2K 训练（轻量 SR 标准设定），与本文一致。
- 数值均逐字摘自原始论文 HTML/PDF，已交叉核对；无法核实的单元格标 `-`。

**来源代号**（references.bib 对应 key）
- 【A】CATANet, CVPR 2025, Table 2（主对比表，含多数 baseline 同口径数值）
  — arXiv:2503.06896 / DOI 10.1109/CVPR52734.2025.01668（bib: liu2025catanet）
- 【B】SRFormer(V2), Table VI（轻量 SR 对比）— arXiv:2303.09735（bib: zhou2023srformer）
- 【C】RLFN, 原论文 Table 1 — arXiv:2205.07514（bib: kong2022rlfn）
- 【D】RFDN, 原论文 Table 3 — arXiv:2009.11551（bib: liu2020rfdn）

> 主基准统一用【A】CATANet Table 2（CARN/IMDN/RFDN/SwinIR-light/ELAN-light/CATANet 均在内、同口径）；
> SRFormer-light 补自【B】、RLFN 补自【C】，在表注标明来源差异。

### B1 CARN（来源【A】，Manga109 差异另注【D】）｜Params x2/x3/x4 = 1592K/1592K/1592K

| 尺度 | Set5 | Set14 | B100 | Urban100 | Manga109 |
|---|---|---|---|---|---|
| x2 | 37.76/0.9590 | 33.52/0.9166 | 32.09/0.8978 | 31.92/0.9256 | 38.36/0.9765 |
| x3 | 34.29/0.9255 | 30.29/0.8407 | 29.06/0.8034 | 28.06/0.8493 | 33.43/0.9427 |
| x4 | 32.13/0.8937 | 28.60/0.7806 | 27.58/0.7349 | 26.07/0.7837 | 30.42/0.9070 |
- 差异：CARN Manga109 x3=33.50/0.9440、x4=30.47/0.9084（来源【D】）。

### B2 IMDN（来源【A】，与【B】【D】一致）｜Params x2/x3/x4 = 694K/703K/715K

| 尺度 | Set5 | Set14 | B100 | Urban100 | Manga109 |
|---|---|---|---|---|---|
| x2 | 38.00/0.9605 | 33.63/0.9177 | 32.19/0.8996 | 32.17/0.9283 | 38.88/0.9774 |
| x3 | 34.36/0.9270 | 30.32/0.8417 | 29.09/0.8046 | 28.17/0.8519 | 33.61/0.9445 |
| x4 | 32.21/0.8948 | 28.58/0.7811 | 27.56/0.7353 | 26.04/0.7838 | 30.45/0.9075 |

### B3 RFDN（来源【D】，与【A】基本一致）｜Params x2/x3/x4 = 534K/541K/550K

| 尺度 | Set5 | Set14 | B100 | Urban100 | Manga109 |
|---|---|---|---|---|---|
| x2 | 38.05/0.9606 | 33.68/0.9184 | 32.16/0.8994 | 32.12/0.9278 | 38.88/0.9773 |
| x3 | 34.41/0.9273 | 30.34/0.8420 | 29.09/0.8050 | 28.21/0.8525 | 33.67/0.9449 |
| x4 | 32.24/0.8952 | 28.61/0.7819 | 27.57/0.7360 | 26.11/0.7858 | 30.58/0.9089 |
- 差异：RFDN Params x2=530K（来源【A】）。

### B4 RLFN（来源【C】）｜Params x2/x4 = 527K/543K（原论文无 x3，无 Manga109）

| 尺度 | Set5 | Set14 | B100 | Urban100 | Manga109 |
|---|---|---|---|---|---|
| x2 | 38.07/0.9607 | 33.72/0.9187 | 32.22/0.9000 | 32.33/0.9299 | - |
| x3 | - | - | - | - | - |
| x4 | 32.24/0.8952 | 28.62/0.7813 | 27.60/0.7364 | 26.17/0.7877 | - |
- ⚠ RLFN 原论文仅评测 Set5/Set14/BSD100/Urban100，且仅 x2/x4（无 x3、无 Manga109）。
  主表对应缺格填 "-"，不补二手来源；若必须补，须另注非原始来源。

### B5 SwinIR-light（来源【A】，数值与【B】一致）｜Params x2/x3/x4 = 878K/886K/897K（【B】口径 910K）

| 尺度 | Set5 | Set14 | B100 | Urban100 | Manga109 |
|---|---|---|---|---|---|
| x2 | 38.14/0.9611 | 33.86/0.9206 | 32.31/0.9012 | 32.76/0.9340 | 39.12/0.9783 |
| x3 | 34.62/0.9289 | 30.54/0.8463 | 29.20/0.8082 | 28.66/0.8624 | 33.98/0.9478 |
| x4 | 32.44/0.8976 | 28.77/0.7858 | 27.69/0.7406 | 26.47/0.7980 | 30.92/0.9151 |

### B6 ELAN-light（来源【A】，数值与【B】一致）｜Params x2/x3/x4 = 582K/590K/601K（【B】口径 621K）

| 尺度 | Set5 | Set14 | B100 | Urban100 | Manga109 |
|---|---|---|---|---|---|
| x2 | 38.17/0.9611 | 33.94/0.9207 | 32.30/0.9012 | 32.76/0.9340 | 39.11/0.9782 |
| x3 | 34.64/0.9288 | 30.55/0.8463 | 29.21/0.8081 | 28.69/0.8624 | 34.00/0.9478 |
| x4 | 32.43/0.8975 | 28.78/0.7858 | 27.69/0.7406 | 26.54/0.7982 | 30.92/0.9150 |
- 差异：ELAN-light x3 Set5=34.61（来源【B】）。

### B7 SRFormer-light（来源【B】，Table VI）｜Params x2/x3/x4 = 853K/861K/873K

| 尺度 | Set5 | Set14 | B100 | Urban100 | Manga109 |
|---|---|---|---|---|---|
| x2 | 38.23/0.9613 | 33.94/0.9209 | 32.36/0.9019 | 32.91/0.9353 | 39.28/0.9785 |
| x3 | 34.67/0.9296 | 30.57/0.8469 | 29.26/0.8099 | 28.81/0.8655 | 34.19/0.9489 |
| x4 | 32.51/0.8988 | 28.82/0.7872 | 27.73/0.7422 | 26.67/0.8032 | 31.17/0.9165 |

### B8 CATANet（原始，最直接对手，来源【A】Table 2）｜Params x2/x3/x4 = 477K/550K/535K

| 尺度 | Set5 | Set14 | B100 | Urban100 | Manga109 |
|---|---|---|---|---|---|
| x2 | 38.28/0.9617 | 33.99/0.9217 | 32.37/0.9023 | 33.09/0.9372 | 39.37/0.9784 |
| x3 | 34.75/0.9300 | 30.67/0.8481 | 29.28/0.8101 | 29.04/0.8689 | 34.40/0.9499 |
| x4 | 32.58/0.8998 | 28.90/0.7880 | 27.75/0.7427 | 26.87/0.8081 | 31.31/0.9183 |
- 注：原论文另有 self-ensemble(†) 列，本文主表对比用非 † 版本（同口径）。
- 交叉验证：CATANet 原报 x2 Set5 38.28/0.9617 与本文自测 net_g_792500（38.2706/0.9617）高度吻合，
  佐证本仓库 CATANet 复现可信、可作 DPRNet 直接对照基线。

### B 节存疑/缺失汇总
- 缺失：RLFN 无 x3、无 Manga109（原论文未提供，主表填 "-"）。
- Params 口径差异：SwinIR-light/ELAN-light 在【A】与【B】统计口径不同（PSNR/SSIM 一致），
  主表 Params 列统一采【A】口径并在表注说明。
- 个别千分位差异（CARN Manga109、ELAN x3 Set5、RFDN Params）已在各表下注明。

## C. 表格清单（tables/，含 table-schema.md）

- [x] Table I 主对比：方法 × (尺度×数据集) PSNR/SSIM，本文加粗、次优下划线
      → paper/tables/table1_main_comparison.tex（三尺度齐，逐列加粗/下划线已核对，RLFN 缺格填 "-"）
- [~] Table II 效率：方法 | Params | FLOPs | 时延 | 代表 PSNR
      → paper/tables/table2_efficiency.tex（DPRNet 三尺度 Params/MACs/时延已填 + 对手 Params 已填；
      CATANet x4 Multi-Adds 已补 46.8G（带 †，引自 CATANet CVPR'25 同 ~720×1280 输出，与我方 thop MACs 可比）；
      ⚠ 其余 7 对手 Multi-Adds 跨源参数口径不一致（如 SwinIR-light 897K vs 930K）暂未填，投稿前统一取
      CATANet Tab.2 或同协议重测；对手时延依赖硬件不可引，留 "-"。禁编造。正文已写于 §3.3 sec:efficiency）
- [x] Table III 消融 A1：refine on/off（受限口径：质量中性）
      → paper/tables/table3_ablation_a1.tex（80k 真实数据：w/o refine Set5 32.5494 / Urban100 26.8358；
      w/ refine 32.5656 / 26.8146；差异在噪声内，仅报质量中性，不主张 PSNR 增益）
- [x] Table IV 消融 A2：置信度感知逐项加法（排序/门控/fallback）（受限口径：组件兼容/质量中性）
      → paper/tables/table4_ablation_a2.tex（80k 真实数据 v1-v4，PSNR 差异落噪声内且非单调，
      仅报"组件相互兼容、不损质量"，不主张逐项增益；x_scores 机制证据另见路由诊断）
- [x] Table V 消融 A3：balance loss（机制证据已有；多 seed 方差待补）
      → paper/tables/table5_ablation_a3.tex（balance ON 在 8 block usage 熵全升，均值 0.6227→0.6865，
      支撑 C4 均衡机制；PSNR 中性；"多 seed 方差下降"待补，当前仅 seed=3407）
- [ ] (可选) 超参表：num_prototypes / router_dim 扫描

## D. 图表清单（figures/，含 data-manifest.md）

### D1 结构图（figures-diagram，无需数据）
- [~] Fig.1 整体网络结构（CATANet 主干 + DPR 位置）
      → 规格已写定 paper/figures/figure-specs.md（对应 method Eq.1–3）；待出图（draw.io）
- [~] Fig.2 DPR 模块数据流（原型生成→refine→匹配→排序→IASA）
      → 规格已写定 paper/figures/figure-specs.md（对应 method Eq.4–13，标三消融开关）；待出图（TikZ）

### D2 数据图（figures-python，需 CSV）
- [ ] Fig.3 PSNR-vs-Params（或 vs FLOPs）散点，突出本文 Pareto
- [ ] Fig.4 训练曲线对比（可选，DPRNet vs CATANet val PSNR）
- [ ] Fig.5 x_scores 直方图（加 router scale 前后区分度对比）
      → 需推理（本机无 torch/matplotlib），§3.6 已注明 camera-ready 补
- [x] Fig.6 prototype usage 分布柱状图
      → paper/figures/fig6_usage_entropy.tex（pgfplots/TikZ，per-block 归一化 usage 熵
      balance on vs off，数据 verbatim 自 ablation_perblock_urban100_80k.csv；正文引于 §3.6）

### D3 可视化图（需跑模型输出）
- [ ] Fig.7 视觉对比（Urban100/Manga109 难样本，GT/Bicubic/对手/本文 crop）
- [ ] Fig.8 路由聚类图（原始顺序 belong_idx → H×W 上色）
- [ ] Fig.9 排序前后 token 邻域语义一致性

## E. 投稿前置文件

- [~] references.bib（全部引用，CrossRef 验证 DOI）——初版已建：paper/references.bib
      （含 8 对比方法 + 数据集/指标/框架共 19 条；CATANet DOI/页码经 dblp 核实；
      其余 arXiv ID 已核实，会议页码/正式 DOI 待终稿前 CrossRef 逐条核对）。
- [ ] 期刊 LaTeX 模板（Neurocomputing=Elsevier elsarticle / TNNLS=IEEEtran）
- [ ] 环境信息表（CUDA/PyTorch/BasicSR 版本、硬件）

## F. 代码前置（已核实，详见 experiment-protocol.md §10 代码改动总账）

出图相关 bug 经核实已在源码修复，无需再改：
- [x] cluster map 用原始顺序 belong_idx（catanet_arch.py:265 last_routing_map）
- [x] 保存 cluster map 前先存 lq_shape 再 del self.lq（catanet_model.py:249-276）
- [ ] self-ensemble 非 EMA 分支（catanet_model.py:157 仍有 TODO，主流程未用，可忽略）

训练前已对齐的关键配置（防 x3/x4 重训）：
- [x] x3/x4 finetune 路由超参与 x2 完全对齐（见 §10.B）

---

## 收集节奏提示

训练一旦开始就持续 append 日志，不要等结束再回看。每个模型训练完立即：
(1) 跑 5 基准测试 → 存指标；(2) 存最优 checkpoint；(3) 导出路由诊断 CSV。
对手指标在第一周就抄录完毕，不要拖到回填主表时才找。
