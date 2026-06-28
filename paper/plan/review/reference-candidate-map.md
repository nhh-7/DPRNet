# DPRNet 参考文献扩充 evidence-citation map

日期：2026-06-28

## 范围

本轮目标是把 `references.bib` 从 20 条扩充到约 45 条，并且每条新增文献都进入正文引用位置。新增文献只用于加强已有论证，不改变实验结论，不引入未验证的性能数字。

## 已核验元数据来源

- arXiv 官方 API：用于 VDSR、DRCN、ESPCN、FSRCNN、MemNet、Non-local、IPT、CSNLN、HAT、EMT、HPINet、ShuffleMixer、Dual Regression、NTIRE 2024 等条目。
- CATANet 原文与 CrossRef：用于把 ESWT arXiv-only 替换为 Omni-SR 和 NGswin 两条 CVPR 2023 正式文献。
- CrossRef 查询：用于 VDSR、RDN、Survey 等条目的 DOI/venue 交叉核对；部分查询因 429 限流中断，未把未核验结果作为唯一证据。
- 既有 `references.bib`：CATANet、SwinIR、ELAN、SRFormer、SPIN、ATD、数据集和指标已在上一轮核对。

## Evidence-claim map

| Source ID | Citation | 可用事实 | 支撑论点 | Citation slot | 风险 |
|---|---|---|---|---|---|
| yang2010sparse | Yang et al., IEEE TIP 2010 | Sparse representation is a classical SISR formulation before deep SR. | SR existed as an ill-posed reconstruction problem before CNNs. | Introduction-P1 SR lineage | 低 |
| glasner2009single | Glasner et al., ICCV 2009 | Single-image self-similarity can support SR reconstruction. | Repetitive/self-similar structures are important for SR. | Introduction-P1, Discussion-P1 | 低 |
| kim2016vdsr | Kim et al., CVPR 2016 | Very deep CNN improves SR accuracy. | Deep CNNs pushed accuracy by increasing depth. | Introduction-P1 | 低 |
| kim2016drcn | Kim et al., CVPR 2016 | Recursive CNN increases depth with parameter sharing. | Deep/recursive designs improved SR but increased training/inference complexity. | Introduction-P1 | 低 |
| shi2016espcn | Shi et al., CVPR 2016 | Sub-pixel convolution enables efficient upsampling. | Efficiency-aware SR design appeared early in deep SR. | Introduction-P2 | 低 |
| dong2016fsrcnn | Dong et al., ECCV 2016 | FSRCNN accelerates SRCNN-style SR. | Lightweight SR reduces computation through compact architecture. | Introduction-P2 | 低 |
| lai2017lapsrn | Lai et al., CVPR 2017 | Laplacian pyramid progressively reconstructs HR images. | Multi-scale/progressive SR improves accuracy-cost trade-off. | Introduction-P1/P2 | 低 |
| haris2018dbpn | Haris et al., CVPR 2018 | Back-projection networks exploit iterative up/down sampling errors. | Later SR methods enlarged reconstruction capacity. | Introduction-P1 | 低 |
| tai2017drrn | Tai et al., CVPR 2017 | Deep recursive residual network uses recursive residual learning. | Recursive/residual SR improves accuracy but is not primarily lightweight deployment. | Introduction-P1/P2 | 低 |
| tai2017memnet | Tai et al., ICCV 2017 | Persistent memory network reuses long-term features. | Feature reuse is a major SR design principle. | Introduction-P2 | 低 |
| zhang2018rdn | Zhang et al., CVPR 2018 | Residual dense features improve SR representation. | Strong CNN SR baselines rely on rich feature reuse. | Introduction-P1 | 低 |
| zhang2018rcan | Zhang et al., ECCV 2018 | Channel attention improves deep residual SR. | Attention mechanisms entered high-accuracy SR before lightweight routing. | Introduction-P1/P3 | 低 |
| wang2018nonlocal | Wang et al., CVPR 2018 | Non-local operators model long-range dependencies. | Long-range dependency modeling motivates attention and routing. | Introduction-P3, Discussion-P1 | 低 |
| mei2020csnln | Mei et al., CVPR 2020 | Cross-scale non-local attention mines self-exemplars. | Self-similarity/non-local matching is useful in SR. | Introduction-P4, Discussion-P1 | 低 |
| yang2020ttn | Yang et al., CVPR 2020 | Texture Transformer transfers reference texture for SR. | Token/attention mechanisms can use semantic or texture-level correspondences. | Introduction-P4 | 低 |
| chen2021ipt | Chen et al., CVPR 2021 | Pre-trained image processing Transformer applies Transformer to restoration. | Transformer-based restoration predates lightweight SR variants. | Introduction-P3 | 低 |
| liu2021swin | Liu et al., ICCV 2021 | Shifted-window Transformer provides efficient local-window attention. | Window attention partitions context spatially. | Introduction-P3 | 低 |
| chen2023hat | Chen et al., CVPR 2023 | HAT activates more pixels in SR Transformer. | Recent SR Transformers seek larger effective receptive fields. | Introduction-P3 | 低 |
| zheng2023emt | Zheng et al., Engineering Applications of Artificial Intelligence 2024 | EMT mixes local pixel operations and efficient Transformer design. | Efficient Transformer SR reduces attention cost through hybrid design. | Introduction-P3 | 低 |
| wang2023omnisr | Wang et al., CVPR 2023 | Omni-SR models interactions across spatial and channel dimensions for lightweight SR. | Lightweight SR explores richer but efficient context aggregation. | Introduction-P3 | 低 |
| choi2023ngswin | Choi et al., CVPR 2023 | NGswin introduces n-gram context into Swin-based lightweight SR. | Window-based lightweight SR can be extended with local contextual patterns. | Introduction-P3 | 低 |
| liu2022hpinet | Liu et al., AAAI 2023 | HPINet combines coarse global access with fine intra-patch attention. | Lightweight SR seeks coarse-to-fine global/local modeling. | Introduction-P3 | 低 |
| sun2022shufflemixer | Sun et al., NeurIPS 2022 | ShuffleMixer is an efficient ConvNet for SR. | Efficient CNN designs remain competitive for deployment. | Introduction-P2 | 低 |
| guo2022dualregression | Guo et al., IEEE TPAMI 2024 | Dual regression targets lightweight SR/compression. | Lightweight SR also uses auxiliary constraints/compression. | Introduction-P2 | 低 |
| ren2024ntire | Ren et al., CVPRW 2024 | NTIRE 2024 efficient SR challenge evaluates runtime, FLOPs, and parameters. | Practical SR compares accuracy together with efficiency. | Introduction-P2, Efficiency | 低 |
| wang2021survey | Wang et al., Electronics 2021 | Survey organizes deep-learning SISR methods, datasets, losses, and metrics. | Broader SISR background and evaluation conventions. | Introduction-P1 | 低 |

## 插入原则

- 每条新增文献至少在正文出现一次。
- 只写作者、方法类别和设计事实，不借用未在本文表格中统一评估的性能数字。
- 引入 challenge report 只作为效率评价背景，不填补 Table II 的缺失时延/MACs。
- C4 改为稳定路由和缓解 prototype collapse，不再在贡献列表中宣称已验证 seed-to-seed variance reduction。
