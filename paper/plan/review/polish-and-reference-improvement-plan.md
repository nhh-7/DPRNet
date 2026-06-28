# DPRNet 论文润色与参考文献扩充计划

日期：2026-06-28

## 当前审查结论

- 论文结构已经基本完整：摘要、引言、方法、实验、讨论、结论、表格、图和 MDPI/CVPR 入口文件均已存在。
- `references.bib` 当前共有 20 条文献，正文实际唯一引用 key 也是 20 条。若按期刊稿约 45 篇参考文献的目标，当前明显不足。
- 现有参考文献覆盖了直接 baseline、基准数据集、指标、DIV2K 和 BasicSR，但对综述/背景、经典 SR 发展脉络、attention/non-local context、高效 CNN 变体、近期轻量 Transformer、confidence/routing 相关工作覆盖不足。
- 正文对 CATANet 的表述目前较稳妥：DPRNet 在重建质量上 matches or slightly surpasses CATANet，但参数量和 Multi-Adds 高于 CATANet。这个边界应保留。
- 消融部分已经诚实说明 shared-init、短预算、单 seed 的局限。除非补做新实验，否则不要删除这个限制。

## 优先级 1：把参考文献扩充到约 45 篇实引文献

不要添加未被正文引用的文献来凑数。每一条新增文献都应支撑正文中的一个具体论点。

建议目标分布：

- SR 基础与发展脉络：新增 5-7 篇。候选位置包括传统/经典 SR 背景，以及 VDSR、DRCN、DRRN、LapSRN、RCAN、RDN、ESPCN 等里程碑，具体以验证后的 BibTeX 为准。
- 轻量 CNN SR：新增 5-6 篇。候选位置是 Introduction 的 "Lightweight CNN-based SR" 段和主对比段。可覆盖高效蒸馏、重参数化、大核卷积、mobile SR 等方向，但必须先核验元数据。
- Transformer 与长程依赖 SR：新增 6-8 篇。候选位置是 "Lightweight Transformer-based SR" 段和 Discussion 的 content routing vs spatial locality 段。当前已有 SwinIR、ELAN、SRFormer、ATD，还需要覆盖 HAT、NLSA/non-local SR、高效 mixed/striped-window Transformer、近期 window-attention 变体。
- Content-aware、clustering、routing、dictionary/token 机制：新增 4-6 篇。候选位置是 "Content-aware / cluster-based routing" 段、Method motivation 和 routing analysis。这一类最能加强 DPR 的创新定位。
- Efficient SR challenge 与实际部署：新增 2-3 篇。候选位置是 Introduction 和 efficiency 小节。NTIRE efficient SR challenge report 可作为效率背景，但需验证且只在相关处引用。
- 指标、数据集和工具：当前覆盖基本够用。只有当正文新增指标或数据集时再补。
- 综述文献：新增 2-3 篇高质量、可追溯综述，用于支撑开篇背景，不替代核心方法论文。

建议 citation-slot 映射：

| 正文位置 | 缺失支撑 | 动作 |
|---|---|---|
| `sections/1_introduction.tex` 第一段 | SRCNN/EDSR 之外的 SISR 发展脉络不足 | 补 2-3 篇已验证里程碑论文。 |
| `sections/1_introduction.tex` CNN 轻量段 | 高效 CNN 家族覆盖偏少 | 补 4-5 篇已验证 efficient SR，并把段落改成综合论述而不是方法清单。 |
| `sections/1_introduction.tex` Transformer 段 | Transformer 分支覆盖不足 | 验证后补 HAT、NLSA/non-local、efficient Transformer 等引用。 |
| `sections/1_introduction.tex` content routing 段 | DPR 的邻近工作铺垫不足 | 补 content-aware、dictionary、token grouping、superpixel、clustering 或 adaptive token 相关论文。 |
| `sections/2_method.tex` motivation | EMA、hard label、confidence 的批评主要来自内部分析 | 精确引用 CATANet；如引入 confidence/uncertainty/routing 论述，再补相关文献。 |
| `sections/3_experiments_efficiency.tex` | 统一 latency/MAC 背景不足 | 可引用 challenge report 或统一效率协议论文，但未重测的表格项继续留空。 |
| `sections/4_discussion.tex` | 适用边界论述合理但引用偏少 | 验证后补 repetitive structure、non-local self-similarity、long-range attention 相关文献。 |

已执行的检索探测：

- `lightweight image super-resolution transformer efficient`
- `single image super resolution survey lightweight deep learning`
- `image super-resolution attention transformer window lightweight`
- `efficient single image super-resolution neural network`

第一轮检索得到的可进一步核验候选题名：

- "Efficient Mixed Transformer for Single Image Super-Resolution" (arXiv:2305.11403)
- "From Coarse to Fine: Hierarchical Pixel Integration for Lightweight Image Super-Resolution" (arXiv:2211.16776)
- "Image Super-Resolution using Efficient Striped Window Transformer" (arXiv:2301.09869)
- "ShuffleMixer: An Efficient ConvNet for Image Super-Resolution" (arXiv:2205.15175)
- "Towards Lightweight Super-Resolution with Dual Regression Learning" (arXiv:2207.07929)
- "The Ninth NTIRE 2024 Efficient Super-Resolution Challenge Report" (arXiv:2404.10343)
- "A comprehensive review of deep learning-based single image super-resolution" (Electronics, DOI: 10.3390/electronics10070867)

这些只是候选，不是最终参考文献条目。写入前必须从 arXiv/CrossRef/DBLP 或出版社页面获取 BibTeX，并核验作者、年份、标题和 venue。

## 优先级 2：文字润色

- Introduction：压缩方法清单式写法。当前段落清楚，但有 catalog 感。建议每个 related-work 段落改成 "设计原则 -> 局限 -> 为什么需要 DPR"。
- Contributions：C4 目前在 Introduction 中包含 "reduce seed-to-seed variance"，但消融和 Discussion 正确地把 variance reduction 留作 future work。建议改成 "stabilize usage and mitigate prototype collapse"，除非补做多 seed 实验，否则不要宣称 seed 方差降低。
- Abstract：保留对 CATANet 的保守表述，但可压缩 DPR 机制那一句。当前摘要技术细节准确，但信息密度偏高。
- Method：技术覆盖充分。轻度润色重点是减少 "addressing limitation" 和 "setting switch=false" 的重复，同时保留可复现性。
- Experiments：保留 protocol caveat。这会提高可信度。
- Discussion：保留显式 limitations。可补一段说明：小 PSNR margin 为什么仍有意义，但必须建立在 routing interpretability 与跨 benchmark 一致性之上。

## 优先级 3：图表

- 图注应说明每张图支撑什么结论，而不仅是展示什么内容。路由置信度直方图和聚类图尤其需要这一点。
- 视觉对比图注建议补一句：crop 是定性展示，不用于最大化 DPRNet 的单 crop 优势。
- Table I 继续作为定量主锚点。除非指标来自同一协议或明确标注来源，否则不要随意加新 baseline。
- Table II 中未统一重测的 competitor latency/MACs 继续用 `-`。不要混用不同硬件下的时延。
- 若版面允许，可在补充材料或 appendix 中加一张 "claim-to-evidence" 小表：贡献、支撑方程、消融/表/图、局限。

## 优先级 4：投稿前收尾

- 投稿前补齐 `main_mdpi.tex` 的作者、单位、ORCID、funding、data availability 和 conflict-of-interest 字段。
- 明确最终目标期刊：当前已迁移到 Applied Sciences，但 `project-overview.md` 仍写 Neurocomputing / Pattern Recognition / TNNLS。需要统一计划和稿件中的目标 venue。
- 新增参考文献后，重新完整编译，并检查 `.blg` 中是否有 unused/malformed entries。
- `references.bib` 的注释中不要写裸 `@...` 字符串，因为 BibTeX 会把它当作条目开头。

## 建议执行顺序

1. 新建 `paper/plan/review/reference-candidate-map.md`，整理 30-35 条候选文献，并为每条绑定 citation slot。
2. 通过 DBLP、CrossRef、arXiv 或出版社页面核验元数据。
3. 只把已验证且正文需要的文献加入 `references.bib`，目标约 45 篇实引文献。
4. 围绕扩充后的 evidence map 重写 Introduction 和 Discussion。
5. 润色 Abstract、Method、图注和表注。
6. 对 `main.tex` 和 `main_mdpi.tex` 分别跑完整 LaTeX 验证。
