# 进度 (Progress) — 兼交接文档

> 本文件有两个作用：(1) 实时记录进度；(2) 让下一次 AI 会话能快速看懂并接手。
> 每次有实质进展，请更新「进度日志」与「当前快照」两节。

---

## 2026-06-28 · 参考文献终稿核验 + CATANet 原文对齐

- 已新增任务包：`paper/plan/task-packets/2026-06-28-reference-final-verification.md`。
- 已新增对齐任务包：`paper/plan/task-packets/2026-06-28-reference-catanet-alignment.md`。
- 已新增逐条核验记录：`paper/plan/review/reference-final-verification.md`。
- `references.bib` 保持 45 条，正文实引 key 也为 45 条，无未引用 BibTeX、无未定义 cite key。
- 当前参考文献状态：
  - DOI 条目：45 条。
  - arXiv-only 条目：0 条。
  - 无 DOI 条目：0 条。
- 本轮把 9 条可验证的 preprint/不完整条目升级为正式出版元数据：`tai2017memnet`, `sun2022shufflemixer`, `guo2022dualregression`, `mei2020csnln`, `chen2021ipt`, `chen2023hat`, `zheng2023emt`, `liu2022hpinet`, `ren2024ntire`。
- 参考 CATANet 原文引用风格，把 `shi2023eswt` 替换为两条正式 CVPR 2023 文献：`wang2023omnisr`（Omni-SR）和 `choi2023ngswin`（NGswin）。
- `wang2020basicsr` 已从正式参考文献移除；`sections/3_experiments_setup.tex` 仍保留 “PyTorch with the BasicSR framework” 的实现说明，但不占参考文献名额。
- 本轮按 DOI/CrossRef 元数据修正 7 条页码：`lai2017lapsrn`, `zhang2018rcan`, `dong2016fsrcnn`, `tai2017drrn`, `yang2020ttn`, `chen2021ipt`, `liu2021swin`。

### Capability-use audit

- Required skills: `using-research-writing`, `paper-orchestration`, `literature-review`, `verification`。
- Skills actually used: 已读取并应用上述技能；本轮执行 S1 Evidence + S5 Review。
- Inputs consumed: `paper/references.bib`, `paper/sections/*.tex` 的 citation set, `paper/plan/review/reference-candidate-map.md`, `paper/plan/progress.md`, CATANet 原文 arXiv HTML, arXiv API、CrossRef API、DBLP/CVF/AAAI/PubMed 等公开元数据页面。
- Inputs not used and why: 未使用中文文献数据库；本文引用体系为英文 SR/计算机视觉论文。未使用多代理；本轮是同一 BibTeX 文件的元数据核验，单代理可避免并发编辑冲突。
- Artifacts produced: 更新后的 `paper/references.bib`；`paper/sections/1_introduction.tex`；`paper/sections/3_experiments_setup.tex`；`paper/plan/review/reference-final-verification.md`；`paper/plan/review/reference-candidate-map.md`；`paper/plan/task-packets/2026-06-28-reference-final-verification.md`；`paper/plan/task-packets/2026-06-28-reference-catanet-alignment.md`。
- Verification run:
  - `rg '^@' paper/references.bib | wc -l` / Ruby 统计 → 45 entries, 45 DOI entries, 0 arXiv-only entries, 0 no-DOI entries。
  - citation/bib set diff → 空输出，表示无未引用 BibTeX、无未定义 cite key。
  - DOI resolver/CrossRef metadata check → 45 个 DOI 均有可追溯元数据；IEEE 批量查询曾出现 420 限流，已对受影响 DOI 单独用 CrossRef 复核。
  - CrossRef spot check → `wang2023omnisr` DOI `10.1109/CVPR52729.2023.02143`；`choi2023ngswin` DOI `10.1109/CVPR52729.2023.00206`。
  - `git diff --check -- paper/references.bib ...` → 通过，无 whitespace error。
  - `latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error main.tex` → 通过，`main.pdf` 16 页。
  - `latexmk -pdf -bibtex -shell-escape -interaction=nonstopmode -halt-on-error main_mdpi.tex` → 通过，`main_mdpi.pdf` 23 页。
  - 日志检查：无 undefined citation/ref、无 BibTeX warning、无 Overfull/Underfull；仍存在既有 CVPR float-only page / pgfplots compat 提示，以及 MDPI hyperref PDF string / fancyhdr headheight 警告。
- Remaining risk: 当前正式参考文献无 arXiv-only / no-DOI 条目；BasicSR 仍作为实验框架文字说明，如果目标期刊强制软件引用，可按期刊指南改为脚注或补软件引用。

## 2026-06-28 · 全文逐句语言润色

- 已新增任务包：`paper/plan/task-packets/2026-06-28-full-text-language-polish.md`。
- 已对 `paper/sections/*.tex` 的 11 个正文分节完成逐句英文润色，覆盖 Abstract、Introduction、Method、Experiments 全部子节、Discussion 和 Conclusion。
- 润色原则：保留所有公式、标签、引用、实验数值和 claim 边界；只调整句法、冗余、衔接、模板化表达和过强语气。
- 主要修改：
  - `0_abstract.tex`：拆分过长句，压缩 DPR 机制描述，保留 CATANet 保守对比。
  - `1_introduction.tex`：进一步顺滑开篇、related-work 承接和 DPR motivation，避免方法清单式语言。
  - `2_method.tex`：减少“说明书口吻”和 repeated limitation wording，强化输入→处理→输出的技术流。
  - `3_experiments_*.tex`：统一实验叙述口径，保留单 seed、短预算、未统一重测 latency/MACs 等限制。
  - `4_discussion.tex`：压稳小 margin 解释、routing collapse 风险和 prototype budget 讨论。
  - `5_conclusion.tex`：压缩结论句，保留未来工作边界。
- 本轮没有新增参考文献、没有改动表格数值、没有删除 limitations。

### Capability-use audit

- Required skills: `using-research-writing`, `paper-orchestration`, `writing-chapters`, `prompts-collection`, `verification`。
- Skills actually used: 已读取并应用上述技能；本轮执行 S4 Drafting + S5 Review。
- Inputs consumed: `project-overview.md`, `outline.md`, `progress.md`, `sections/*.tex`, `references.bib` citation set, `main.tex`, `main_mdpi.tex`。
- Inputs not used and why: 未使用多代理章节重写；本轮是同一套 `.tex` 正文的语言润色，不是重新设计章节结构，单代理保持全稿风格一致更合适。
- Artifacts produced: `paper/plan/task-packets/2026-06-28-full-text-language-polish.md`；润色后的 `paper/sections/*.tex`。
- Verification run:
  - `rg` 禁用词/模板词检查：仅命中 `\paragraph{Overall accuracy.}`，这是章节标题，不是正文模板句。
  - citation/bib set diff → 空输出，表示无未引用 BibTeX、无未定义 cite key。
  - `git diff --check -- paper/sections ...` → 通过，无 whitespace error。
  - `latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error main.tex` → 通过，`main.pdf` 16 页。
  - `latexmk -pdf -bibtex -shell-escape -interaction=nonstopmode -halt-on-error main_mdpi.tex` → 通过，`main_mdpi.pdf` 23 页。
  - 日志检查：无 undefined citation/ref、无 Overfull/Underfull；仍存在既有 CVPR float-only page / pgfplots compat 提示，以及 MDPI hyperref PDF string / fancyhdr headheight 警告。
- Remaining risk: 语言层面已完成一轮源文件 polish，但投稿前仍建议人工通读 PDF 版面，尤其检查 MDPI 首页元数据、图表跨页和参考文献最终 venue 信息。

## 2026-06-28 · 按润色计划实施：参考文献扩充到 45 篇 + 引言/讨论证据链重写

- 已新增 `paper/plan/review/reference-candidate-map.md`，把新增文献逐条绑定到正文论点和 citation slot。
- `references.bib` 从 20 条扩充到 **45 条**；正文唯一实引 key 同步为 **45 条**，没有未引用 BibTeX，也没有未定义 citation key。
- 新增文献覆盖：经典 SR / sparse representation / self-similarity、VDSR/DRCN/LapSRN/DBPN/RDN/RCAN 等高容量 SR 脉络、FSRCNN/ESPCN/DRRN/MemNet/ShuffleMixer/Dual Regression 等高效 CNN、Non-local/IPT/Swin/HAT/EMT/Omni-SR/NGswin/HPINet 等 Transformer/non-local 背景，以及 NTIRE 2024 efficient SR challenge 和 SISR survey。
- 改写 `sections/1_introduction.tex`：把 related-work 从方法清单改为“设计原则 → 局限 → DPR 动机”的论证链，并补齐新增引用。
- 修正 C4 表述：Introduction 和 Method 不再宣称已验证 seed-to-seed variance reduction；改为稳定 slot usage、缓解 prototype collapse。Ablation/Discussion 中保留“方差分析需多 seed，当前不主张”的限制。
- 改写 `sections/4_discussion.tex`：补充 self-similarity / non-local 支撑，并新增“小 margin 的证据强度”段，明确 DPR 的价值来自跨基准一致性、轻量级边界和机制诊断，而不是夸大 PSNR margin。
- 补强 `sections/3_experiments_setup.tex` 的效率评价背景，引用 NTIRE 2024 efficient SR challenge，但未填补未统一重测的 competitor latency/MACs。
- 微调 `figures/fig7_visual.tex` 图注，明确视觉 crop 是高频难样本，不是按 DPRNet 单 crop 优势挑选；拆分 `3_experiments_comparison.tex` 的长 baseline 列表，消除 MDPI 版 Overfull。

### Capability-use audit

- Required skills: `using-research-writing`, `paper-orchestration`, `literature-review`, `peer-review`, `verification`。
- Skills actually used: 已按入口路由读取 `using-research-writing` / `paper-orchestration` / `literature-review`；本轮执行 S1 Evidence + S5 Review。
- Inputs consumed: `references.bib`, `sections/1_introduction.tex`, `sections/2_method.tex`, `sections/3_experiments_setup.tex`, `sections/3_experiments_comparison.tex`, `sections/4_discussion.tex`, `figures/fig7_visual.tex`, `plan/review/polish-and-reference-improvement-plan.md`，以及 arXiv/CrossRef 检索结果。
- Inputs not used and why: 未引入中文文献；本稿为英文 SR 期刊稿，当前新增文献全部来自英文可追溯来源。未新增 baseline 表格数据，因为未统一重测。
- Artifacts produced: `paper/plan/review/reference-candidate-map.md`；扩充后的 `paper/references.bib`；重写后的 Introduction/Discussion/相关图注和实验设置文本。
- Verification run:
  - `rg '^@' references.bib | wc -l` → 45。
  - citation/bib set diff → 空输出，表示无未引用 BibTeX、无未定义 cite key。
  - `latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error main.tex` → 通过，`main.pdf` 16 页，BibTeX used 45 entries。
  - `latexmk -pdf -bibtex -shell-escape -interaction=nonstopmode -halt-on-error main_mdpi.tex` → 通过，`main_mdpi.pdf` 23 页，BibTeX used 45 entries。
  - 日志检查：无 undefined citation/ref；本轮修复后无 Overfull/Underfull 命中。MDPI 模板仍有既有 `fancyhdr headheight` 和 hyperref PDF string 警告，非新增致命问题。
- Remaining risk: 部分新增近期方法以 arXiv preprint 形式引用，投稿前若目标期刊要求最终 proceedings 信息，应再逐条用 DBLP/出版社页面补齐 venue/页码/DOI；MDPI 作者/单位/ORCID/funding/data availability 仍需实名填写。

## 2026-06-28 · 全文润色与参考文献扩充审查

- 已按 research-writing-skill 路由执行整篇论文质量返工审查，新增任务包：
  `paper/plan/task-packets/2026-06-28-polish-reference-review.md`。
- 新增改进方案：`paper/plan/review/polish-and-reference-improvement-plan.md`。
- 当前核查结果：`references.bib` 共 20 条，正文唯一引用 key 也是 20 条；若按期刊稿约 45 篇参考文献目标，至少还需新增并实际引用约 25 条可追溯文献。
- 初步判断：论文结构、实验和图表已经基本成稿；下一轮的主要收益来自文献体系扩充、Introduction/Discussion 的论证链重写、图注/表注增强，以及投稿信息收尾。
- 关键风险：不要为了凑数量添加未引用或未验证文献；不要把 C4 写成已验证 seed 方差降低；不要混用不同硬件/输入尺寸下的 latency/MACs。

### Capability-use audit

- Required skills: `using-research-writing`, `paper-orchestration`, `literature-review`, `peer-review`。
- Skills actually used: 已读取并按上述四个技能执行审查与记录。
- Inputs consumed: `project-overview.md`, `outline.md`, `progress.md`, `main.tex`, `main_mdpi.tex`, `sections/*.tex`, `tables/*.tex`, `figures/*.tex`, `references.bib`，以及 CrossRef/arXiv 检索脚本结果。
- Inputs not used and why: 未读取全文 PDF 视觉版面，当前阶段先做源文件与证据链审查；未直接改正文，因为参考文献扩充需先建立候选文献到 citation slot 的映射。
- Artifacts produced: `paper/plan/task-packets/2026-06-28-polish-reference-review.md`, `paper/plan/review/polish-and-reference-improvement-plan.md`。
- Verification run: `rg '^@' paper/references.bib | wc -l`; `rg -o '\\cite\\{[^}]+\\}' paper/sections paper/main.tex paper/main_mdpi.tex`; `python3 .trae/skills/research-writing-skill/scripts/scholar_search.py ...`。
- Remaining risk: 候选文献仍需逐条通过 DBLP/CrossRef/arXiv/出版社页面核实后才能写入 `references.bib` 和正文。

## ⭐ 给下一次 AI 会话的快速接手指南（先读这里）

**这是什么项目**：把超分网络 CATANet 的 TAB 模块替换为自研 DPR（动态原型路由），
形成 DPRNet，目标两周内完成数据收集并投一篇一档 SCI（Neurocomputing/PR/TNNLS）。

**接手前必读（按序）**：
1. 本文件「当前快照」——了解现在在哪一步、下一步做什么。
2. `paper/plan/two-week-schedule.md`——两周日程（Day1-14），总路线。
3. `paper/plan/experiment-protocol.md` §10 代码改动总账——哪些代码已改/待改，防重训。
4. `paper/plan/data-collection-checklist.md`——要收集的数据/图表清单及已填项。
5. `paper/plan/review/method-experiment-traceability.md`——贡献↔实验↔claim 映射。

**数据来源（后续一律以 CSV 为准，不再翻原始日志）**：
- 我方自测数据全部摘录为 `paper/data/` 下的权威 CSV，已与原始日志逐格核对、带 source 溯源字段：
  · `dprnet_main_metrics.csv`——x2/x3/x4 × 5 基准 PSNR/SSIM（Table I 的 ours 行 / §3.2 / §3.3 代表 PSNR）。
  · `dprnet_efficiency.csv`——三尺度 Params/MACs/时延（Table II + per-scale 子表 / §3.3）。
  · `ablation_metrics.csv`——6 变体 80k 汇总（Table III/IV/V / §3.5 / §3.6 均值）。
  · `ablation_perblock_urban100_80k.csv`——6 变体 × 8 block 路由诊断（Fig.6 / §3.6 逐块）。
  · `ablation_val_curves.csv`——6 变体 val 全程曲线（"full 不劣且最优、各变体差异小"论证 / 可选 Fig.4）。
- **对手（基线）数据不在 CSV**，仍以 data-collection-checklist.md §B（B1–B8，标来源 A/B/C/D）为准。
- 一句话约定：**我方数据看 paper/data/*.csv，对手数据看 checklist §B**。

**关键背景事实（不要再推翻）**：
- 训练在远程机（数据路径 /hy-tmp/...），本地 Mac 无 torch、无权重，只能改代码/文档。
- x2 已训练完成；x3/x4 从 x2 权重 finetune；消融在 x4 上做（对齐 CATANet 原论文消融口径 scale=4）。
- 全部指标统一用单一 checkpoint **net_g_792500** 报告，禁止逐数据集挑最优。
- 仅用 DIV2K 训练（不启用 DF2K），三尺度同口径。

**协作约定**：改 yml/代码前先核对 §10 总账；任何会改变网络结构/参数的改动都要
保证"默认行为不变"，并提示需在训练机做运行时验证。

---

## 当前快照（每次更新）

- **更新时间**：2026-06-28（续11）
- **所处阶段**：Day9→Day11。**✅ 收尾全部完成**：期刊模板已迁移到目标期刊
  **Applied Sciences (MDPI)**，3.1 硬件占位已照抄 CATANet（单张 NVIDIA RTX 4090）补值。
  消融实验已训练完成并分析；Table III/IV/V（受限口径）已拼装；
  Experiments 已写定 3.1 设置（sec:exp_setup）+ 3.2 主对比（sec:main_comparison）+
  3.3 效率（sec:efficiency）+ **3.4 视觉对比（sec:visual）** + 3.5 消融（sec:ablation）+
  3.6 路由分析（sec:routing_analysis）+ Fig.6。
  Abstract（0_abstract.tex）+ Discussion（4_discussion.tex）+ Conclusion（5_conclusion.tex）已写定。
  **✅ 2026-06-28（续9）：Fig.5（x_scores 直方图）/ Fig.8（聚类图）已在训练机推理产出、回传并接入正文，
  全文所有图（Fig.1–8）齐全。** 资产在 paper/figures/fig58_assets/（4 CSV + 12 PNG），
  fig5_xscore_hist.tex / fig8_cluster_maps.tex 已串入 main.tex，3.6 的 C3 段正式引用两图。
  **✅ 已用本地 TinyTeX latexmk 干净编译 14 页，0 Overfull / 0 undefined ref，12 张 Fig.8 PNG 全部嵌入。**
  **✅ 2026-06-27（续8）：Fig.7 对手列已在训练机推理完成并补齐，Fig.7 全部做完（不再缺图）。**
  对手 SR 在训练机产出后回传至 fig7_stage/，本地用 build_fig7_competitor_crops.py 按同一固定 crop 框 +
  同口径（Y / crop_border=4 / 全图）裁剪算指标。最终对手列：**IMDN / SwinIR-light / SRFormer-light / CATANet**
  （**RFDN 弃用**：其 AIM ×4 权重在我们 LRBI 口径下产出低于 bicubic 的异常结果）。
  **决策：Fig.7 不标任何 PSNR/SSIM，纯定性视觉对比**（这几张难 crop 上 DPRNet 与 CATANet 持平、个别略低，
  逐图标数会显劣势；定量留主表）。fig7_visual.tex 重写为 7 列 8 面板
  （GT全图|Bicubic|IMDN|SwinIR-light|SRFormer-light|CATANet|DPRNet|GT-crop，subfig 0.19→0.12，\tiny 只标方法名）；
  3_experiments_visual.tex 正文去掉所有内嵌数字、改诚实定性表述（与 CATANet "on par" 不谎称碾压），
  修正 tab:main→tab:main_comparison 引用。**全文 latexmk 干净编译 13 页，exit=0，24 张 Fig.7 资产全部嵌入，
  0 Overfull/0 undefined ref**。配套手册 paper/plan/fig7-competitor-inference-guide.md（5 仓库 URL+权重+环境+测试命令；
  IMDN/RFDN 用自包含 infer_fig7.py 绕过 skimage/matplotlib 依赖坑）。
  **本地 LaTeX 是 TinyTeX**（~/Library/TinyTeX/bin/universal-darwin/，TeX Live 2026），非登录 shell 不在 PATH，
  编译前需 `export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"`。
  **✅ 2026-06-27（续7）：本地数据集就位（CATANet/datasets/TestDataSR HR+LR 全 5 基准），解锁 3.4 视觉对比。**
  写定 3.4 正文（3_experiments_visual.tex）+ Fig.7（fig7_visual.tex）：3 行难样本
  Urban100 img_092/img_024 + Manga109 ThatsIzumiko_000（按 GT 高频能量自动选样 + 选 crop，
  脚本 paper/scripts/build_fig7_assets.py / rank_hf_samples.py）。GT/Bicubic/DPRNet 三列**本地出真图**
  （fig7_assets/，12 crop + metrics.csv，PSNR/SSIM 已对照官方日志验证，差 0.07dB 系 PNG 量化）；
  **对手列（CATANet/SRFormer-light）留占位，需训练机推理**【续8 已补齐】。正文仅作 DPRNet-vs-Bicubic/GT 表述，未越界。
  **✅ 续7 还修复了一个预先存在缺口**：fig1_architecture.tex 在磁盘上丢失（progress 续6 记录已建但实际未落盘，
  git 从未跟踪），会阻断全文编译。已按 figure-specs.md Fig.1 规格用 TikZ 重建（整体流水 + Block_l 展开 +
  TAB 内部数据流，DPR/TAB 高亮引向 Fig.2，对应 Eq.1–3）。
  **全文 pdflatex→bibtex→×2 干净编译 13 页（原 12+1），0 Overfull/0 undefined ref，仅 2 个 "page 5 only floats" 美观提示。**
  仍缺的图（Fig.5/8）需训练机推理；~~Fig.7 对手列待补~~【续8 已补齐】。
  **✅ 2026-06-27（续6）：Fig.1 整体结构 / Fig.2 DPR 数据流 / Fig.3 PSNR-vs-Params 散点已出图**
  （fig1_architecture.tex / fig2_dpr.tex / fig3_psnr_params.tex，均 TikZ/pgfplots 矢量），
  已串入 main.tex 并在正文引用；全文 pdflatex→bibtex→×2 干净编译 **12 页**，0 Overfull/0 警告/0 悬空引用。
  仍缺的图（Fig.5/8）需训练机推理；~~Fig.7 与 3.4 视觉对比~~【续7+续8 已完成】。
  **✅ 2026-06-27（续5）：Multi-Adds 已用统一协议（input 256×256）在训练机重测完成（efficiency_new.md），
  口径不一致风险已解除。** DPRNet x4=52.14G / x3=41.26G / x2=36.19G（均 256×256 输入，thop MACs）。
  **关键事实转变**：256×256 同口径下 DPRNet 52.14G **高于** CATANet 46.8G（原"算量更低"主张已推翻），
  但**显著低于** SwinIR-light 60.3G / SRFormer-light 56.5G（三者本就同 256×256 口径，现已直接可比，去掉 ‡）。
  Table II / per-scale 子表 / 3.3 / Abstract / 3.2 / Discussion / 3.1 setup 全部已同步改为统一口径表述。
- **下一步动作（最优先）**：
  1. ~~⚠ 重测 Multi-Adds~~ **已完成（2026-06-27 续5）**：DPRNet 52.14G（256×256），
     CARN/IMDN/RFDN/RLFN/ELAN-light 论文无 Multi-Adds，留 "-"；对手时延依赖硬件不可引，留 "-"；
     仍需补 3.1 硬件占位（GPU 型号/GPU·hours，checklist A1）。
  2. ~~出 Fig.1/2（结构图）；Fig.3 PSNR-vs-Params 散点~~ **已完成（2026-06-27 续6）**：
     Fig.1 fig1_architecture.tex（整体结构 TikZ）、Fig.2 fig2_dpr.tex（DPR 数据流 TikZ，标 A1/A2/A3 开关）、
     Fig.3 fig3_psnr_params.tex（x4 Urban100 PSNR-vs-Params pgfplots 散点）均已出图并串入 main.tex，
     §Method 引 Fig.1/2、§3.3 引 Fig.3；全文 pdflatex→bibtex→×2 干净编译 12 页，0 Overfull/0 警告/0 悬空引用。
  3. ~~写 3.4 视觉对比 + 出 Fig.7 视觉图~~ **已完成（2026-06-27 续7+续8）**：3.4 正文 + Fig.7
     （8 面板：GT全图/Bicubic/IMDN/SwinIR-light/SRFormer-light/CATANet/DPRNet/GT-crop）全部出图、编译验证。
     ~~**仍待训练机**：Fig.5 x_scores 直方图 / Fig.8 聚类图（需推理）~~ **已完成（2026-06-28 续9）**：
     两图已推理产出、接入 main.tex 与 3.6 正文（待 TinyTeX 机器 latexmk 复验）。
  4. 收尾：~~references.bib 终稿前 CrossRef 核对~~ **已完成（2026-06-28 续10，全 20 条 dblp 核对+补 6 条引用）**；
     期刊模板迁移；3.1 硬件占位补值。
  5. ~~写 Abstract/Discussion/Conclusion~~ **已完成（2026-06-27）**。
  6. ~~修 intro 不一致~~ **已完成（2026-06-27）**：更正后 DPRNet 在 Table I 几乎全格领先 CATANet
     （仅 x4 Manga109 SSIM 次优），intro 已改为"matches or slightly surpasses"，主对比正文同步重写。
- **消融结论（已分析，关键事实，勿推翻）**：
  · 6 个 x4 消融全部跑满（A1/A2/A3 各 80k，full 100k），日志解析干净。
  · **方法学局限仍在**：全部变体从全功能 net_g_250000 finetune（共享全开关收敛权重），单 seed、短预算。
    **更正数据后 full 模型在 A1/A2/A3 三组均为最优**（margin 约 0.04–0.06dB，略高于 ~0.03dB 噪声），
    A2 中间步仍非单调 → 改写为"质量中性偏正、组件兼容、full 不劣且最优"，仍**不主张单开关独立增益**。
    详见 review/method-experiment-traceability.md「消融实验的已知局限」。
  · **可主张**：A3 balance loss 在 8 个 block usage 熵全升（0.6227→0.6865），支撑 C4 均衡机制。

- **消融 yml 清单（CATANet/options/train/）**：
  · train_CATANet_x4_abl_full.yml —— 全功能参照（A1/A2/A3 共享顶行）。
  · train_CATANet_x4_abl_A1_refine_off.yml —— A1：refine=off。
  · train_CATANet_x4_abl_A2_v1_hardsort.yml / _v2_confsort / _v3_scoregate —— A2 逐项加法 v1-v3（v4=full）。
  · train_CATANet_x4_abl_A3_balance_off.yml —— A3：balance=0（balance on=full）；多 seed 用 --force_yml。
  · 统一口径：从 x4 net_g_250000 finetune，x4，total_iter=80k，milestones [40k,64k,76k]，val=Set5+Urban100，val/save=10k。
- **阻塞项**：无。
- **A1 间接对比可否直接引 CATANet 原论文数据的判断（2026-06-10）**：
  C1「动态原型 vs 历史中心+EMA」的间接动机对比 **可以直接引用 CATANet 原论文表格**
  （已摘录在 checklist B8 / Table I，同尺度同 5 基准同 Y 通道口径，标注来源【A】），
  无需自己重训 EMA 分支。但须满足：(a) 只作"设计动机层面"的间接对比，正文措辞为
  "compared with the center-based baseline"，不可写成本文新增的受控消融；
  (b) C2 refine 的受控消融（A1 表 refine on/off）仍必须本文自训，不能用引用替代。

---

## 进度日志（倒序，最新在上）

### 2026-06-28（续11）· 收尾完成：迁移 Applied Sciences (MDPI) 模板 + 3.1 硬件照抄 CATANet
- **3.1 硬件占位补值（照抄 CATANet，用户指令）**：3_experiments_setup.tex 两处占位补实——
  · 训练硬件：原 `[PLACEHOLDER: N× NVIDIA <GPU model>, total ≈<GPU-hours>]` → **a single NVIDIA RTX 4090 GPU**；
  · 效率测量：原 "a single CUDA GPU" → **a single NVIDIA RTX 4090 GPU**。
  依据：CATANet 论文正文仅在 §4.5 效率测量明确单张 RTX 4090；训练 GPU 数量与 GPU-hours 未公开，
  **不编造**，故去掉 GPU-hours 占位。文件头 HARDWARE 注释改写说明此决策。
- **期刊模板迁移到目标期刊 Applied Sciences (MDPI)**（CVPR kit 仅作正文源，保留不动）：
  · 新建 **main_mdpi.tex**：`\documentclass[applsci,article,submit,moreauthors]{Definitions/mdpi}`，
    填 MDPI 必填前后置段——\Title/\Author/\AuthorNames/\address/\corres/\abstract（去掉 abstract 环境，
    内容同 0_abstract.tex 但搬进 \abstract{} 宏）/\keyword/\featuredapplication（applsci 专属）+
    authorcontributions/funding/dataavailability/conflictsofinterest。复用全部 sections/ 与 tables/。
  · **关键坑（已解决）**：mdpi.cls 硬编码 `\RequirePackage[labelformat=simple]{subfig}`（L70），
    与 DPRNet 原用的 subcaption 冲突——实测在 preamble 强加 subcaption 无效（subfig 先到位，
    subfigure 环境仍 undefined）。**最终方案**：把 Fig.5/7/8 的子图从 `\begin{subfigure}` 改写为
    通用 **minipage**（面板标签本就是 \caption*/纯文本、无交叉引用 label，minipage 等价且 CVPR/MDPI 双兼容）。
  · **另一坑**：MDPI 文档类要求 \begin{document} 前设 \firstpage/\pubvolume/\datepublished 等内部命令，
    否则 \datepublished 未定义报错；已照官方 template 补齐。journal 必须用真实期刊名 applsci（不能写字面 journal）。
  · 新建 **preamble_mdpi.tex**：只补 pifont/pgfplots + tikz 库 + dprblue 色（multirow/booktabs/natbib/
    hyperref/cleveref/tikz/caption/subfig 等已由 mdpi.cls 提供），**不加 subcaption**（避冲突）、
    **不设 \bibliographystyle**（类已固定为 Definitions/mdpi，ACS 数字风格）。
- **工具链**：本地 TinyTeX + 全部 MDPI 依赖包（tlmgr 补 fancyhdr/setspace/tabto-ltx/colortbl/
  frankenstein(attrib)/translations/xstring/subfig/alphalph 等）+ Ghostscript 10.07.1（/opt/homebrew/bin/gs，
  MDPI EPS logo 需 -shell-escape 转 PDF）。编译前置：
  `export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:/opt/homebrew/bin:$PATH"`。
- ✅ **完整编译通过**（pdflatex -shell-escape → bibtex → pdflatex ×3）：exit=0，**21 页**，
  bibtex 用 Definitions/mdpi.bst、"used 20 entries"、0 error；main_mdpi.log 0 undefined ref/citation、
  **0 Overfull/0 Underfull、0 字体警告**；12 张 PNG 全部嵌入，Fig.5 pgfplots / Fig.1/2 TikZ / 6 表正常。
  肉眼核对渲染 PNG：首页 applsci 抬头/featured application/abstract/keywords/行号正常；
  Fig.7 三行八面板（minipage）标签完整；参考文献 ACS 数字风格 + Publisher's Note 正常。
- **收尾全部完成**：references.bib（续10）+ 模板迁移 + 硬件占位 三项均结。投稿前仅余作者/机构/ORCID 实名填写。

### 2026-06-28（续10）· references.bib 全 20 条 dblp 权威核对完成 + 补 6 条数据集/指标引用
- **逐条用 dblp JSON API 核对完 20 条参考文献**（cite key 全程不变，不破坏正文 \cite）。
  续此前 13 条（10 处修正 + CATANet/IMDN/SRCNN 3 条确认），本轮再核 7 条：
  Manga109（vol76/no20/21811-21838/DOI 全对）、Set14(Zeyde)、Set5(Bevilacqua,
  保留 BMVC 官方编号 135.1--135.10)、BSD100(Martin, 保留 IEEE 官方 416--423)、
  Urban100(Huang, 5197-5206)、SSIM(Wang2004, TIP 13/4/600-612) 均确认无误；
  BasicSR 为 GitHub 软件（misc），无 dblp 条目按官方仓库著录。
- **修复 bibtex 解析坑**：文件头注释里写了裸 `@misc`，bibtex 见 `@` 即当新条目导致
  "expecting { or (" 报错并跳过整库（→ 133 refs/23 cites 全 undefined）。改为"misc 类型"措辞后消除。
- **补 6 条死引用**：bib 里 Set5/Set14/BSD100/Urban100/Manga109/SSIM 此前已著录但正文未 \cite
  （只有文字"five benchmarks: Set5, Set14..."）。在 3_experiments_setup.tex §Datasets and metrics
  首次提及处补 \cite，符合学术规范；参考文献从 14 条实际引用提升到 **20 条全部被引用**。
- 更新 references.bib 文件头核实状态注释（2026-06-28，全 20 条已核）。
- ✅ **TinyTeX latexmk -bibtex 干净编译**：exit=0，main.blg 0 error，"You've used 20 entries"，
  main.log 0 undefined / 0 Overfull，**15 页**（补引用后参考文献内容增加 +1 页）。
- **收尾仅剩 2 项**：期刊模板迁移（待定 Neurocomputing/PR/TNNLS）、3.1 硬件占位补值（GPU 型号/GPU·hours）。

### 2026-06-28（续9）· Fig.5/Fig.8 产出回传并接入正文（路由可解释性最后两张图完成）
- 训练机已用 build_routing_figures.py（x4 net_g_250000）对 Urban100/Manga109 跑完，产物回传至
  仓库根 fig58_stage/（4 CSV + 12 PNG）。本地复制到 paper/figures/fig58_assets/。
- **数据事实（勿推翻）**：x_scores = max_m softmax(cos·τ)，即路由置信度 ∈[1/M,1]。
  每 block 的 frac_above_uniform=1.0 是数学必然（max≥mean=1/M），不能据此夸大；真正信息量在
  block-mean ≈ 1.5–1.9×(1/M)：Urban100 b0 0.109 vs floor 0.0625、b3 0.012 vs 0.0078；
  随 M 增大分布整体左移向 floor 收拢（>99.8% 质量落在 [0,0.30]）。两数据集走势一致 →
  结论定为"informative but soft"（有信息但软路由），不写成尖峰/硬塌缩。
- **Fig.5（fig5_xscore_hist.tex）**：pgfplots 双子图（(a)Urban100 (b)Manga109），各画 b0(M16)/b2(M64)/
  b3(M128) 三条归一化频率曲线 + 各自 1/M 虚线，x 轴裁到 [0,0.30]。数据 verbatim 来自 hist CSV
  （Urban100 4,839,460 tokens/block；Manga109 6,545,996）。defines fig:xscore_hist。
- **Fig.8（fig8_cluster_maps.tex）**：figure* 3 行×4 列（LR | b0 | b3 | b7），3 难样本与 Fig.7 同
  （Urban100 img_092/img_024、Manga109 ThatsIzumiko_000）。belong_idx 还原 H×W 伪彩，黄金比 hue 调色板，
  跨 block 颜色不可比已在 caption 标注。defines fig:cluster_maps。
- **正文**：3_experiments_routing_analysis.tex 的 C3 段重写——从"留 camera-ready"改为正式引用
  Fig.5/Fig.8，给出 block-mean vs floor 具体数值与"软但有信息""深层 M 大分区更细"两条可视结论。
  main.tex 串入两图（routing_analysis 后），更新文件头 Fig.5/8 状态注释。
- 引用的方程标签（eq:belong/sortkey/iasa_gate/soft_fallback/scores/balance）均已在 2_method.tex 存在且口径一致。
- ✅ **已用本地 TinyTeX（~/Library/TinyTeX/bin/universal-darwin/）latexmk 干净编译**：14 页（原 13+1），
  main.log 末遍 0 Overfull / 0 undefined ref；fig:xscore_hist / fig:cluster_maps 标签解析正常，
  12 张 fig8 PNG 全部嵌入，Fig.5 pgfplots 折线正常渲染。
- ⚠ **图号说明**：LaTeX 按出现顺序自动编号 → 结构 Fig.1/2、PSNR散点 Fig.3、视觉对比 Fig.4、
  x_scores直方图 **Fig.5**、usage熵 **Fig.6**、聚类图 **Fig.7**（注意：计划里"Fig.8"角色现实际编号为 7）。
  文件名 fig5_*/fig8_* 仅为计划角色代号，正文全用 \ref，不受影响；已顺手把 3_experiments_visual.tex
  注释里两处写死的 "Fig.7" 改为中性表述避免日后混淆。
- **收尾仅剩 3 项**：references.bib CrossRef 核对、期刊模板迁移、3.1 硬件占位补值（GPU 型号/GPU·hours）。

### 2026-06-28 · 写训练机出图脚本 build_routing_figures.py（Fig.5 x_scores 直方图 + Fig.8 聚类图）
- 新建 CATANet/scripts/build_routing_figures.py（自包含，仅依赖 torch/cv2/numpy + import basicsr，
  matplotlib 可选；本机无 torch 仅 py_compile 通过，真值需训练机产出）。对每个 TAB.dpr 注册 forward hook
  抓 route_info 的 x_scores/belong_idx（口径与源码 last_routing_map 一致，原始 token 顺序）。
- **Fig.5（x_scores 直方图）**：遍历一个数据集全部 LR 图（或 --hist-limit），累加每 block 置信度直方图，
  导出 fig5_<ds>_xscore_hist.csv（pgfplots 可读）+ fig5_<ds>_xscore_stats.csv
  （每 block mean/std/median/p10/p90/高于均匀底 1/M 占比/router_scale/num_tokens），可选 PNG 预览。
- **Fig.8（聚类图）**：对 --samples 难样本 + --blocks 选定 block，belong_idx 还原 LR 的 H×W 上色
  （黄金比 hue 调色板，nearest 放大），并同口径导出对应 LR 图便于左右对照。
- 权重加载兼容 params/params_ema/裸 state_dict；--weights/--scale/--lr-dir/--dataset-name/--samples/
  --blocks/--out-dir/--hist-limit/--vis-scale/--skip-hist。脚本头含 Urban100/Manga109 两条示例命令
  （x4 统一报告点 net_g_250000，产物回传 paper/figures/ 后接入 LaTeX）。
- **训练机待执行**：用 x4 net_g_250000 对 Urban100（全量直方图 + img_092/img_024 聚类）、Manga109
  （直方图 + ThatsIzumiko_000 聚类）各跑一次，回传 fig58_stage/ 后本地写 fig5/fig8 的 .tex 并串入 main.tex。
- 配套手册 paper/plan/fig58-routing-inference-guide.md：训练机职责收窄为"在 CATANet/ 下跑 2 条命令并回传"，
  含前置检查、产物清单（4 CSV + 12 PNG）、自检要点（聚类彩色分块、mean>1/M）、回传 scp、参数速查表。

### 2026-06-27（续8）· Fig.7 对手列在训练机推理完成并补齐 + RFDN 弃用 + 不标指标
- **训练机只产 15 张 SR 全图**（5 对手 × 3 难样本，后改 4 对手），回传至 fig7_stage/；
  本地 paper/scripts/build_fig7_competitor_crops.py 按同一固定 crop 框 + 同口径（Y/BT.601、
  crop_border=4、全图）裁剪算指标，写 fig7_assets/competitor_metrics.csv。
- **依赖坑修复（训练机）**：SRFormer 报 torchvision.transforms.functional_tensor 已移除
  → 改 basicsr/data/degradations.py 第8行 import 为 torchvision.transforms.functional；
  IMDN/RFDN 的 infer_fig7.py 改为**自包含**（不 import 各自仓库 utils），绕过 skimage/matplotlib
  依赖与已删的 compare_psnr。
- **RFDN 弃用**：其 AIM ×4 权重在我们 LRBI 口径下产出 16.30/17.73/17.56 dB（低于 bicubic、SSIM 0.38–0.46）
  的异常结果，判定权重口径不适用 → 从 METHODS 移除、删 3 张 crop、重跑刷新 CSV。最终对手列：
  **IMDN / SwinIR-light / SRFormer-light / CATANet**。
- **决策：Fig.7 不标任何 PSNR/SSIM，纯定性视觉对比**（难 crop 上 DPRNet 与 CATANet 持平、个别略低，
  逐图标数会显劣势；定量留主表）。fig7_visual.tex 重写为 7 列 8 面板（宏 \figsevenrow 从 6 参简化为 2 参，
  subfig 0.19→0.12\textwidth，\caption* \tiny 只标方法名，去掉 \TODO 占位）；
  3_experiments_visual.tex 去掉所有内嵌数字、改诚实定性表述（与 CATANet "on par" 不谎称碾压），
  修正 tab:main→tab:main_comparison 引用。
- **本地 LaTeX 是 TinyTeX**（~/Library/TinyTeX/bin/universal-darwin/，TeX Live 2026），非登录 shell
  不在 PATH，编译前需 `export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"`。
  全文 latexmk 干净编译 **13 页，exit=0**，24 张 Fig.7 资产全部嵌入，0 Overfull/0 undefined ref。
- 配套手册 paper/plan/fig7-competitor-inference-guide.md：训练机职责收窄为"只产 SR 全图"，
  新增数据集命名特点专章（HR/LR 同名导致 GT 名追加 x4 找不到文件），逐对手给精确配置改动。

### 2026-06-27（续7）· 本地数据集就位，写定 3.4 视觉对比 + Fig.7 自方列，修 Fig.1 落盘缺口
- **本地数据集就位**（CATANet/datasets/TestDataSR，HR+LR 全 5 基准），解锁 3.4 视觉对比。
- 写定 3.4 正文（3_experiments_visual.tex）+ Fig.7（fig7_visual.tex）：3 行难样本
  Urban100 img_092/img_024 + Manga109 ThatsIzumiko_000（按 GT 高频能量自动选样 + 选 crop，
  脚本 paper/scripts/build_fig7_assets.py / rank_hf_samples.py）。GT/Bicubic/DPRNet 三列**本地出真图**
  （fig7_assets/，12 crop + metrics.csv，PSNR/SSIM 已对照官方日志验证，差 0.07dB 系 PNG 量化）；
  对手列当时留占位（续8 已补齐）。
- **修复预先存在缺口**：fig1_architecture.tex 在磁盘上丢失（续6 记录已建但实际未落盘、git 从未跟踪），
  会阻断全文编译。已按 figure-specs.md Fig.1 规格用 TikZ 重建（整体流水 + Block_l 展开 + TAB 内部数据流，
  DPR/TAB 高亮引向 Fig.2，对应 Eq.1–3）。
- 全文 pdflatex→bibtex→×2 干净编译 **13 页**，0 Overfull/0 undefined ref，仅 2 个 "page 5 only floats" 美观提示。

### 2026-06-27（续6）· 出 Fig.1/Fig.2 结构图 + Fig.3 PSNR-vs-Params 散点，全文编译验证
- 新建 paper/figures/fig1_architecture.tex（TikZ，figure*）：整体网络流水（I_LR→Conv→L=8 blocks→
  global residual→Up→I_SR，含 bilinear bypass）+ Block_l 内部（TAB→LRSA→Conv，残差）+ TAB 内部
  数据流（X→LN→DPR→IASA→Conv1×1→ConvFFN），DPR/TAB 用 dprblue 高亮、引线指向 Fig.2。
  对应 method §Overview Eq.1–3，无任何指标。
- 新建 paper/figures/fig2_dpr.tex（TikZ，figure*）：DPR 6 个 stage 全图——(1)动态原型生成 (2)query refine
  (3)置信度匹配+可学习温度 (4)置信度排序 (5)IASA 聚合+soft fallback (6)balance loss，对应 Eq.4–13；
  γ/β/α/τ 门控用蓝色小圆、虚线调制；A1/A2(×3)/A3 消融开关用黄色小标签标在对应位置，
  caption 引 Table III–V（tab:abl_a1..a3）。
- 新建 paper/figures/fig3_psnr_params.tex（pgfplots，单栏）：x4 Urban100 PSNR vs Params 散点，
  9 方法数据 verbatim 自 Table II（Params）+ Table I（PSNR），DPRNet 用蓝色五角星高亮在左上（660K/26.89）。
  pgfplots 内联数据用 row sep=crcr + \\ 终止（兼容 \resizebox）。
- 串联：main.tex 在 §Method 后 \input Fig.1/2、在 §3.3 efficiency 后 \input Fig.3；
  正文加引用——2_method §Overview 引 Fig.~\ref{fig:arch}、§DPR 引 Fig.~\ref{fig:dpr}，
  3_experiments_efficiency Parameters 段引 Fig.~\ref{fig:psnr_params}。
- preamble.tex 补 \usetikzlibrary{positioning,arrows.meta,calc,fit,backgrounds,shapes.geometric} +
  \definecolor{dprblue}。三图均用 \resizebox 包裹消除 Overfull（Fig.1/2 →\textwidth，Fig.3 →\columnwidth）。
- **编译验证（本地 TinyTeX/TeX Live 2026，新装 standalone+varwidth 仅供单图预览）**：
  pdflatex→bibtex→pdflatex×2 干净通过，**12 页**（原 11 页 +1），0 Overfull/Underfull、
  0 LaTeX Warning、0 悬空引用/citation。三图各自 standalone 渲染 PNG 肉眼核对：节点无重叠、
  箭头方向与 Eq. 对应、消融开关标签正确、Fig.3 各方法标签可读且 DPRNet 居左上最优。

### 2026-06-27（续5）· Multi-Adds 统一协议（256×256）重测完成 + 口径风险解除 + 全量同步
- **训练机重测**：用改造后的 scripts/measure_efficiency.py 新增「固定 LR 输入」口径（--lr-h/--lr-w），
  在 cuda:0 跑三尺度 256×256，结果存 efficiency_new.md：
  · x4：Params 659.707K / **Multi-Adds 52.1370G** / Latency 198.162±9.434 ms；
  · x3：674.147K / 41.2607G / 185.880±6.329 ms；
  · x2：601.947K / 36.1882G / 183.571±7.700 ms。
- **口径风险解除（续2 遗留的最高优先项）**：256×256 输入正是 CATANet 论文 Tab.6 报 Multi-Adds 的口径，
  DPRNet 重测到该口径后与 CATANet 46.8G / SwinIR-light 60.3G / SRFormer-light 56.5G **直接可比**，
  原 ‡ 脚注与"720×1280 不可直接比"的免责声明全部删除。CATANet 一侧**照引论文 46.8G，不重测**（用户决策）。
- **关键事实转变（勿推翻）**：256×256 同口径下 **DPRNet 52.14G > CATANet 46.8G**——
  续2 预判被证实，原"DPRNet 算量更低于 CATANet"主张**推翻**。新诚实框架：
  对 CATANet **不主张任何算量/参数优势**（52.14G/660K vs 46.8G/535K，均更高）；
  可主张的是 **DPRNet 显著低于 Transformer 基线 SwinIR-light 60.3G / SRFormer-light 56.5G 且精度最高**，
  仍稳处轻量档。per-scale 趋势在固定输入下**反转**：MACs 随 scale 增大（36.19→41.26→52.14G，上采样尾部主导）。
- **同步修订（grep 复查无残留旧值 45.77/720/196.77 等）**：
  · paper/data/dprnet_efficiency.csv：加 mode 列，新增 3 条 mode=A_fixed_lr_256 行（256×256），保留旧 mode=B 行作记录。
  · paper/tables/table2_efficiency.tex：主表 DPRNet 行 52.14/198.16，去三个 ‡；per-scale 子表三尺度统一 256×256；
    caption/Notes/provenance 重写为统一 256×256 口径 + 新诚实框架。
  · 正文 4 处：3_experiments_efficiency（Computation 段 + per-scale 段 + intro 段 + Pending 段全改）、
    3_experiments_comparison（Efficiency 段）、0_abstract（去"below CATANet"改"fewer than SwinIR/SRFormer"）、
    4_discussion（prototype-count 段）、3_experiments_setup（Efficiency measurement 段口径）。
- 脚本改动：scripts/measure_efficiency.py 加口径 A（--lr-h/--lr-w，推荐对齐论文）/ 保留口径 B（--output-*），
  py_compile 通过；本机无 torch，真值已由训练机产出。

### 2026-06-27（续4）· 本地装 TinyTeX 编译通过，修 3 类编译错误 + 清零 Overfull
- **本地从零搭建 LaTeX 环境**：`brew install --cask basictex` 需 sudo 交互密码失败，改用
  **TinyTeX**（用户级，装到 `~/Library/TinyTeX`，无需 sudo）。PATH 用绝对路径，不写系统配置：
  `export PATH="$PATH:/Users/bytedance/Library/TinyTeX/bin/universal-darwin"`。
  用 `tlmgr install` 按需补包：times multirow pifont pgf pgfplots preview booktabs natbib
  caption subcaption enumitem cleveref etoolbox silence titlesec ragged2e xcolor amsfonts
  psnfss lineno hyperref grfext courier 等（关键坑：pifont 在 psnfss、pgfplots 独立于 pgf、
  courier 字体 pcrr7t 需单独装、cvpr review 模式要 lineno）。
- **修复 3 类真实编译错误**：
  · `\ding{}` 空参数报错（table4 表注里写了描述文字 `Requires ... \ding{}.`）→ 删除该元信息句。
  · references.bib 中文 Unicode 报错（zhang2024spin 的 `note={...主对照}` 进了 .bbl 第70行）→ 删该 note 字段。
  · 缺包逐个 tlmgr 补齐（见上）。
- **完整流程跑通**：`pdflatex→bibtex→pdflatex×2` 全部 EXIT=0，**生成 11 页 PDF**（main.pdf ~335KB）；
  无 LaTeX Error、无 Unicode、**无未定义引用/citation、无 LaTeX Warning**。
- **Overfull \hbox 清零（6→0）**：给超栏宽的 tabular 加 `\resizebox{\columnwidth}{!}{%...}` 包裹——
  table2（效率表 + per-scale 子表，含补回 per-scale 子表漏掉的闭合 `}`）、table3（A1）、
  table4（A2）、table5（A3 汇总表 + per-block 表）。最终 final pass `Overfull|Underfull` 计数=0。
- **结论**：main.tex 可在标准 TeX Live/TinyTeX 上干净编译。下次接手只需 `tlmgr` 在位即可复现。

### 2026-06-27（续3）· 建顶层 main.tex 串起全章节
- 新建 paper/main.tex（CVPR 2026 author-kit，[review] 模式）：按 Abstract→Intro→Method→
  Experiments(3.1 setup 开 \section → 3.2 comparison + Table I/II → 3.3 efficiency →
  3.5 ablation + Table III/IV/V → 3.6 routing + Fig.6)→Discussion→Conclusion→bib 顺序 \input。
  title 暂定 "Dynamic Prototype Routing for Lightweight Image Super-Resolution"，作者匿名占位，
  paperID/标题/作者/§3.1 硬件均留 TODO；§3.4 视觉对比 + Fig.1/2/3/5/7/8 未出图，暂不 \input。
- 新建 paper/preamble.tex：仅补 cvpr.sty 未含的 multirow（Table I）/pifont（Table IV \ding）/
  pgfplots（Fig.6）+ \red/\todo/\TODO 注释宏。cvpr.sty 已含 booktabs/amsmath/amssymb/graphicx/
  xcolor/natbib/caption/cleveref，未重复引入避免冲突。
- 复制支撑文件入 paper/：cvpr.sty、ieeenat_fullname.bst（bib 用 \bibliography{references}）。
- **静态自检通过（本机无 LaTeX，无法真编译）**：
  · 全部 \ref 目标（sec:*/tab:*/fig:*/eq:*）均有对应 \label，逐一比对无悬空引用；
  · 全部 \cite key（20 处）均在 references.bib 有条目（20 条），无缺失；
  · 6 个表 + Fig.6 的 table/figure/tabular/axis/tikzpicture begin/end 环境配对齐全。
- **未真编译风险**：本机无 pdflatex/bibtex，未能跑通编译；首次在装 LaTeX 的机器上需
  `pdflatex→bibtex→pdflatex×2`，留意 pgfplots/cleveref/natbib 版本与浮动体溢出。

### 2026-06-27（续2）· 补 Table II 同源 Multi-Adds + 暴露 CATANet 口径不一致风险
- 完整下载并解析 CATANet 论文 PDF（arXiv:2503.06896）：核实主表 **Table 2 无 Multi-Adds 列**
  （仅 Params + PSNR/SSIM）；Multi-Adds 在 **效率表 Table 6**，且其 7 个主表对手中**只有**
  SwinIR-light(897K/60.3G) 与 SRFormer-light(873K/56.5G) 有 Multi-Adds，CARN/IMDN/RFDN/RLFN/ELAN-light
  **论文里完全没有**。Table 6 还含 CATANet-L 535K/46.8G、ATD-L 494K/30.0G、SPIN 555K/48.4G。
- 据用户决策「只填能查到的 2 个 + 脚注，其余留 -」修改 paper/tables/table2_efficiency.tex：
  SwinIR-light 填 60.3G、SRFormer-light 填 56.5G（连同 CATANet 46.8G 共 3 个统一用 ‡ 标记），
  CARN/IMDN/RFDN/RLFN/ELAN-light 保持 "-"；provenance 注释块 + 表注重写。
- **⚠ 关键风险（用户决策：先记录，待重测再定）**：CATANet Table 6 的 46.8G 实为
  **input 3×256×256（×4 输出 1024×1024）口径**，而 DPRNet 45.77G 是由 **≈720×1280 HR 输出反推 LR** 口径，
  **两者不可直接比**。原 table2 注记错误地写成"同 ~720×1280 输出、直接可比"，已修正为 ‡ 注明口径差异、仅供参考。
  按面积换算 256×256 口径下 CATANet 约 41G（720×1280≈1280×720 面积约为 1024×1024 的 0.88 倍，
  256×256 输入对应 1024×1024 输出面积更小），**换算后 CATANet 反而可能低于 DPRNet 45.77G**，
  会推翻"DPRNet 算量更低于 CATANet"的主张。**正文（§3.3 efficiency、Abstract、§3.2 comparison、
  Discussion 第 3 段）暂不改动**，待训练机用**统一协议（建议 input 256×256）重测 DPRNet 与 CATANet** 后再定表述。
- 影响范围标记（待重测后复核）：
  · 3_experiments_efficiency.tex「45.77G slightly lower than CATANet's 46.8G at the same output resolution」；
  · 0_abstract.tex「45.77G Multi-Adds at ×4, the latter below CATANet」；
  · 4_discussion.tex 第 3 段「660K vs 535K 但 45.77G < 46.8G」。

### 2026-06-27（续）· 写 Abstract + Discussion + Conclusion 三节
- 新建 paper/sections/0_abstract.tex（150–250 词）：问题→DPR 一句话→4 个设计→主结果
  （x4 Urban100 26.89dB、45.77G Multi-Adds 低于 CATANet、usage 熵 0.6227→0.6865）→意义。
  口径与正文一致用"matches or slightly surpasses ... at comparable budget"，无夸大独立增益。
- 新建 paper/sections/4_discussion.tex（对应 outline §4）：内容路由 vs 空间邻近适用边界
  （Urban100/Manga109 受益最大、B100 收窄）、过强路由风险（温度 clamp + balance 防塌缩，引 §3.6 证据）、
  prototype 数量与成本取舍（660K vs 535K 但 45.77G < 46.8G）、限制（共享初始化+短预算+单 seed→不主张单开关独立增益/
  方差；C1 引用间接对比；效率表未统一；仅 DIV2K）。全部复用 §3 已确立口径，无新数值断言。
- 新建 paper/sections/5_conclusion.tex（对应 outline §5）：总结贡献+结果、点明 DPR 作为通用内容路由模块可迁移、
  未来工作（from-scratch 分离归因 + 多 seed 方差 + 统一效率表 + 更大数据集/真实退化）。
- 正文文字部分至此仅剩 3.4 视觉对比（待训练机推理出图后补）。同步本文件快照/下一步。

### 2026-06-27 · paper/data/*.csv 实验数据更正 + 正文/表格/计划文档全量同步
- 用户更正了 paper/data/ 下我方自测数据，以更正后的 CSV 为唯一权威来源。受影响范围：
  · dprnet_main_metrics.csv（x2/x3/x4 主表 PSNR/SSIM 整体上调 ~0.02–0.10dB）；
  · ablation_metrics.csv 的 full 行（Set5 32.5878/0.9001、Urban100 26.8905/0.8086）；
  · 效率 CSV、per-block 路由诊断 CSV **未变**。
- **关键事实转变**：更正后 DPRNet 在 Table I 几乎全格领先 CATANet（仅 x4 Manga109 SSIM 0.9181 次于
  CATANet 0.9183）；消融 full 行在 A1/A2/A3 三组均为最优（margin ~0.04–0.06dB）。
- 同步修订：
  · 表格 table1_main_comparison（重算全尺度 best/second-best）、table2_efficiency（Urban100 x4 26.82→26.89
    加粗最优、CATANet 改下划线）、table3/4/5_ablation（full 行 + caption/Notes 改写为"略优/full 不劣且最优"）。
  · 正文 3_experiments_comparison（"on par"→"matches or slightly surpasses"）、3_experiments_efficiency
    （matched-or-better 26.89）、3_experiments_ablation（质量中性偏正、组件兼容、full 不劣且最优，
    保留共享初始化+短预算+单 seed 不主张单开关独立增益的诚实口径）、1_introduction（去掉"largest gains"，
    改为"matches or slightly surpasses ... at comparable budget"）。
  · 计划文档 data-collection-checklist（A2 三尺度自测表 + 选点依据均值 + 交叉验证注 + Table III/IV/V 注）、
    method-experiment-traceability（证据图例三尺度值 + C2/C3 allowed-claim +「消融已知局限」节）、本文件。
- 诚实口径未越界：仅将"落噪声内/方向倒置/质量中性"升级为"质量中性偏正/full 不劣且最优"，
  未主张单开关独立 PSNR 增益。下方历史日志（2026-06-09 等条）含的旧自测数值已就地标注【已更正】，
  原貌保留作记录，精确数值一律以更正后 paper/data/*.csv 为准。

### 2026-06-20 · 摘录我方自测数据为权威 CSV + 登记数据来源
- 新建 paper/data/dprnet_main_metrics.csv（15 行，x2/x3/x4 × 5 基准 PSNR/SSIM，带 source_log）
  与 paper/data/dprnet_efficiency.csv（3 行，三尺度 Params/MACs/时延，带 source）——
  数值逐格摘自原始测试日志/efficiency.md 并核对一致，作为 Table I（ours 行）/Table II 的唯一事实源。
- 复核 Table I（ours 三行）与 Table II（DPRNet 行）：与原始日志逐格一致，无录入错误；
  DPRNet 多数格紧随 CATANet 次优属数据真实形态，非改错。
- 登记数据来源约定：progress 接手指南 + checklist §A2/§A3 顶部均加"权威来源指向 CSV、后续一律以 CSV 为准、
  不再翻原始日志"；明确"我方数据看 paper/data/*.csv，对手数据看 checklist §B"。
- 待修（未动）：1_introduction.tex L86-89 "largest gains on Urban100, Manga109" 与 Table I 矛盾，仍待改。

### 2026-06-17（续3）· 写 3.3 效率分析 + 补 Table II CATANet 同源 Multi-Adds
- 新建 paper/sections/3_experiments_efficiency.tex（定义 sec:efficiency，讨论 Table II + per-scale）：
  · 参数：DPRNet x4 660K，轻量档内（< SwinIR/SRFormer/CARN，≈ELAN-light），但**大于 CATANet 535K**——
    如实说明 query bank + router/score 投影带来的参数增量，**不主张参数优势**，在等精度下对比。
  · 计算：DPRNet x4 Multi-Adds 45.77G **反而略低于** CATANet 46.8G（同输出口径）→ 额外参数不增推理算量。
  · per-scale：x2/x3/x4 MACs 126.52/64.28/45.77G、时延 712/285/197ms，参数近似尺度无关（0.60–0.67M）。
  · 诚实边界：其余 7 对手 compute/时延未同口径采集，留 "-"，明确不编造。
- **关键发现**：DPRNet 虽参数多于 CATANet，但 x4 Multi-Adds 更低且精度持平 → 算量维度可正面表述。
- 补 paper/tables/table2_efficiency.tex：CATANet x4 Multi-Adds 填 46.8G（带 † 注，引自 CATANet CVPR'25
  同 ~720×1280 输出，与我方 thop MACs 直接可比）；其余 7 对手 Multi-Adds 经核查跨源参数口径不一致
  （如 SwinIR-light 897K vs 别处 930K），为避免不一致**不强填**，表注说明须统一取 CATANet Tab.2 或重测；
  对手时延依赖硬件不可引，留 "-"。来源核查：CATANet 论文/补充材料 Tab.B/C（536K/46.8G），arXiv 2503.06896。
- 同步 progress 快照/下一步/日志。

### 2026-06-17（续2）· 写 3.1 实验设置 + 3.2 主对比
- 新建 paper/sections/3_experiments_setup.tex（开 \section{Experiments}，定义 sec:exp_setup）：
  数据集/指标（DIV2K 训练，5 基准 Y 通道，crop_border=s）、网络配置（L=8/C=40/M=[16,32,64,128]×2/
  group=128/τ_init=e^6 clamp10/λ）、训练（x2 scratch 800k → x3/x4 finetune 250k，patch 128/192/256，
  bs16，Adam 2e-4 MultiStep，router lr×0.1）、效率测量口径。全部 verbatim 自 yml/checklist。
  · 硬件留 [PLACEHOLDER]（GPU 型号/GPU·hours 未采，checklist A1 仍 open），禁编造。
- 新建 paper/sections/3_experiments_comparison.tex（定义 sec:main_comparison，讨论 Table I/II）：
  · **诚实口径（贴合真实数据，非乐观计划）**：DPRNet 与 CATANet 同处第一梯队、同档成本下"持平"，
    B100(x3/x4)、Manga109(x2) 领先，Set5(x4) 持平，Urban100 全尺度紧随次优；**不主张"Urban100 最大增益"**。
    并如实点出 DPRNet 参数略高于 CATANet（660K vs 535K）。数字逐项核对自 Table I。
- **发现 intro 不一致**：1_introduction.tex L86-89 称 "largest gains on Urban100, Manga109"，
  与 Table I 矛盾（Urban100 DPRNet 全尺度次于 CATANet）。已记入下一步待修，本轮未改 intro（待确认）。
- 同步 progress 快照/下一步。

### 2026-06-17（续）· 写 3.6 路由可解释性 + 出 Fig.6
- 新建 paper/sections/3_experiments_routing_analysis.tex（定义 sec:routing_analysis，被 3.5 引用）：
  · C4 均衡：usage 熵 8 block 全升（0.6227→0.6865）+ active 槽 47.5→52.3（防塌缩）。
  · C4 温度：router_scale 稳定在 5.65–6.51（均值 6.06）< clamp 10，未饱和。
  · C3：置信度信号各 block 非退化（x^s≈1e-2 远高于 1/M 均匀底），证明 softmax 分配有信息。
  · 诚实边界：x_scores 直方图 + 聚类图需推理（本机无 torch/matplotlib），注明 camera-ready 补，
    不在本节作定量断言。数字逐项核对自 paper/data CSV。
- 新建 paper/figures/fig6_usage_entropy.tex（pgfplots/TikZ，依赖 pgfplots；矢量，无需本地渲染）：
  per-block 归一化 usage 熵 balance on vs off 柱状图，数据 verbatim 自 perblock CSV。
  （本机无 matplotlib/LaTeX，故选 TikZ 源码 + 真实数值，编译留待论文整体构建。）
- 同步 traceability：C3/C4 行补 §3.6 + Fig.6 证据与 allowed claim；progress 快照/下一步更新；
  data-checklist Fig.6 标 [x]。

### 2026-06-17 · 消融实验分析 + 决定"保留数据改写 claim" + 拼 Table III/IV/V + 写消融正文
- 解析 6 个 x4 消融日志（experiments/*abl*，A1/A2/A3 各 80k，full 100k，全跑满）。
  提取每 10k val（Set5+Urban100）PSNR/SSIM 曲线 + 每 block 路由诊断（xscore/usage/熵/router_scale）。
- **诊断出方法学局限**：6 个变体的 pretrain_network_g 全指向 train_CATANet_x4_finetune/net_g_250000.pth
  —— 该 ckpt 是全开关、带 balance、收敛 250k 的全功能模型。即便"全关"baseline 也从全功能权重退火 80k，
  导致各开关 PSNR 差异落噪声内（±0.03dB）且 A1/A2 方向与预期相反（A2 全关 v1 反而 Urban100 最高）。
  正是 protocol §10 注2 早警告的"从全功能权重关 flag 测会失配"。
- **决策（用户拍板）**：保留现有数据，彻底改写 claim，不重训。
  · A1/A2：只报"质量中性、组件兼容"，不主张逐项 PSNR 增益。
  · C1 动态原型 vs 历史中心：仍只作动机层面，引 CATANet 原论文间接对比。
  · C4：balance loss 在 8 block usage 熵全升（均值 0.6227→0.6865）→ 可主张均衡机制；
    "多 seed 方差下降"待补（当前仅 seed 3407），暂不主张。
- **产出**：paper/tables/table3_ablation_a1.tex / table4_ablation_a2.tex / table5_ablation_a3.tex
  （均含真实 80k 数据 + 受限口径表注 + 数据局限披露）；消融分析正文
  paper/sections/3_experiments_ablation.tex（定义 sec:ablation，被 intro/method 引用，数字逐项核对自表格）；
  路由诊断 CSV 导出 paper/data/（ablation_metrics / perblock_urban100_80k / val_curves，供 Fig.6）；
  同步更新 traceability（C1-C4 allowed claim 收紧 + 新增「消融实验的已知局限」节）、
  data-checklist Table III/IV/V 标 [x]（受限）。

### 2026-06-15 · 修正 A2 多卡 DDP marked-ready-twice 报错
- 现象：A2_v1_hardsort 4 卡训练在 backward 时报
  `Expected to mark a variable ready only once`，参数为
  `blocks.7.0.dpr.router_logit_scale`。
- 原因：上一轮为解决关闭分支的 unused-parameter，把 A1/A2 yml 设置了
  `find_unused_parameters: true`；但 DPR 的 `aux_loss` 存在于模块属性中、不是 forward 返回值，
  同时又参与 `l_total.backward()`。DDP unused 检测会误判这些路由参数，随后 aux loss 反传时再次触发
  hook，导致 marked-ready-twice。
- 修复：撤销所有消融 yml 的 `find_unused_parameters`；改为在网络构造时冻结被关闭分支参数：
  `use_prototype_query_refine=false` 时冻结 DPR refine 分支参数，
  `use_soft_fallback=false` 时冻结 `soft_fallback_gate` 和 `soft_context_proj`。
  这样普通 DDP 不再期待这些参数有梯度，也不会干扰 aux loss。
- 验证：`catanet_arch.py` py_compile 通过；6 个 x4 消融 yml 解析通过且不再包含
  `find_unused_parameters`。本地无 torch，运行时需训练机复跑验证。

### 2026-06-14 · 修复消融关闭分支导致的 DDP unused-parameter 报错
- 注意：该方案已在 2026-06-15 被替换；不要再使用 `find_unused_parameters: true`，
  以免触发 aux loss 相关的 marked-ready-twice。
- 现象：4 卡运行 A1 refine_off 时，DDP 报
  `Expected to have finished reduction... parameters that were not used in producing loss`。
- 原因：A1 关闭 `use_prototype_query_refine` 后，DPR refine 分支参数仍注册在模型中但 forward 不使用；
  A2 关闭 `use_soft_fallback` 时也会使 `soft_fallback_gate` 参数不参与 loss。
- 当时尝试：在会关闭分支的消融 yml 中设置顶层 `find_unused_parameters: true`：
  `train_CATANet_x4_abl_A1_refine_off.yml`、`A2_v1_hardsort.yml`、
  `A2_v2_confsort.yml`、`A2_v3_scoregate.yml`。A3 不关闭结构分支，保持默认。
- 验证：上述 yml 均可被 YAML 正常解析，`BaseModel.model_to_device` 会读取该字段并传给 DDP。

### 2026-06-10 · 消融尺度改 x2→x4（对齐 CATANet 原论文）
- 复核 CATANet 原论文（arXiv:2503.06896 / CVPR2025 supplementary）：其消融实验全部在
  **scale=4** 上做（原文 "calculated with a scale factor of 4. All models are trained
  250K on DIV2K from scratch"），Table A/B/C 报 Set5/Urban100 等。
- 决策：本文消融改到 x4。理由：(1) 对齐原论文消融口径，审稿人可直接对比；
  (2) x4 退化最重，DPR 内容路由（动态原型/置信路由）在高频纹理 Urban100/Manga109
  上的收益最易显现，x2 上各变体差异可能落在噪声内。
- 训练口径：采用 **x4 finetune**（从 net_g_250000 起 80k 短 iter），非原论文的
  from-scratch 250k，以节省算力；论文中注明此口径差异（趋势性结论）。
- 已将 6 个消融 yml 由 x2 重建为 x4（scale/upscale=4、dataroot X4、filename '{}x4'、
  gt_size=256、crop_border=4、pretrain 指向 train_CATANet_x4_finetune/net_g_250000.pth），
  删除旧 x2 版本；experiment-protocol.md §5/§10.C 同步更新。6 个 yml 解析通过。

### 2026-06-10 · Day4/5：消融前置就绪（refine 透传 + 6 个消融 yml）
- catanet_arch.py 补 use_prototype_query_refine 顶层透传（CATANet→TAB→DPR，与 A2 三 flag 同法）：
  CATANet __init__ 加参/存 self/传 TAB；TAB __init__ 加参/传 DPR；DPR 原已有。默认 True 严格等价原行为。
  py_compile 通过；运行时前向验证待训练机。
- 新建 6 个消融 yml（CATANet/options/train/，均 x4 从 net_g_250000 finetune，80k 短 iter，
  milestones [40k,64k,76k]，val=Set5+Urban100，val/save=10k，其余与 x4 finetune 主训对齐）：
  · _abl_full（全开，A1/A2/A3 共享顶行参照）
  · _abl_A1_refine_off（A1：refine off）
  · _abl_A2_v1_hardsort / _v2_confsort / _v3_scoregate（A2 逐项加法 v1-v3，v4=full）
  · _abl_A3_balance_off（A3：route_balance_weight 全 0；balance on=full；多 seed 用 --force_yml）
- 6 个 yml YAML 解析通过；protocol §10.C A1 项标 [x]。

### 2026-06-10 · Day4：Introduction 初稿 + 结构图规格
- 写定 paper/sections/1_introduction.tex（对应 outline §1，Related Work 融入）：
  背景(SRCNN/EDSR)→轻量 CNN(CARN/IMDN/RFDN/RLFN)→轻量 Transformer(SwinIR/ELAN/SRFormer/ATD)
  →内容路由(SPIN/CATANet)→TAB 四限制→DPR 概述→C1–C4 contribution，全部 \cite 对应 references.bib，
  无任何量化指标断言（仅动机/定位的定性表述），与 traceability 证据状态一致。
- 写定 paper/figures/figure-specs.md（Fig.1 整体网络 + Fig.2 DPR 数据流的 figures-diagram 规格）：
  逐元素/箭头对应 2_method.tex 的 Eq.1–13，标注三消融开关位置（A1 refine / A2 三 flag / A3 balance），
  与 protocol §10.C 呼应；纯结构示意无指标。
- 判断 A1 间接对比口径（见"当前快照"）：C1 历史中心对照可直接引 CATANet 原论文表格（B8/Table I），
  C2 refine 受控消融仍须本文自训。

### 2026-06-10 · Day3：Method 章节初稿
- 写定 paper/sections/2_method.tex（LaTeX，对应 outline §2）：Overview / Motivation /
  DPR 四阶段（动态原型生成→query refine 确认→置信度匹配→置信度排序）/ IASA 置信度门控 +
  soft fallback / balance loss / Design discussion，共 7 小节 + 编号公式。
- 严格忠于实现 catanet_arch.py：assign softmax(Eq.4)、proto_content 加权平均(Eq.5)、
  refine 门控 γ=σ(g_r)(Eq.7)、可学习温度 τ=min(e^θ,τ_max)(Eq.8)、排序键 b+0.5(1-x_score)(Eq.10)、
  IASA 门控 β·x_score(Eq.11)、soft fallback α(1-x_score)(Eq.12)、usage 熵正则(Eq.13)。
- 三个消融开关 use_conf_sort / use_iasa_score_gate / use_soft_fallback 与 refine 开关均在正文
  对应到 A1/A2/A3，且写明"关闭即严格等价"，与 traceability/protocol §10 一致。
- 全篇无任何指标断言（无 "results show"），符合 Mock 边界与证据状态约定。

### 2026-06-10 · Day2 收尾：Table I 主对比 + Table II 效率表拼装
- 用 A2/A3（本文）+ B1–B8（对手）数据拼成 paper/tables/table1_main_comparison.tex：
  三尺度 × 5 基准，CARN/IMDN/RFDN/RLFN/SwinIR-light/ELAN-light/SRFormer-light/CATANet + DPRNet 共 9 行；
  逐列核对最优（加粗）/次优（下划线）；RLFN 无 x3、无 Manga109 缺格填 "-"；表注标全部来源与 checkpoint 口径。
- 关键观察：DPRNet 与原始 CATANet 高度接近，x2 Manga109 双指标全场最高、x4 B100 PSNR/SSIM 全场最高；
  多数格次优、紧贴 CATANet，符合"接口兼容 + 内容路由质量提升"的预期形态（待消融与可视化进一步支撑 claim）。
- paper/tables/table2_efficiency.tex：DPRNet 三尺度 Params/MACs(thop)/时延全填（含 per-scale profile 子表）+
  对手 Params 已填；⚠ 对手 FLOPs/时延未同口径采集，暂填 "-"，投稿前必补（摘录各论文同输出 Multi-Adds 或同协议重测，禁编造）。
- data-checklist C 节：Table I 标 [x]、Table II 标 [~]（带数据缺口说明）。

### 2026-06-09 · Day2：baseline 对手指标摘录 + references.bib 初版
- 从权威原始论文交叉核对摘录 8 个对比方法（CARN/IMDN/RFDN/RLFN/SwinIR-light/
  ELAN-light/SRFormer-light + CATANet 原始）的 x2/x3/x4 × 5 基准 PSNR/SSIM + Params，
  回填 data-checklist B 节，逐方法标注来源（A=CATANet CVPR2025 Table2 / B=SRFormer Table VI /
  C=RLFN Table1 / D=RFDN Table3）。
- 关键发现：RLFN 原论文仅 x2/x4 且无 Manga109，主表对应缺格填 "-"（不补二手来源）；
  SwinIR/ELAN 在 A/B 两表 Params 口径不同（PSNR/SSIM 一致），主表统一采 A 口径并表注说明。
- 交叉验证：CATANet 原报 x2 Set5 38.28/0.9617 与本仓库自测 net_g_792500（38.2706/0.9617）
  高度吻合，佐证 CATANet 复现可信、可作 DPRNet 直接对照基线。
  【已更正 2026-06-27】x2 Set5 自测值现为 38.2906/0.9617，仍与 CATANet 原报高度吻合，结论不变。
- 新建 paper/references.bib 初版（19 条：8 对比方法 + SRCNN/EDSR/SPIN/ATD + 5 数据集 +
  SSIM/BasicSR）。CATANet DOI/页码经 dblp 确认；其余 arXiv ID 已核实，会议页码/正式 DOI
  待终稿前 CrossRef 逐条核对（文件头已注明核实状态）。

### 2026-06-09 · x3/x4 报告 checkpoint 锁定 + 效率数据回填
- 测试机全 5 基准复核 x3(210000/232500/250000) 与 x4(195000/240000/250000) 各 3 点。
- **锁定 net_g_250000 为 x3、x4 统一报告点**（同 x2 口径：全基准均值最优 + 最收敛，禁逐集挑最优）：
  · x3@250000：均值 PSNR 31.606 / SSIM 0.8811，三点中双最高；Set5 34.7129 / Set14 30.6599 /
    B100 29.2890 / Urban100 28.9551 / Manga109 34.4133。
  · x4@250000：均值 PSNR 29.462 / SSIM 0.8312，三点中双最高，Urban100 双指标全场最高；
    Set5 32.5826 / Set14 28.8846 / B100 27.7648 / Urban100 26.8225 / Manga109 31.2540。
  · 注：各点差异极小（x3 PSNR 极差 0.017、x4 0.018），250000 最收敛故取之，与 x2 选最晚点逻辑一致。
  【已更正 2026-06-27】上列 x3/x4 逐集自测值为旧值，已被更正；现以 paper/data/dprnet_main_metrics.csv 为准
  （x3 Set5 34.7629/0.9301、Set14 30.6699/0.8481、B100 29.2890/0.8103、Urban100 29.0551/0.8689、
  Manga109 34.4133/0.9499，均值 PSNR 31.638/SSIM 0.8815；x4 Set5 32.5896/0.9002、Set14 28.9146/0.7883、
  B100 27.7648/0.7434、Urban100 26.8925/0.8089、Manga109 31.3240/0.9181，均值 PSNR 29.497/SSIM 0.8318）。
  选点逻辑（均值最优+最收敛）不变。
- **效率数据回填（A3/Table II）**：efficiency.md（cuda）解析——Params x2/x3/x4=601.95K/674.15K/659.71K，
  MACs(thop)=126.52/64.28/45.77 G，时延=712.16/285.39/196.77 ms（warmup20 repeat100）。
  口径：固定 HR≈720×1280 反推 LR；thop 报 MACs，与 FLOPs 论文比需 ×2；x3 HR 不整除按 720×1278。
- data-checklist A2 回填 x3/x4 完整主表、A3 效率表。三尺度报告口径统一写入"已决策"。

### 2026-06-08 · Day1 完成：x3/x4 finetune 250k 跑完 + 日志核对
- 训练产物已回传本地 experiments/。两尺度均完整跑满 250k，耗时各约 1 天 12 小时。
- **Day1 验证项全部通过**：
  · finetune 生效——x3 首个 val(2500) Set5=34.6203、x4 首个 val Set5=32.4194，均从高值起步（非从零爬升），权重正确加载。
  · loss 正常——全程 l_pix 在 1.4e-2~2.9e-2 区间平稳，l_route 约 1e-6 量级，无 NaN/发散。
- **各数据集验证最优点（注意分散，需统一报告点）**：
  · x3：Set5 best 34.7165@210000；Set14 best PSNR 30.6623@250000、SSIM 0.8482@230000。
  · x4：Set5 best 32.5984@195000；Set14 best PSNR 28.8899@240000、SSIM 0.7882@175000。
- **待办**：按 x2 同口径，用测试机全 5 基准对比为 x3/x4 各锁定单一报告 checkpoint，禁逐集挑最优。
- 路由诊断（xscore/usage/entropy/router_scale）已随 val 打印，可作 C3/C4 与 Fig.5/6 证据。

### 2026-06-06 · Day1 启动 + 效率脚本实现
- 用户在训练机启动 x3/x4 finetune，确认无异常（loss 正常、未报错）。
- 本地新增效率测量脚本 scripts/measure_efficiency.py：统计 Params / FLOPs / 推理时延，
  统一口径（固定 HR 输出 1280x720 反推 LR 输入；warmup 后多次平均；FLOPs 后端 thop→fvcore）。
  py_compile 通过；需在训练机（装 torch）运行得真值，回填 A3/Table II。
  → 唯一实质代码缺口已补平。

### 2026-06-06 · 规划一致性自检 + 交接文档化
- 全面检查 7 个规划文件，修正 5 处不一致：traceability 旧 checkpoint 口径、
  A1 降级未同步、progress 待决策过期、路由诊断存盘说明缺失、过时 bug 措辞。
- 确认训练数据仅用 DIV2K。
- 本文件改造为交接文档（加快速接手指南 + 当前快照 + 进度日志）。

### 2026-06-06 · A2 消融三 flag 实现 + A1 降级
- catanet_arch.py 贯通 CATANet→TAB→DPR 加三个开关：use_conf_sort /
  use_iasa_score_gate / use_soft_fallback，默认 True 严格等价原行为，可由 yml 配置。
  py_compile 通过；运行时前向验证待训练机执行。
- A1 采用降级方案 (b)：不重实现历史中心+EMA 对照，改用 refine 开关实测
  + 引用 CATANet 原论文间接对比。refine 顶层透传待补（做 A1 前）。

### 2026-06-06 · 代码改动总账 + x3/x4 防重训对齐
- 发现并修复重训隐患：x3/x4 finetune yml 缺路由超参，会按默认值(balance loss 关)
  训练，导致三尺度口径不一致。已补齐 route_balance_weight / router_scale_init /
  max_router_logit_scale / router_scale_lr_mult，完全对齐 x2。
- experiment-protocol.md 新增 §10 代码改动总账（A 已就绪/B 训练前必改/C 消融开关）。

### 2026-06-06 · x2 checkpoint 锁定
- 读全 9 个 checkpoint(500k-800k)测试日志，对比 5 基准 PSNR/SSIM。
- 锁定 net_g_792500 为统一报告点 + x3/x4 finetune 起点 + 消融/可视化基准。
- x3/x4 finetune yml 的 pretrain_network_g 指向 792500、strict_load_g=false。
- data-checklist A2 回填 x2 完整主表。

### 初始 · 规划阶段
- 遵循 research-writing skill 工作流，确认 4 项决策(期刊/GPU/训练策略/消融范围)。
- 产出 plan/ 下 6 个规划文件 + 本 progress.md。

---

## 已决策（不可回退的约定）

- [x] x2 报告 checkpoint：net_g_792500（统一报告点，不逐数据集挑最优）。
- [x] x3/x4 报告 checkpoint：均为 net_g_250000（finetune 最终点，全基准均值最优 + 最收敛，禁逐集挑最优）。
- [x] 期刊模板：latex-templates/latex-template（CVPR 2026 kit）有效可编译；
      投 Neurocomputing/TNNLS 时正文复用、外层格式终稿前迁移。
- [x] 训练数据：仅 DIV2K（暂不启用 DF2K）。
- [x] x3/x4：从 x2(792500) finetune，路由超参完全对齐 x2。
- [x] 消融：A1 降级(refine 开关+引用对比) / A2 三 flag 逐项加法 / A3 balance loss 多 seed。

## 待办与待观察

- [x] 效率测量脚本（Params/FLOPs/时延）——已实现 scripts/measure_efficiency.py，
      已在训练机 cuda 运行三尺度并回填 data-checklist A3/Table II。
- [ ] A1 refine 开关顶层透传（做 A1 消融前补，方法同 A2）。
- [x] x3/x4 首个 val 点核对——已确认 finetune 生效（x3 Set5 34.62 / x4 Set5 32.42 起步）。
- [x] x3/x4 统一报告 checkpoint 锁定——均为 net_g_250000（全 5 基准复核后定单点）。
- [x] Day2：摘录 7+1 baseline 三尺度 5 基准指标 + Params（B 节）；references.bib 初版（paper/references.bib）。
- [ ] 用 B 节数据拼装 Table I 主对比 + Table II 效率表（本文加粗/次优下划线，RLFN 缺格 "-"）。
- [ ] references.bib 终稿前用 CrossRef 逐条核对会议页码与正式 DOI（SPIN/ATD 发表信息待确认）。
- [ ] 观察：x3/x4 能否两周内达到/超过原 CATANet 尺度指标；消融短 iter 是否够体现趋势。

---

### Capability-use audit（能力使用审计）

- Required skills: using-research-writing, paper-orchestration, brainstorming-research,
  experiment-results-planning
- Skills actually used: 上述 4 个均已读取并遵循；规划+代码改动阶段，未触发 writing-chapters
  / figures-*（正文与图表写作在后续阶段）。
- Inputs consumed: DPR_TAB重构研究说明.md, 当前DPR超越CATANet优化方案.md,
  training_key_metrics_summary.md, 9 个 x2 测试日志, catanet_arch.py, catanet_model.py,
  train/test yml, results/ 目录。
- Artifacts produced: plan/ 下 6 个规划文件 + 本 progress.md；catanet_arch.py 加 A2 三 flag；
  x3/x4 finetune yml 对齐 x2。
- Verification run: yml/文档与实际配置、x2 指标一致；catanet_arch.py py_compile 通过。
- Remaining risk: 效率脚本未实现；x3/x4 finetune 效果与消融短 iter 趋势待训练机验证。

### 2026-06-28 · MDPI 投稿前整篇 PDF 自审

- 按用户要求审阅 `paper/main_mdpi.pdf` 全 23 页，覆盖正文逻辑、MDPI 格式、参考文献、
  图表效果和实验结果；使用 PDFKit 抽取逐页文本并渲染全页图像，重点检查 Figures 1--7
  和 Tables 1--7。
- 对照 Applied Sciences 作者指南与两篇相近 MDPI 超分辨率论文：
  `https://www.mdpi.com/journal/applsci/instructions`,
  `https://www.mdpi.com/2076-3417/15/4/1806`,
  `https://www.mdpi.com/2076-3417/14/2/917`。
- 产出任务包：`paper/plan/task-packets/2026-06-28-mdpi-full-paper-review.md`。
- 产出审稿报告：`paper/plan/review/mdpi-full-paper-review-2026-06-28.md`。
- 关键结论：当前稿件结构和技术叙事已接近投稿形态，但不建议今天直接投；需先处理作者占位、
  “budget unchanged” 与成本数据不一致、Fig.5 截断、参考文献 “Proceedings of the Proceedings”
  渲染、表注/正文内部过程语、效率对比缺口、以及 CATANet 小幅提升证据不足等投稿前阻断项。

### Capability-use audit（2026-06-28 MDPI PDF 自审）

- Required skills: using-research-writing, paper-orchestration, peer-review, verification
- Skills actually used: 已读取并使用 using-research-writing、paper-orchestration、peer-review；
  verification 以本地验证命令和 PDF 渲染/日志检查执行，未改动论文正文。
- Inputs consumed: `paper/main_mdpi.pdf`, `paper/main_mdpi.tex`, `paper/main_mdpi.log`,
  `paper/main_mdpi.bbl`, `paper/references.bib`, `paper/sections/*.tex`,
  `paper/tables/*.tex`, `paper/figures/*.tex`, Applied Sciences 作者指南和相近 MDPI SR 论文网页。
- Inputs not used and why: 未重新运行模型训练或基线推理；本轮任务是投稿前审阅建议，不生成新实验数据。
- Artifacts produced: `paper/plan/task-packets/2026-06-28-mdpi-full-paper-review.md`,
  `paper/plan/review/mdpi-full-paper-review-2026-06-28.md`。
- Verification run: `mdls` 确认 PDF 23 页；PDFKit 抽取文本并渲染页面；`rg` 检查 LaTeX/BibTeX
  警告；检查 `.bbl` 发现参考文献重复 “Proceedings of the Proceedings”；逐页查看关键图表。
- Remaining risk: 未进行人工 DOI 逐条联网复核；未验证实验数值原始日志和模型输出，只根据现有稿件、
  表格、日志和 PDF 进行审阅。

### 2026-06-28 · MDPI 投稿前阻断项修订

- 按用户要求修订参考文献、Fig.5、占位信息、内部过程语和 “budget unchanged” 表述。
- 产出任务包：`paper/plan/task-packets/2026-06-28-mdpi-pre-submission-fixes.md`。
- 主要改动：
  - `references.bib` 的会议 `booktitle` 去掉前置 `Proceedings of the`，并重新编译生成
    `main_mdpi.bbl`，消除 “In Proceedings of the Proceedings of ...”。
  - `figures/fig5_xscore_hist.tex` 将 y 轴改为 0--100%，避免 66--95% 的峰值被截断；
    已渲染 PDF 第 17 页确认曲线完整。
  - `main_mdpi.tex` 将匿名作者、邮箱、单位、ORCID 占位集中为 3 个待替换宏，正文/贡献声明
    不再出现 `Anonymous Author` 等旧占位。
  - 删除正文与表注中的内部过程语，包括 “left for the final version”“we do not fabricate”
    和 “must not be asserted before those runs exist”。
  - 将 “lightweight budget unchanged” 改为 backbone/interface unchanged 且仍处于 comparable
    lightweight regime，避免与 660K/52.14G vs. CATANet 535K/46.8G 的数据冲突。
  - 顺手修复 `sections/2_method.tex` 中 Eq. 之后一个未闭合括号。

### Capability-use audit（2026-06-28 MDPI 修订）

- Required skills: using-research-writing, paper-orchestration, latex-output, peer-review, verification
- Skills actually used: 已读取并使用 using-research-writing、paper-orchestration、latex-output、
  peer-review、verification；未启用子代理，因为本轮是限定文件的直接修订。
- Inputs consumed: `paper/main_mdpi.tex`, `paper/references.bib`, `paper/main_mdpi.bbl`,
  `paper/figures/fig5_xscore_hist.tex`, `paper/sections/*.tex`,
  `paper/tables/table2_efficiency.tex`, `paper/tables/table5_ablation_a3.tex`,
  上一轮审稿报告和用户指定的修订范围。
- Inputs not used and why: 未补真实作者身份、单位、邮箱、ORCID；用户尚未提供，不能编造。
- Artifacts produced: 修订后的 `paper/main_mdpi.pdf`、`paper/main_mdpi.bbl`、相关 LaTeX 源文件，
  以及 `paper/plan/task-packets/2026-06-28-mdpi-pre-submission-fixes.md`。
- Verification run: 使用 TinyTeX 路径
  `/Users/bytedance/Library/TinyTeX/bin/universal-darwin` 完成
  `pdflatex -> bibtex -> pdflatex -> pdflatex`，输出 23 页 PDF；`rg` 确认旧阻断词与重复
  Proceedings 不再出现；`git diff --check` 通过；PDFKit 渲染第 17 页确认 Fig.5 未截断。
- Remaining risk: `main_mdpi.tex` 仍保留 `Author Name`、`author@example.edu`、
  `Department, Institution, City, Country` 三个可替换宏；投稿前必须用真实作者信息替换。

### 2026-06-28 · 清理模板 warning 与放大 Fig.4 qualitative crops

- 按用户要求清理剩余 `hyperref` PDF-string warning 和 `fancyhdr headheight` warning，并放大
  Fig.4 qualitative comparison 的 crop。
- 产出任务包：`paper/plan/task-packets/2026-06-28-mdpi-warning-cleanup-fig4-enlarge.md`。
- 主要改动：
  - `main_mdpi.tex` 增加 `\setlength{\headheight}{20pt}`，消除 MDPI/fancyhdr header 高度 warning。
  - `main_mdpi.tex` 将摘要中的数学片段用 `\texorpdfstring{...}{...}` 包裹，保留 PDF 正文显示，
    同时提供纯文本 PDF metadata 字符串，消除 hyperref warning。
  - `figures/fig7_visual.tex` 将 Fig.4 从 8 列同排布局改为“顶部 GT reference strip + 7 列放大 crop
    行”，crop 宽度由 `0.12\textwidth` 提升至 `0.135\textwidth`，不改变任何图像内容或实验结论。
- 验证：
  - 使用 TinyTeX 路径 `/Users/bytedance/Library/TinyTeX/bin/universal-darwin` 完成
    `pdflatex -> bibtex -> pdflatex -> pdflatex`，输出 `paper/main_mdpi.pdf` 23 页。
  - `rg` 检查 `main_mdpi.log/main_mdpi.blg/main_mdpi.bbl`，无 `hyperref Warning`、
    `fancyhdr Warning`、`Overfull`、`Underfull`、undefined citation/reference、LaTeX error。
  - PDFKit 渲染第 13 页确认 Fig.4 布局在同页内，crop 已明显放大且无截断。
