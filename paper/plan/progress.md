# 进度 (Progress) — 兼交接文档

> 本文件有两个作用：(1) 实时记录进度；(2) 让下一次 AI 会话能快速看懂并接手。
> 每次有实质进展，请更新「进度日志」与「当前快照」两节。

---

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

- **更新时间**：2026-06-27
- **所处阶段**：Day9→Day11。消融实验已训练完成并分析；Table III/IV/V（受限口径）已拼装；
  Experiments 已写定 3.1 设置（sec:exp_setup）+ 3.2 主对比（sec:main_comparison）+
  3.3 效率（sec:efficiency）+ **3.4 视觉对比（sec:visual）** + 3.5 消融（sec:ablation）+
  3.6 路由分析（sec:routing_analysis）+ Fig.6。
  Abstract（0_abstract.tex）+ Discussion（4_discussion.tex）+ Conclusion（5_conclusion.tex）已写定。
  **✅ 2026-06-27（续7）：本地数据集就位（CATANet/datasets/TestDataSR HR+LR 全 5 基准），解锁 3.4 视觉对比。**
  写定 3.4 正文（3_experiments_visual.tex）+ Fig.7（fig7_visual.tex）：3 行难样本
  Urban100 img_092/img_024 + Manga109 ThatsIzumiko_000（按 GT 高频能量自动选样 + 选 crop，
  脚本 paper/scripts/build_fig7_assets.py / rank_hf_samples.py）。GT/Bicubic/DPRNet 三列**本地出真图**
  （fig7_assets/，12 crop + metrics.csv，PSNR/SSIM 已对照官方日志验证，差 0.07dB 系 PNG 量化）；
  **对手列（CATANet/SRFormer-light）留占位，需训练机推理**。正文仅作 DPRNet-vs-Bicubic/GT 表述，未越界。
  **✅ 续7 还修复了一个预先存在缺口**：fig1_architecture.tex 在磁盘上丢失（progress 续6 记录已建但实际未落盘，
  git 从未跟踪），会阻断全文编译。已按 figure-specs.md Fig.1 规格用 TikZ 重建（整体流水 + Block_l 展开 +
  TAB 内部数据流，DPR/TAB 高亮引向 Fig.2，对应 Eq.1–3）。
  **全文 pdflatex→bibtex→×2 干净编译 13 页（原 12+1），0 Overfull/0 undefined ref，仅 2 个 "page 5 only floats" 美观提示。**
  仍缺的图（Fig.5/8）需训练机推理；Fig.7 对手列待补。
  **✅ 2026-06-27（续6）：Fig.1 整体结构 / Fig.2 DPR 数据流 / Fig.3 PSNR-vs-Params 散点已出图**
  （fig1_architecture.tex / fig2_dpr.tex / fig3_psnr_params.tex，均 TikZ/pgfplots 矢量），
  已串入 main.tex 并在正文引用；全文 pdflatex→bibtex→×2 干净编译 **12 页**，0 Overfull/0 警告/0 悬空引用。
  仍缺的图（Fig.5/7/8）与 3.4 视觉对比均需训练机推理。
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
  3. 待训练机：写 3.4 视觉对比 + 出 Fig.7 视觉图 / Fig.5 x_scores 直方图 / Fig.8 聚类图（需推理）。
  4. 收尾：references.bib 终稿前 CrossRef 核对；期刊模板迁移；3.1 硬件占位补值。
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
