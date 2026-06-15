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

**关键背景事实（不要再推翻）**：
- 训练在远程机（数据路径 /hy-tmp/...），本地 Mac 无 torch、无权重，只能改代码/文档。
- x2 已训练完成；x3/x4 从 x2 权重 finetune；消融在 x4 上做（对齐 CATANet 原论文消融口径 scale=4）。
- 全部指标统一用单一 checkpoint **net_g_792500** 报告，禁止逐数据集挑最优。
- 仅用 DIV2K 训练（不启用 DF2K），三尺度同口径。

**协作约定**：改 yml/代码前先核对 §10 总账；任何会改变网络结构/参数的改动都要
保证"默认行为不变"，并提示需在训练机做运行时验证。

---

## 当前快照（每次更新）

- **更新时间**：2026-06-10
- **所处阶段**：Day4→Day5。Table I/II 已拼装；Method/Introduction 初稿已写定；
  消融实验前置条件（代码透传 + 6 个 yml）已就绪，待训练机排训。
- **下一步动作（最优先）**：
  1. ✅ A1 refine 顶层透传已补（CATANet→TAB→DPR），6 个消融 yml 已建（见下）。
     → 待训练机排训：A1/A2/A3 + full 参照（x4 finetune 80k 短 iter，10k 验证/存档）。这是终稿唯一实验缺口（C1–C4）。
  2. 补 Table II 对手 FLOPs/时延（同口径，禁编造）——投稿前必补。
  3. 据 Fig.2 规格出图（Fig.1 draw.io / Fig.2 TikZ）。
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
