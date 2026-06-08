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
- x2 已训练完成；x3/x4 从 x2 权重 finetune；消融在 x2 上做。
- 全部指标统一用单一 checkpoint **net_g_792500** 报告，禁止逐数据集挑最优。
- 仅用 DIV2K 训练（不启用 DF2K），三尺度同口径。

**协作约定**：改 yml/代码前先核对 §10 总账；任何会改变网络结构/参数的改动都要
保证"默认行为不变"，并提示需在训练机做运行时验证。

---

## 当前快照（每次更新）

- **更新时间**：2026-06-08
- **所处阶段**：Day1 完成。x3/x4 finetune 已 250k 完整跑完（无异常）；本地完成效率脚本。
- **下一步动作（最优先）**：
  1. 为 x3/x4 各定一个**统一报告 checkpoint**（同 x2 流程：全 5 基准对比后定单点，禁逐集挑最优）。
     候选见进度日志：x3 验证最优分散在 210000/232500，x4 在 195000/215000，需测试机全基准复核。
  2. 训练机运行效率脚本：`python scripts/measure_efficiency.py --scale {2,3,4}`，回填 A3/Table II。
  3. Day2 写作：抄录 baselines 主表对手指标 + references.bib 初版。
- **阻塞项**：无（DF2K、checkpoint、模板均已决策闭环）。

---

## 进度日志（倒序，最新在上）

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
- [x] 期刊模板：latex-templates/latex-template（CVPR 2026 kit）有效可编译；
      投 Neurocomputing/TNNLS 时正文复用、外层格式终稿前迁移。
- [x] 训练数据：仅 DIV2K（暂不启用 DF2K）。
- [x] x3/x4：从 x2(792500) finetune，路由超参完全对齐 x2。
- [x] 消融：A1 降级(refine 开关+引用对比) / A2 三 flag 逐项加法 / A3 balance loss 多 seed。

## 待办与待观察

- [x] 效率测量脚本（Params/FLOPs/时延）——已实现 scripts/measure_efficiency.py，
      待训练机运行得真值并回填。
- [ ] A1 refine 开关顶层透传（做 A1 消融前补，方法同 A2）。
- [x] x3/x4 首个 val 点核对——已确认 finetune 生效（x3 Set5 34.62 / x4 Set5 32.42 起步）。
- [ ] x3/x4 统一报告 checkpoint 锁定——需测试机全 5 基准对比定单点（同 x2 流程）。
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
