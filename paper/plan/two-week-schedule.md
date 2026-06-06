# 两周时间计划表 (Two-Week Schedule)

目标：两周内完成 x3/x4 训练、全部投稿所需数据/图表收集、消融实验、论文初稿。
前提：2-4 张 GPU 可并行；x3/x4 从 x2 权重 finetune（各 250k iter）。
策略：训练（机器跑）与写作/数据整理（人/AI 做）并行，最大化两周产出。

注：以下「天」为工作日序号，可按实际起始日平移。⚙=机器任务，✍=写作，📊=数据/图表。

---

## 第一周：训练启动 + 数据基建 + 写作开篇

### Day 1（启动日）
- ⚙ 启动 x3、x4 finetune（双卡并行，pretrain_network_g 已指向 x2 锁定权重 net_g_792500.pth）。
  - 命令：`CUDA_VISIBLE_DEVICES=0 torchrun ... -opt options/train/train_CATANet_x3_finetune.yml`
  - 另一卡同时跑 x4（--master_port 错开避免冲突）。
  - 已就绪：yml 的 pretrain_network_g、strict_load_g=false、路由超参均已对齐 x2（见 protocol §10.B）。
- 📊 用 x2 锁定 checkpoint（792500）跑全 5 基准测试，指标已导出回填（见 checklist A2）。
- ✍ 整理 x2 已有训练曲线数据（training_key_metrics.csv），x2 报告点已定为 792500。
- 验证：x3/x4 首个 val 点 PSNR 应从高值起步（finetune 生效）；loss 正常下降、无 NaN。

### Day 2
- ⚙ 监控 x3/x4 训练（看 val Set5/Set14 是否接近/超过原 CATANet 对应尺度指标）。
- 📊 跑效率测试脚本：统计 DPRNet 与 CATANet 的 Params / FLOPs / 推理时延（固定输入）。
- ✍ 完成文献检索：列出 baselines（CARN/IMDN/RFDN/RLFN/SwinIR-light/ELAN-light/
  SRFormer-light/CATANet）并从原论文/公开表格抄录其 x2/x3/x4 指标（标注来源）。
- 产出：references.bib 初版 + 主表对手指标填入 tables/。

### Day 3
- ⚙ 训练持续；若 x3/x4 早期指标不及预期，检查 lr / 是否需调 finetune milestones。
- ✍ 写 Method 章节初稿（基于 DPR_TAB重构研究说明.md，可直接转化为正式方法描述）。
  - 子节：动态原型生成 / prototype refine / token-to-prototype 路由 / 置信度排序 / IASA 聚合。
- 📊 画方法结构图（DPR 数据流图、整体网络图）——用 figures-diagram 出 prompt。

### Day 4
- ⚙ 训练持续（x3/x4 应过半 ~125k+）。
- ✍ 写 Introduction + Related Work 初稿（先用 evidence-driven-writing 建 evidence map）。
  - 讲清 motivation：CATANet TAB 的 4 个限制 → DPR 如何解决 → 4 个 contribution。
- 📊 整理路由诊断日志（x_scores / prototype usage / 空槽率 / router scale）为 CSV。

### Day 5
- ⚙ 启动核心消融训练（在 x2 上，最快）：A1/A2/A3 各变体并行排队，每个 finetune 较短 iter。
  - 若 GPU 紧张，消融可用更短 iter（如 50-100k）保证两周内出全。
- ✍ Method 章节自审 + 补全公式、符号表。
- 📊 跑 x2 路由可视化（聚类图，原始顺序 belong_idx；相关 bug 已修复，见 protocol §10.A）。

### Day 6-7（周末，机器全速 + 写作缓冲）
- ⚙ x3/x4 训练接近完成（200k+）；消融变体陆续完成。
- 📊 收集所有已完成模型的 5 基准测试结果，填入主表（区分真实/占位）。
- ✍ 写 Experiments 章节的「数据集与实验设置」小节（可定稿，不依赖最终指标）。
- 里程碑检查：Method + Intro 初稿完成；主表对手数据齐全；x3/x4 收尾中。

---

## 第二周：训练收尾 + 全数据回填 + 实验章节 + 自审定稿

### Day 8
- ⚙ x3/x4 训练完成（250k）；选最优 checkpoint，跑全 5 基准测试导出指标。
- 📊 回填主表 Table I（x2/x3/x4 × 5 基准），去除占位标记，加粗最优。
- ✍ 开始写 Experimental Results 章节主对比分析（基于真实指标）。

### Day 9
- ⚙ 消融全部完成；汇总 A1/A2/A3 结果。
- 📊 制作消融表（A1/A2/A3）+ balance loss 多 seed 方差表。
- ✍ 写消融分析段落（逐 contribution 对应到表，说明增益来源与失败情形）。

### Day 10
- 📊 制作效率对比表 Table II + PSNR-vs-Params/FLOPs 散点图（figures-python）。
- 📊 制作视觉对比图（Urban100/Manga109 难样本，多方法 crop 对齐）。
- ✍ 写 Results 中效率分析 + 可视化分析段落。

### Day 11
- 📊 完成路由可解释性图（聚类图、x_scores 直方图、排序前后邻域一致性）。
- ✍ 写 Discussion（内容路由 vs 空间邻近的取舍、过强路由的风险、泛化边界、限制）。
- ✍ 写 Conclusion + Abstract。

### Day 12
- ✍ 全文连贯性整合：统一符号、术语、图表编号、引用。
- ✍ 两阶段 Review 阶段一：规范合规（字数、结构、引用格式、图表齐全）。
- 📊 校验所有表/图数据来源可追溯，删除全部 mock 占位。

### Day 13
- ✍ 两阶段 Review 阶段二：质量检查（去 AI 化、逻辑、claim 有据、英文流畅）。
- ✍ peer-review 技能自审：找审稿人会质疑的点（公平性、缺失基线、效率口径），补救。
- 📊 补跑任何审稿薄弱点需要的小实验（预留缓冲）。

### Day 14（定稿日）
- ✍ 转 LaTeX（若目标期刊模板已定），生成可编译 .tex + 图表。
- ✍ 终检：摘要-结论一致、贡献-实验一一对应、参考文献完整。
- 产出：可投稿初稿 + 全部源数据/图表归档 + capability-use audit。

---

## 关键风险与缓冲

| 风险 | 缓解 |
|---|---|
| x3/x4 finetune 不及原 CATANet | Day2-3 早发现，调 lr/milestone 或延长 iter；用第二周缓冲 |
| 消融训练挤占 GPU | 消融用短 iter（50-100k）即可体现趋势；x3/x4 优先 |
| 主表对手数据缺尺度 | 只对比有公开数据的尺度，缺失格标注 "-" 不编造 |
| 可视化 bug（聚类图错位） | 已确认源码修复（§10.A）；出图后肉眼核对聚类是否与纹理对应 |
| 两周写不完全文 | Method/Intro/Setting 不依赖最终指标，第一周先写定 |

## 并行原则

- 机器永远在跑：任何时刻 GPU 不空闲（训练优先级 x3/x4 > 消融 > 超参扫描）。
- 不依赖最终指标的写作（Method/Intro/Related Work/Setting/图）第一周做完。
- 依赖最终指标的部分（主表/消融表/Results 分析）随训练完成即时回填。
