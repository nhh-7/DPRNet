# 论文章节大纲 (Outline)

题目（暂定）：Dynamic Prototype Routing for Lightweight Image Super-Resolution
结构：CS/工程 SCI 标准 body（Related Work 融入 Introduction）。

## Abstract（150-250 词）
问题 → 方法（DPR 一句话）→ 4 个设计 → 主结果（5 基准 + 效率）→ 一句意义。

## 1. Introduction（含 Related Work）
- 1.1 轻量 SR 背景与需求（边缘设备、参数/FLOPs 约束）。
- 1.2 相关工作融入：CNN 轻量 SR、Transformer 轻量 SR、内容/聚类路由（CATANet）。
- 1.3 研究空白：CATANet TAB 的 4 个限制（历史中心、硬标签、无纯度排序、未解耦）。
- 1.4 本文方法概述 + 4 个 contribution（C1 动态原型 / C2 解耦 refine /
  C3 置信度感知 / C4 路由稳定化）。
- 必须先建 evidence map（evidence-driven-writing）。

## 2. Method
- 2.1 整体框架（CATANet 主干 + DPR 替换 TAB，接口 B×C×H×W 不变）。
- 2.2 动态语义原型生成（soft assignment from current tokens）。
- 2.3 原型确认：prototype query refine（learnable query）。
- 2.4 token-to-prototype 反向匹配 + 可学习路由温度（router scale）。
- 2.5 置信度感知排序（belong_idx + 0.5(1-x_scores)）。
- 2.6 IASA 聚合 + confidence-aware global gate + soft prototype fallback。
- 2.7 prototype 使用率均衡损失（balance loss）。
- 配图：DPR 数据流图 + 整体网络图（figures-diagram）。

## 3. Experiments
- 3.1 数据集与实验设置（DIV2K 训练，5 基准测试，退化/指标/超参/硬件）。
- 3.2 与 SOTA 轻量方法对比（Table I：x2/x3/x4 × 5 基准 PSNR/SSIM）。
- 3.3 效率对比（Table II：Params/FLOPs/时延；PSNR-vs-Params 图）。
- 3.4 视觉对比（Urban100/Manga109 难样本）。
- 3.5 消融实验（A1 动态原型 / A2 置信度感知 / A3 balance loss）。
- 3.6 路由可解释性分析（聚类图、x_scores 直方图、邻域一致性）。

## 4. Discussion
- 内容路由 vs 空间邻近的适用边界；过强路由对 B100 的风险；
  prototype 数量取舍；泛化与限制。

## 5. Conclusion
- 总结贡献与结果，点明可迁移性（DPR 作为通用内容路由模块），未来工作。

## References
- 真实可追溯，CrossRef/arXiv/官方出处。

---

## 贡献-章节-实验映射（速查，详见 traceability 表）

| 贡献 | Method 节 | 实验 | 表/图 |
|---|---|---|---|
| C1 动态原型 | 2.2 | A1（refine 实测 + 引用 CATANet 间接对比） | 消融表 + 聚类图 |
| C2 解耦 refine | 2.3 | A1 refine on/off | 消融表 |
| C3 置信度感知 | 2.4-2.6 | A2 逐项加法（三 flag） | 消融表 + x_scores 直方图 |
| C4 路由稳定化 | 2.7 | A3 | 方差表 + usage 分布 |
| 整体有效性 | 全 | 主对比 | Table I/II + 视觉图 |
