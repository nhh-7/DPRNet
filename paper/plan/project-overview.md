# 项目概览

## 基本信息

- 论文类型：SCI 期刊投稿（Neurocomputing / Pattern Recognition / TNNLS 一档）
- 学科领域：计算机视觉 / 图像超分辨率 / 轻量级网络
- 工作代号：DPRNet（基于 CATANet，将 TAB 路由替换为 Dynamic Prototype Routing）
- 暂定题目：Dynamic Prototype Routing for Lightweight Image Super-Resolution
- 创建时间：2026-06-06
- 输出格式：先用 CVPR 2026 官方 LaTeX 模板（latex-templates/latex-template/，
  含 cvpr.sty / preamble.tex / ieeenat_fullname.bst / sec/ 分章节 / GitHub 自动编译）
  撰写正文；确定具体期刊后再迁移到对应期刊模板（Neurocomputing/PR → Elsevier
  elsarticle；TNNLS → IEEEtran）。注意：CVPR 为会议双栏模板，最终投期刊需换格式。
- 语言：英文
- 当前阶段：x2 模型训练完成，进入 x3/x4 训练 + 论文撰写并行阶段

## 研究信息

### 研究背景

CATANet 原始 TAB 模块基于固定中心迭代（means + center_iter + EMA）与硬标签
排序完成内容路由，存在四个限制：(1) 路由中心依赖历史缓冲区而非当前输入；
(2) token 归属仅用硬标签 argmax，不保留置信度；(3) 排序只按类别聚集，缺少
组内纯度约束；(4) 原型生成与确认未解耦，可解释性弱。

### 研究目的

提出 Dynamic Prototype Routing (DPR)：由当前输入 token 通过 soft assignment
动态生成语义原型，经 prototype query refine 稳定槽位，再做 token-to-prototype
反向匹配，按 `belong_idx + 0.5*(1 - x_scores)` 联合排序。在保持轻量与接口兼容
前提下，提升内容路由质量与重建精度。

### 核心创新点（论文 contribution）

1. C1 动态语义原型：用当前输入自适应生成原型，替代历史中心 + EMA。
2. C2 原型生成与确认解耦：内容聚合得 P_content 后用 learnable query refine。
3. C3 置信度感知路由：显式保留 x_scores，参与排序 / IASA 门控 / soft fallback。
4. C4 路由稳定化：可学习路由温度 + prototype 使用率均衡损失，降低 seed 方差。

### 研究方法

- 基础框架：BasicSR + PyTorch，CATANet 主干。
- 训练：x2 已 scratch 训练至 800k；x3/x4 从 x2 权重 finetune（250k iter）。
- 评估：Set5/Set14/B100/Urban100/Manga109，PSNR/SSIM（Y 通道）。
- 效率：Params / FLOPs / 推理时延，对比同档轻量 SR 方法。
- 消融：核心 3 项（C1 动态原型、C3 置信度排序+门控、C4 balance loss），见实验协议。

## 资源与约束

- GPU：2-4 张可并行（x3、x4 可同时跑）。
- 时间：两周内完成训练 + 数据收集 + 初稿。
- 训练策略：finetune（已确认）。

## 章节结构

见 outline.md。采用 CS/工程 SCI 标准结构：Introduction(含 related work) /
Method / Experiments(含 dataset & setting) / Discussion / Conclusion。

## 写作规范

- 语言：英文，客观第三人称（this paper / the proposed method）。
- 去 AI 化：禁机械过渡词、空壳强调句。
- 引用：绝不编造，必须可追溯（CrossRef / arXiv / 官方出处）。
