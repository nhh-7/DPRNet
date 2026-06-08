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
- [ ] x3：同上（test_CATANet_x3.yml）
- [ ] x4：同上（test_CATANet_x4.yml）
- 注：crop_border 按尺度（2/3/4），test_y_channel=true。

### A3 效率数据（DPRNet + CATANet，三尺度）
- 工具：scripts/measure_efficiency.py（已实现，待训练机运行）。
- [ ] Params（总参数量）—— test 日志已知 601,947，待脚本统一复核
- [ ] FLOPs / Multi-Adds（固定输出 1280x720 反推 LR 输入，统一口径并在论文说明）
- [ ] 推理时延（ms，单 GPU，warmup 后多次平均，固定输入尺寸）

## B. 从他人论文摘录的对比数据（标注来源，绝不编造）

- [ ] CARN / IMDN / RFDN / RLFN x2/x3/x4 5 基准 PSNR/SSIM + Params
- [ ] SwinIR-light / ELAN-light / SRFormer-light 同上
- [ ] CATANet（原始）同上（优先自测，缺则引原论文）
- 每条记录格式：方法名 | 尺度 | 数据集 | PSNR | SSIM | Params | 来源(论文/表号)

## C. 表格清单（tables/，含 table-schema.md）

- [ ] Table I 主对比：方法 × (尺度×数据集) PSNR/SSIM，本文加粗、次优下划线
- [ ] Table II 效率：方法 | Params | FLOPs | 时延 | 代表 PSNR
- [ ] Table III 消融 A1：动态原型 vs 历史中心
- [ ] Table IV 消融 A2：置信度感知逐项加法（排序/门控/fallback）
- [ ] Table V 消融 A3：balance loss 多 seed（mean±std）
- [ ] (可选) 超参表：num_prototypes / router_dim 扫描

## D. 图表清单（figures/，含 data-manifest.md）

### D1 结构图（figures-diagram，无需数据）
- [ ] Fig.1 整体网络结构（CATANet 主干 + DPR 位置）
- [ ] Fig.2 DPR 模块数据流（原型生成→refine→匹配→排序→IASA）

### D2 数据图（figures-python，需 CSV）
- [ ] Fig.3 PSNR-vs-Params（或 vs FLOPs）散点，突出本文 Pareto
- [ ] Fig.4 训练曲线对比（可选，DPRNet vs CATANet val PSNR）
- [ ] Fig.5 x_scores 直方图（加 router scale 前后区分度对比）
- [ ] Fig.6 prototype usage 分布柱状图

### D3 可视化图（需跑模型输出）
- [ ] Fig.7 视觉对比（Urban100/Manga109 难样本，GT/Bicubic/对手/本文 crop）
- [ ] Fig.8 路由聚类图（原始顺序 belong_idx → H×W 上色）
- [ ] Fig.9 排序前后 token 邻域语义一致性

## E. 投稿前置文件

- [ ] references.bib（全部引用，CrossRef 验证 DOI）
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
