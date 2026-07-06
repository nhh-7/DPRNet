# 实验协议 (Experiment Protocol)

本文件锁定投稿所需的全部实验。所有论文 claim 必须映射到此处某个实验或限制说明。
对应技能 Gate：D0 协议锁定 → D1 可追溯 → D2 表图数据契约 → D3 结果 → D4 去污 → D5 自审。

## 1. 数据集与划分

| 用途 | 数据集 | 说明 |
|---|---|---|
| 训练 | DIV2K (800 train) | 标准 SR 训练集；可选加 Flickr2K (DF2K) 增强 |
| 验证(训练中) | Set5, Set14 | 训练时 val_freq=2500 监控 |
| 测试(最终) | Set5, Set14, B100, Urban100, Manga109 | 全部 5 个基准，PSNR/SSIM (Y) |
| 退化方式 | Bicubic (LRBI) | x2/x3/x4 三个尺度 |

- crop_border：x2=2, x3=3, x4=4；test_y_channel=true（与 SR 惯例一致）。
- 训练 patch：x2 gt_size=128 等（按现有 yml），x3 gt_size=192，x4 gt_size=256。
- 随机种子：manual_seed=3407（主结果）；from-scratch 消融与 C4 方差分析用 seed∈{3407,42,1234}
  （见 §5.0 新协议）。

## 2. 基线方法 (Baselines)

主表对比同档轻量 SR 方法（参数量量级相近，公平）：

| 类别 | 代表方法 | 取数来源 |
|---|---|---|
| CNN 轻量 | CARN, IMDN, RFDN, RLFN | 原论文公开表格 |
| Transformer 轻量 | SwinIR-light, ELAN-light, SRFormer-light | 原论文公开表格 |
| 直接对手 | CATANet (原始 TAB) | 复现 / 原论文 |
| 本文 | DPRNet (DPR 替换 TAB) | 自测 |

公平性说明：所有对比方法均为轻量 SR 设定（参数 < ~1M 量级 / 同 FLOPs 区间），
同 DIV2K 训练、同 5 基准 Y 通道评测。CATANet 为最直接对照（仅路由模块不同）。

注：从他人论文摘录的指标必须标注来源；自测指标用自己跑的 checkpoint。

## 3. 评价指标

- 重建质量：PSNR (dB)、SSIM，Y 通道，对应 crop_border。
- 效率：
  - Params (K/M)
  - FLOPs / Multi-Adds（固定输出尺寸，如 1280x720 或惯例设定，需统一说明）
  - 推理时延（ms，单 GPU，固定输入尺寸，warmup 后多次平均）
- 路由诊断（消融与可解释性，非主指标）：
  - x_scores mean/std/min/max（每个 TAB block）
  - prototype usage 分布、空槽比例、熵
  - router scale 值

## 4. 主对比实验 (Main Comparison)

- Table I：x2/x3/x4 × 5 基准 PSNR/SSIM，DPRNet vs 上述基线。本文结果加粗/次优下划线。
- Table II：效率对比（Params / FLOPs / 时延 / Set5 或 Urban100 PSNR）。
- Figure：视觉对比图（Urban100/Manga109 难样本，含 GT/Bicubic/对手/本文 crop）。
- Figure：PSNR-vs-Params 或 PSNR-vs-FLOPs 散点（突出本文 Pareto 优势）。

期望结果形态（来自优化方案文档）：Urban100/Manga109 明显提升，Set5/Set14 持平，
B100 不降或微升。若仅 Urban100 升而 B100 明显降，说明内容路由过强，需调 gate。

## 5. 消融实验 (Ablation) — 核心 3 组 + C1 直接对照

全部在 x4 上做（对齐 CATANet 原论文：其消融均在 scale=4 进行；且 x4 退化最重，
DPR 内容路由在高频纹理上的收益最易显现）。

### 5.0 协议演进（2026-07-05 修订）

**旧协议（finetune，已有数据，受限口径）**：统一从 x4 net_g_250000 finetune、相同 iter
（80k），每 10k 验证/存档，报 Set5+Urban100。局限：所有变体共享同一全功能收敛权重，
"关某开关"是从已适配该开关的表示上退火，参数失配，**无法归因单开关增益、无方差**
（详见 method-experiment-traceability.md「消融实验的已知局限」）。对应 yml：
`options/train/train_CATANet_x4_abl_*.yml`。此数据保留，正文按"质量中性偏正、组件兼容"
写，不主张单开关独立增益。

**新协议（from-scratch + 多 seed，支撑因果归因）**：为分离每个开关的真实贡献并给出
显著性，每个变体**从头训练**满 250k（去掉共享初始化），并在 **≥3 个 seed（3407/42/1234）**
上重复，报 mean±std。对齐 CATANet 原论文 x4 from-scratch 250k 口径。对应 yml：
`options/train/train_CATANet_x4_ablfs_*_s{seed}.yml`（`ablfs` = ABLation-From-Scratch）。
生成与运行：
```bash
python options/train/gen_seed_variants.py          # 由 _s3407 基准克隆出 _s42/_s1234
RUN_SET=A3 bash options/train/run_ablfs.sh         # 优先 A3（C4 方差子主张必须多 seed）
RUN_SET=C1 bash options/train/run_ablfs.sh         # C1 EMA 对照
RUN_SET=ALL bash options/train/run_ablfs.sh        # 全量 6 变体 × 3 seed = 18 次 250k
```
算力代价：全量 18 次 from-scratch x4 训练；受限时按 A3 → C1 → A2 → A1 优先级分批。
每个变体训完必须：测 5 基准 → 存指标；导出逐 block 路由诊断 CSV（tee 到日志再解析）。

### C1 动态原型 vs 历史中心+EMA (验证 C1) — 直接对照（2026-07-05 升级）
说明：EMA-center 分支**已在本文代码库实现**（catanet_arch.py `EMACenterRouter`，
复用原 center_iter/ema_inplace，输出接口与 DPR 完全一致，IASA/门控/诊断无需改动），
由 `network_g.routing_mode` 切换（`dpr` / `ema_center`）。两臂 from-scratch 同协议同
seed，仅原型生成机制不同，从而把 C1 从"引用间接动机"升级为**受控结论**。
| 变体 | 配置（yml network_g） | 期望 |
|---|---|---|
| DPR 动态原型 | routing_mode=dpr（默认） | PSNR/路由质量 ≥ EMA 臂 |
| EMA 历史中心 | routing_mode=ema_center | 参照（原 CATANet TAB 路由机制） |
- 先跑 `python scripts/smoke_test_ema_router.py` 验证两模式建图/前向/EMA 更新/接口对齐。
- 备注：ema_center 模式下 DPR 专属开关（refine/conf_sort/score_gate/soft_fallback/balance）
  按构造失效，两臂公平性以"仅生成机制不同"为准，参数量差异在正文说明。

### A1 DPR 内部设计：prototype query refine (验证 C2)
| 变体 | 配置（yml network_g） | 期望 |
|---|---|---|
| baseline | use_prototype_query_refine=false | 参照 |
| +refine | use_prototype_query_refine=true（默认） | PSNR/路由质量 ≥ baseline |

### A2 置信度感知路由 (验证 C3) — 逐项加法（三 flag 已实现）
| 变体 | 配置（yml network_g flag） | 期望 |
|---|---|---|
| 纯硬排序 | use_conf_sort=F, use_iasa_score_gate=F, use_soft_fallback=F | 参照 |
| +置信度排序 | use_conf_sort=T（其余 F） | Urban100 升 |
| +IASA 置信度门控 | +use_iasa_score_gate=T | 进一步升 |
| +soft fallback | +use_soft_fallback=T（全开=完整模型） | Urban100/Manga 升 |

### A3 prototype balance loss (验证 C4)
| 变体 | route_balance_weight | 报告 |
|---|---|---|
| 无 | 0.0 | usage 分布 + 多 seed 方差 |
| 弱 | 默认（[5e-4]×6,[7e-4]×2） | usage 更均衡 + 方差下降 |
- C4「方差下降」子主张仅在新协议多 seed 下成立；旧 finetune 单 seed 只报均衡机制。

可选超参研究（时间允许）：num_prototypes ∈ {8,16,32}，router_dim 扫描。

## 6. 可解释性 / 路由分析 (支撑 C1/C3)

- 路由聚类图可视化（原始顺序 belong_idx → reshape H×W，上色）。
- 排序前后 token 邻域语义一致性对比图。
- x_scores 直方图（加 router scale 前后对比，证明区分度提升）。
- 注意：必须用原始顺序 belong_idx（last_routing_map）。聚类图 self.lq 删除顺序与
  原始/排序顺序 bug 经核实已在源码修复（见 §10.A），无需再改。

## 7. 硬件与软件环境（论文必报）

- GPU 型号、数量、显存；CUDA / PyTorch / BasicSR 版本。
- batch size、optimizer(Adam, betas)、lr、scheduler(MultiStepLR milestones)、total_iter。
- 训练耗时（GPU·hours）。

## 8. Method-Experiment 可追溯表 (Gate D1)

见 plan/review/method-experiment-traceability.md（每个 contribution → 模块 →
实验 → 表/图 → 允许的 claim → 证据状态）。

## 9. Mock 数据边界

真实指标产出前，表格可用占位但必须：文件名 `mock_` 前缀、表注
`PLANNING DATA - replace before submission`、正文保留 `[待真实实验替换]`。
绝不把占位值写成 "results show" / "实验结果表明"。

## 10. 代码改动总账 (Code-Change Ledger) — 防回退/重训

本节集中列出"为产出某项数据所必须做的代码/配置修改"，避免散落各处导致漏改重训。
分三类：A 已就绪（已核实，勿重复改） / B 训练前必改（漏则重训） / C 各实验按需开关。
状态：[x]已完成 / [ ]待办。证据为已核实的源码行号。

### A. 已就绪（已核实代码实现，无需再改）
- [x] 置信度感知排序 belong_idx + 0.5*(1-score)：catanet_arch.py:213（支撑 A2 消融）
- [x] balance loss + 归一化熵正则：catanet_arch.py:206-208（支撑 A3 消融）
- [x] 原始顺序 + 排序两套 routing_map 均已缓存：catanet_arch.py:265-266
      （last_routing_map / last_sorted_routing_map，支撑可解释性图）
- [x] cluster map 用原始顺序 belong_idx、且先存 lq_shape 再 del self.lq：
      catanet_model.py:249-276（聚类图错位 bug 已修，无需再动）
- 说明：原"优化方案 §8 待修 bug"经核实绝大多数已修复，data-checklist F 节据此更新。

### B. 训练前必改（漏改将导致 x3/x4 或消融重训）
- [x] x3/x4 finetune yml 补齐路由超参，与 x2 完全对齐（三尺度同口径）：
      network_g.router_scale_init=6.0 / max_router_logit_scale=10.0 /
      route_balance_weight=[0.0005x6,0.0007x2]，train.router_scale_lr_mult=0.1
      （已写入 train_CATANet_x3_finetune.yml / train_CATANet_x4_finetune.yml）
- [x] x3/x4 pretrain_network_g 指向 net_g_792500.pth、strict_load_g=false
- [ ] 启动后首个 val 点核对：x3/x4 PSNR 应从高值起步（finetune 生效），
      若从极低值爬升说明权重未加载或超参不一致，立即停训排查（防训完才发现）

### C. 各消融变体需要的开关（开始对应消融前改，改完即训）
- 核查结论（已读 catanet_arch.py 全部路由/原型/IASA 实现，2026-06-06）：

- [x] C1 动态原型 vs 历史中心+EMA：【2026-07-05 升级：降级方案 (b) 已被直接对照取代】
      EMA-center 对照分支已在本文代码库实现，不再仅靠引用间接对比。
      → ✅ catanet_arch.py 新增 `EMACenterRouter`（复用文件内原有 center_iter/ema_inplace/
        dists_and_buckets，输出 (sorted_x, idx_last, prototypes, route_info) 接口与 DPR 完全
        一致；持久 means buffer 做跨批次 EMA 更新、硬 argmax 标签排序）。
        TAB/CATANet 新增 `routing_mode` 开关（'dpr'/'ema_center'），`self.dpr` 属性名在两
        模式下不变，故 catanet_model.py 的损失聚合(aux_loss)与诊断收集(tab.dpr.router_logit_
        scale 等)无需改动。py_compile 通过；前向验证用 scripts/smoke_test_ema_router.py（训练机）。
        对照 yml：train_CATANet_x4_ablfs_c1_dpr_s{seed}.yml vs _c1_emacenter_s{seed}.yml
        （from-scratch 同协议同 seed，仅 routing_mode 不同）。
- [x] A1 DPR 内部设计（验证 C2）：use_prototype_query_refine 已贯通 CATANet→TAB→DPR
      （catanet_arch.py：CATANet __init__/存/传，TAB __init__/传，DPR L139）。
      默认 True 严格等价原行为，可由 yml 配置。
      消融 yml：finetune 口径 train_CATANet_x4_abl_A1_refine_off.yml（旧，受限）；
      from-scratch 口径 train_CATANet_x4_ablfs_A1_refine_off_s{seed}.yml（新，可归因）。
- [x] A2 逐项加法：✅ 三个独立 flag 已实现（catanet_arch.py 2026-06-06）：
      · use_conf_sort（DPR）：关闭→sort_key=belong_idx（去掉 +0.5(1-score)）
      · use_iasa_score_gate（TAB）：关闭→IASA 传 sorted_scores=None（退化分支）
      · use_soft_fallback（TAB）：关闭→y=hard_y（去掉 soft 项）
      · 三 flag 默认 True，全开严格等价于改动前行为；已贯通 CATANet→TAB→DPR，可由 yml 配置。
      逐项加法 yml 序列：全关 → +conf_sort → +score_gate → +soft_fallback（全开=完整模型）。
- [x] A3 balance loss：✅ route_balance_weight 设 0.0（无）vs 当前值（弱），多 seed
      （改 yml 即可，无需改码）
- 注1：C 类每加一个开关都要保证"关闭时严格等价于原行为"，否则消融不干净。
- 注2（2026-07-05 修订）：因果归因用 **from-scratch 多 seed** 新协议（ablfs_*_s{seed}.yml）：
      各变体去掉 pretrain_network_g、训满 250k、seed∈{3407,42,1234}，报 mean±std。
      旧 finetune 短 iter 数据（abl_*.yml）保留但仅作"组件兼容/质量中性偏正"口径，
      不能从全功能 x3/x4 权重直接关 flag 测（参数已适配全功能，关掉会失配，结论不可信）。

### 效率脚本（A3 数据 / Table II）
- [x] Params / FLOPs / 时延已在训练机测完（efficiency_new.md）：
      scripts/measure_efficiency.py 支持两套口径——口径 A（固定 LR 输入 --lr-h/--lr-w，**推荐对齐 CATANet 论文 Tab.6 的 256×256**）、
      口径 B（固定 HR 输出反推 LR）。**续5 已用统一口径 A（256×256）重测三尺度**，
      DPRNet x4=52.14G / x3=41.26G / x2=36.19G，与 CATANet 46.8G / SwinIR-light 60.3G / SRFormer-light 56.5G 直接可比。
      旧 1280×720 反推数据（45.77G 等）已废弃。
