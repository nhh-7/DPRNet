# Method-Experiment 可追溯表 (Gate D1)

每个 Introduction 中的 contribution 必须映射到方法模块、实验、表/图、允许的 claim
与证据状态。若某 contribution 无实验/限制说明支撑，不得保留在 Introduction。

| Contribution | Method 模块 | 实验 | 表/图 | 允许的 claim | 证据状态 |
|---|---|---|---|---|---|
| C1 动态语义原型（替代历史中心+EMA） | DPR soft assignment (2.2) | **C1 直接对照（2026-07-06 完成）**：routing_mode=dpr vs ema_center，from-scratch 250k 同协议 3 seed（EMACenterRouter） | **Table C1** (table_c1_dpr_vs_ema.tex) + 聚类图 Fig.8 | 可主张"动态原型在同协议下优于历史中心+EMA"的受控结论（Set5 +0.06dB、Urban100 +0.11dB，且方差更低） | **已完成**：ablfs_c1_dpr(=full) vs c1_emacenter × 3 seed，正文/表已回填 |
| C2 原型生成与确认解耦（refine） | prototype query refine (2.3) | A1 refine on/off（use_prototype_query_refine）；from-scratch 250k × 3 seed | **Table III** | 可主张单开关归因："refine 提升重建（Set5 +0.04/Urban100 +0.05dB），增益大于 seed 间 std" | **已完成**：from-scratch 3-seed mean±std 已回填 |
| C3 置信度感知路由（排序+门控+fallback） | 2.4-2.6 | A2 逐项加法（三 flag）+ 路由诊断（§3.6 sec:routing_analysis）；from-scratch 250k × 3 seed | **Table IV** + x_scores 直方图 Fig.5 | "各组件相互兼容、完整通路(v4)最优且优于 seed std；中间步(v2/v3)非单调，不主张逐项单调增益"；"置信度信号各 block 非退化、可用"（§3.6） | **已完成**：from-scratch 3-seed mean±std 已回填 |
| C4 路由稳定化（路由温度+balance loss） | router scale (2.4) + balance loss (2.7) | A3（from-scratch 250k × 3 seed）+ 路由诊断（§3.6）+ Fig.6 | **Table V** + usage 熵图 Fig.6 | balance loss 使 prototype 使用更均衡（8 block 熵全升 0.623→0.686、active 槽 47.5→52.3）；温度稳定未饱和（τ≈6.06<10）；**"多 seed 方差下降"现已成立**（Set5/Urban100 std 0.05→0.02dB、usage 熵 std 0.008→0.003） | **已完成**：3-seed from-scratch，均衡机制 + 方差下降均已回填 |
| 整体有效性 | 全 DPR | 主对比 | Table I/II + 视觉图 Fig.7 | DPRNet 在轻量设定下达到有竞争力的 PSNR/SSIM 与效率 | 自测数据已全(x2/x3/x4 主表+效率)；对手指标已摘录(B节,8方法)；Table I/II 已拼装(paper/tables/)；**视觉图 Fig.7 已完成**(GT/Bicubic/IMDN/SwinIR-light/SRFormer-light/CATANet/DPRNet/GT crop，纯定性不标指标，续7+续8) |

## 证据状态图例
- 已有：x2/x3/x4 主结果均已产出，统一报告点 x2=net_g_792500 / x3=net_g_250000 /
  x4=net_g_250000（三尺度同口径：全基准均值最优 + 最收敛，不逐数据集挑最优）。
  · x2（crop_border=2）：Set5 38.2906/0.9617, Set14 34.0614/0.9220, B100 32.3809/0.9031,
    Urban100 33.1500/0.9372, Manga109 39.3804/0.9789。
  · x3（crop_border=3）：Set5 34.7629/0.9301, Set14 30.6699/0.8481, B100 29.2890/0.8103,
    Urban100 29.0551/0.8689, Manga109 34.4133/0.9499。
  · x4（crop_border=4）：Set5 32.5896/0.9002, Set14 28.9146/0.7883, B100 27.7648/0.7434,
    Urban100 26.8925/0.8089, Manga109 31.3240/0.9181。
  · 效率（cuda, thop MACs, **统一 256×256 输入**，续5 重测，efficiency_new.md）：Params 601.95K/674.15K/659.71K，
    MACs x2/x3/x4 = 36.19/41.26/52.14 G，时延 183.57/185.88/198.16 ms。详见 data-collection-checklist A3。
    （旧 720×1280 反推数据 126.52/64.28/45.77G 已废弃。x4 52.14G 高于 CATANet 46.8G 但低于 SwinIR-light 60.3G/SRFormer-light 56.5G。）
  注：均为自测数据；baseline 对手指标（B 节）已摘录回填（8 方法三尺度，标注来源 A/B/C/D），
  可与本文自测数据拼装 Table I/II 对比列。
- 已完成（2026-07-06）：from-scratch 250k × 3 seed 新协议消融全部跑完并回填 C1–C4
  （新增 Table C1；Table III/IV/V 改为 mean±std；正文/摘要/讨论/结论已同步）。
- 严禁：在证据状态非"已有/已完成"时，正文用"results show/实验表明"等断言。

## ⭐ 消融升级完成（2026-07-06）：from-scratch 3-seed 新协议已回填

旧的「共享全功能初始化 + 80k + 单 seed」受限口径消融已被 **from-scratch 250k × 3 seed
（3407/42/1234）mean±std** 新协议取代并回填入稿件。核心结果（数据源
`paper/plan/ablation-results-to-fill.md`）：

- **C1（新增 Table C1）**：DPR（=full）vs EMA-center，同协议仅换原型生成机制。
  Set5 32.57±0.02 vs 32.51±0.04 dB；Urban100 26.87±0.02 vs 26.76±0.06 dB。
  动态原型在两基准均更优且方差更低 → C1 从"引用间接动机"升级为**受控直接结论**。
- **C2（Table III）**：refine on/off，Set5 32.57±0.02 vs 32.53±0.03，Urban100 26.87±0.02
  vs 26.82±0.04；增益 > seed std → 可做**单开关归因**。
- **C3（Table IV）**：v1–v4，v4(full) 两基准最优且优于 seed std；v2/v3 中间步非单调 →
  按"组件兼容、完整通路最优、不主张逐项单调增益"写。
- **C4（Table V + Fig.6）**：usage 熵 8 block 全升（0.623→0.686）、active 槽 47.5→52.3、
  τ≈6.06<10；**方差下降现已成立**：Set5/Urban100 PSNR std 0.05→0.02 dB、usage 熵 std
  0.008→0.003 → C4 方差子主张不再"待定"。

下面「消融实验的已知局限」一节为旧受限口径的历史记录，**已被上述新协议取代**，仅作留档。

## 消融实验的已知局限（2026-06-17 起；2026-06-27 按更正数据修订；2026-07-06 已被 from-scratch 3-seed 取代，仅留档）

所有 6 个 x4 消融变体（full / A1_refine_off / A2_v1-v3 / A3_balance_off）均已跑满预算
（A1/A2/A3 各 80k，full 100k），日志在 CATANet/experiments/，解析干净。更正后的 full 行为
Set5 32.5878/0.9001、Urban100 26.8905/0.8086。仍存在一处方法学局限：

- **共享全功能初始化**：全部变体的 pretrain_network_g 均指向 train_CATANet_x4_finetune/net_g_250000.pth，
  而该 checkpoint 是一个全开关、带 balance loss、已收敛 250k 的全功能模型（其 yml 无 switch 键→默认全 True）。
  即便"全关"的 baseline 变体也是从全功能最优权重出发，只退火 80k。这正是 experiment-protocol.md §10 注2
  早已警告的"从全功能权重关 flag 测会失配"情形。
- **更正后的形态**：full 模型在 A1/A2/A3 三组里都是最优（A1 refine 开 vs 关 Set5 +0.038/Urban100 +0.055；
  A3 balance 开 vs 关 Set5 +0.019/Urban100 +0.064；A2 v4 全开为四者最高）。但margin 较小（约 0.04–0.06 dB，
  略高于 full 自身相邻 10k 点 ~0.03dB 抖动），且 A2 中间步（v2/v3）非单调——先降后由完整组合反超。
  **因此仍按"质量中性偏正、组件兼容"写，不主张单个开关的独立 PSNR 增益。**
- **仍可信的机制性证据**：A3 的 balance loss 在 8 个 TAB block 上全部提升归一化 usage 熵
  （block 均值 0.6227→0.6865），这是从同初始化退火也能体现的"原型使用更均衡"效应，可支撑 C4 的均衡机制
  （但非方差）。
- **改写后的口径**：A1/A2/A3 报"质量中性偏正、组件兼容、full 始终不劣且为最优"，但因共享初始化+短预算+单 seed
  不主张单开关独立增益；C1 动态原型 vs 历史中心仍只作动机层面，引 CATANet 原论文间接对比；C4 均衡机制可主张，
  方差下降待多 seed。
- **若日后重训**：正确做法是各变体从开关中性初始化（x2 基座或 from-scratch）训足够 iter，对齐 CATANet
  原论文 x4 from-scratch 250k 口径，方能分离开关的真实增益。本次保留现有数据，按上述受限口径写作。

## 局限的解决方案（2026-07-05 落地）

上节「若日后重训」的建议已落地为可执行协议与代码；旧 finetune 数据保留为受限口径记录，
新数据产出后据此升级各 claim。

- **from-scratch + 多 seed 新协议**：新增 `train_CATANet_x4_ablfs_*_s{seed}.yml`
  （`ablfs`=ABLation-From-Scratch），每变体去掉 pretrain_network_g、训满 250k、
  seed∈{3407,42,1234} 报 mean±std。生成 `options/train/gen_seed_variants.py`，
  运行 `options/train/run_ablfs.sh`（RUN_SET=A3/C1/A2/A1/ALL）。这直接消除"共享全功能初始化"
  失配，使 A1/A2/A3 可做单开关归因，A3 可主张 C4 方差下降。详见 experiment-protocol.md §5.0。
- **C1 EMA-center 直接对照（代码已实现）**：catanet_arch.py 新增 `EMACenterRouter`
  （复用文件内原有 center_iter/ema_inplace/dists_and_buckets，持久 means buffer 跨批次 EMA 更新、
  硬 argmax 排序；输出接口与 DPR 完全一致，IASA/门控/soft-fallback/诊断无需改动）。
  TAB/CATANet 新增 `routing_mode` 开关（'dpr'/'ema_center'），self.dpr 属性名不变，
  故 catanet_model.py 损失聚合与诊断收集零改动。对照 yml：
  `ablfs_c1_dpr_s{seed}` vs `ablfs_c1_emacenter_s{seed}`（from-scratch 同协议同 seed，仅生成机制不同）。
  跑前用 `scripts/smoke_test_ema_router.py` 验证建图/前向/EMA 更新/接口对齐。
  → C1 从"引用 CATANet 间接动机"升级为**受控对照**；新数据达标后可删去正文中"未直接验证"的声明。
- **正文措辞已同步（2026-07-05）**：3_experiments_ablation.tex / main_mdpi_zh.md §3.5「协议注意事项」
  已改写——0.03dB 抖动重新定位为"已用于校准噪声的标尺"，核心证据改为"三组×两基准一致正向"
  （随机波动不会系统性偏向 full），归因于设计要素共同作用而非单开关。
- **状态**：以上均为**代码/配置就绪、待在训练机跑新实验**；py_compile 通过，本机无 torch 未做前向。
  实验产出后：回填 Table III/IV/V 为 mean±std、新增 C1 对照表、更新上表各 claim 与证据状态。
