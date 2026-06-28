# Method-Experiment 可追溯表 (Gate D1)

每个 Introduction 中的 contribution 必须映射到方法模块、实验、表/图、允许的 claim
与证据状态。若某 contribution 无实验/限制说明支撑，不得保留在 Introduction。

| Contribution | Method 模块 | 实验 | 表/图 | 允许的 claim | 证据状态 |
|---|---|---|---|---|---|
| C1 动态语义原型（替代历史中心+EMA） | DPR soft assignment (2.2) | A1（降级方案 b）：本文实测 refine 开关消融；动态原型 vs 历史中心改用引用 CATANet 原论文间接对比 | Table III + 聚类图 Fig.8 | 仅动机层面（动态原型贴合当前输入）；不得主张 PSNR 增益 | 已训练（受限，见下） |
| C2 原型生成与确认解耦（refine） | prototype query refine (2.3) | A1 refine on/off（use_prototype_query_refine） | Table III | "refine 不损质量、保持槽位稳定，且更正数据下略优（Set5 +0.04/Urban100 +0.05dB）"；因共享初始化+短预算不主张单开关独立增益 | 已训练（受限） |
| C3 置信度感知路由（排序+门控+fallback） | 2.4-2.6 | A2 逐项加法（三 flag 已实现）+ 路由诊断（§3.6 sec:routing_analysis） | Table IV + x_scores 直方图 Fig.5（待推理出图） | "各组件相互兼容、质量中性偏正（v4 全开为四者最高）"（A2）；"置信度信号在各 block 非退化、可用"（§3.6 已有日志统计支撑）；中间步非单调，不主张逐项 PSNR 单调增益 | 已训练（受限）；§3.6 机制分析已写 |
| C4 路由稳定化（路由温度+balance loss） | router scale (2.4) + balance loss (2.7) | A3 单 seed（多 seed 待补）+ 路由诊断（§3.6） | Table V + usage 熵图 Fig.6（已出，fig6_usage_entropy.tex） | balance loss 使 prototype 使用更均衡（8 block 熵全升 0.6227→0.6865、active 槽 47.5→52.3）；温度稳定未饱和（τ≈6.06<10）；"多 seed 方差下降"待多 seed 重训，暂不得主张 | 部分已训练（均衡/温度机制已有，§3.6+Fig.6；方差待补） |
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
- 待训练：消融变体（A1/A2/A3）训练完成后回填 C1–C4。
- 严禁：在证据状态非"已有/已完成"时，正文用"results show/实验表明"等断言。

## 消融实验的已知局限（2026-06-17 起；2026-06-27 按更正数据修订）

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
