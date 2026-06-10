# Method-Experiment 可追溯表 (Gate D1)

每个 Introduction 中的 contribution 必须映射到方法模块、实验、表/图、允许的 claim
与证据状态。若某 contribution 无实验/限制说明支撑，不得保留在 Introduction。

| Contribution | Method 模块 | 实验 | 表/图 | 允许的 claim | 证据状态 |
|---|---|---|---|---|---|
| C1 动态语义原型（替代历史中心+EMA） | DPR soft assignment (2.2) | A1（降级方案 b）：本文实测 refine 开关消融；动态原型 vs 历史中心改用引用 CATANet 原论文间接对比 | Table III + 聚类图 Fig.8 | 动态原型贴合当前输入（动机层面）；refine 提升路由质量（实测） | 待训练 |
| C2 原型生成与确认解耦（refine） | prototype query refine (2.3) | A1 refine on/off（use_prototype_query_refine） | Table III | refine 提升槽位稳定性与路由质量 | 待训练 |
| C3 置信度感知路由（排序+门控+fallback） | 2.4-2.6 | A2 逐项加法（三 flag 已实现） | Table IV + x_scores 直方图 Fig.5 | 置信度信息提升内容路由质量，Urban100/Manga109 受益 | 待训练 |
| C4 路由稳定化（路由温度+balance loss） | router scale (2.4) + balance loss (2.7) | A3 多 seed | Table V + usage 图 Fig.6 | balance loss 使 prototype 使用更均衡、多 seed 方差下降 | 待训练 |
| 整体有效性 | 全 DPR | 主对比 | Table I/II + 视觉图 Fig.7 | DPRNet 在轻量设定下达到有竞争力的 PSNR/SSIM 与效率 | 自测数据已全(x2/x3/x4 主表+效率)；对手指标已摘录(B节,8方法)；Table I/II 已拼装(paper/tables/，Table II 对手 FLOPs/时延待补)；待视觉图 Fig.7 |

## 证据状态图例
- 已有：x2/x3/x4 主结果均已产出，统一报告点 x2=net_g_792500 / x3=net_g_250000 /
  x4=net_g_250000（三尺度同口径：全基准均值最优 + 最收敛，不逐数据集挑最优）。
  · x2（crop_border=2）：Set5 38.2706/0.9617, Set14 33.9614/0.9210, B100 32.3509/0.9020,
    Urban100 33.0500/0.9362, Manga109 39.3804/0.9787。
  · x3（crop_border=3）：Set5 34.7129/0.9298, Set14 30.6599/0.8481, B100 29.2890/0.8103,
    Urban100 28.9551/0.8675, Manga109 34.4133/0.9499。
  · x4（crop_border=4）：Set5 32.5826/0.8998, Set14 28.8846/0.7880, B100 27.7648/0.7434,
    Urban100 26.8225/0.8065, Manga109 31.2540/0.9181。
  · 效率（cuda, thop MACs, 固定 HR≈720×1280）：Params 601.95K/674.15K/659.71K，
    MACs 126.52/64.28/45.77 G，时延 712.16/285.39/196.77 ms。详见 data-collection-checklist A3。
  注：均为自测数据；baseline 对手指标（B 节）已摘录回填（8 方法三尺度，标注来源 A/B/C/D），
  可与本文自测数据拼装 Table I/II 对比列。
- 待训练：消融变体（A1/A2/A3）训练完成后回填 C1–C4。
- 严禁：在证据状态非"已有/已完成"时，正文用"results show/实验表明"等断言。
