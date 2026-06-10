# Figure Specifications — DPRNet (Fig.1 / Fig.2)

> 结构图规格（figures-diagram prompt）。供绘图（draw.io / TikZ / PowerPoint）逐元素落图用。
> 严格对应 paper/sections/2_method.tex 的公式与数据流；任何符号/箭头都能在 method 中找到出处。
> 约定：实线箭头=张量数据流；虚线箭头=旁路/门控调制；圆角矩形=模块；圆=逐元素运算。

---

## Fig.1 — 整体网络结构（Overall architecture，对应 method §Overview，Eq.1–3）

**标题建议**：`Overall architecture of DPRNet. The contribution is confined to the DPR
module inside each TAB; the rest of the CATANet backbone is unchanged.`

**横向主流水（从左到右）**：
1. `I_LR (3×H×W)` —— 输入图块。
2. `Conv 3×3` —— 浅层特征，输出 `F0 (C=40, H×W)`。
3. `Deep feature extractor`：`L=8` 个相同 block 串联（画 3 个 + 省略号 `…` + 第 8 个）。
   - 每个 block 内部（画一个放大子框，标 "Block l"）：
     - 顺序：`TAB_l → LRSA_l → Conv 3×3`，外加残差 `F_{l-1} ⊕`（对应 Eq.1）。
     - 在 `TAB_l` 上做高亮（红框/底色），标注 "contains DPR (Fig.2)"，并用一条引线指向 Fig.2。
4. 全局残差：`F0` 与 `F_L` 相加（圆 `⊕`），对应 Eq.2 的 `F0 + F_L`。
5. `Up (PixelShuffle ×s)` —— 上采样重建分支。
6. 旁路：`I_LR → Bilinear ↑s`（虚线），与上采样输出相加（圆 `⊕`）得 `I_SR`（Eq.2）。
7. `I_SR (3×sH×sW)` —— 输出。

**TAB 内部展开（Fig.1 右侧或底部子图，对应 Eq.3 的 TAB 数据流）**：
`X (B×N×C, N=HW) → LN → DPR → IASA → Conv 1×1 → ConvFFN`，
其中 `Conv1×1` 与 `ConvFFN` 各带残差连接。把 "DPR" 框高亮，作为 Fig.2 的入口。

**标注要点**：
- 标 `C=40`、`L=8`、`s∈{2,3,4}`、`N=HW`。
- 文字注明 "interface R^{C×H×W}→R^{C×H×W} unchanged"（强调 drop-in 替换）。

**配色**：backbone（Conv/Up/LRSA）用中性灰；DPR/TAB 用强调色（如蓝），突出贡献位置。

---

## Fig.2 — DPR 模块数据流（DPR pipeline，对应 method §DPR + §IASA，Eq.4–13）

**标题建议**：`Dynamic Prototype Routing (DPR). Tokens generate dynamic prototypes,
confirm them by query refinement, are matched back with a learnable temperature,
and are ordered by a confidence-aware key. The aggregator (IASA) is
confidence-modulated.`

**输入**：`X (B×N×C)`（来自 TAB 内 LN 之后）。

**Stage (1) Dynamic prototype generation（Eq.4–5）**：
- `X → E = GELU(Linear(LN(X)))`（token embedding，标 `E (B×N×C)`）。
- `E → Linear W_a → softmax over M slots → A (B×N×M)`（Eq.4，软分配）。
- 原型聚合：`P^c_m = (Σ_n A_{n,m} X_n)/(Σ_n A_{n,m}+ε)`（Eq.5），后接 `LN + ℓ2 normalize`，
  输出 `P^c (B×M×C)`。标注 "from current tokens, no EMA buffer (C1)"。

**Stage (2) Prototype confirmation by query refine（Eq.6–7，对应 C2）**：
- 可学习 `query bank Q (M×C)`；seed `P̃ = P^c + Q`。
- 单层 cross-attention：`Q←P̃W_q`，`K←E W_k`，`V←X W_v`，得 `P^r`（Eq.6）。
- 门控融合：`γ = σ(g_r)`（虚线调制），`P = normalize(LN(P^c + γ·P^r))`（Eq.7）。
- 在 `γ` 旁标注开关：`use_prototype_query_refine=false ⇒ P=P^c (ablation A1)`。

**Stage (3) Confidence-aware matching（Eq.8–9，对应 C3 一部分）**：
- 共享 router 空间归一化：`ê_n = normalize(E_n W_t)`，`p̂_m = normalize(P_m W_p)`。
- 可学习温度：`τ = min(e^θ, τ_max)`；`S = softmax(τ · Ê P̂^T) (B×N×M)`（Eq.8）。
- 读出：`x^s_n = max_m S_{n,m}`（置信度），`b_n = argmax_m S_{n,m}`（硬标签）（Eq.9）。
- 标注 "explicitly retain confidence x^s (C3)"，"learnable temperature (C4)"。

**Stage (4) Confidence-aware ordering（Eq.10，对应 C3）**：
- 排序键：`k_n = b_n + 0.5(1 - x^s_n)`，`π = argsort(k_n)`（Eq.10）。
- 用一条小示意：原始 token 序 → 按 `k` 排序后 token 序（同簇连续，簇内高置信在前）。
- 输出 `X^π, b^π, x^{s,π}`，并记录逆置换 `π^{-1}`（scatter-back 用）。
- 标注开关：`use_conf_sort=false ⇒ k_n=b_n (ablation A2)`。
- DPR 输出汇总：`X^π, P, S, x^s` → 送入 IASA。

**Stage (5) IASA aggregation（Eq.11–12，对应 C3）**：
- 在 `X^π` 上：
  - `intra-group local attention`（组大小 g，overlapping）→ `O_loc`。
  - `global branch`：attend to 投影后的 prototypes `P`（K/V）→ `O_glb`。
- 置信度门控合并：`O = O_loc + β·x^{s,π}·O_glb`，`β = σ(g_g)`（Eq.11，虚线调制）。
  - 标注开关：`use_iasa_score_gate=false ⇒ O = O_loc + β·O_glb (A2)`。
- soft prototype fallback：`X̄ = S P`，`Y = O + α(1 - x^s)(X̄ W_s)`，`α = σ(g_s)`（Eq.12）。
  - 标注开关：`use_soft_fallback=false ⇒ Y = O (A2)`。
- scatter-back via `π^{-1}` → 投影 → 送回 TAB 的 `Conv1×1 + ConvFFN`。

**Stage (6) Balance loss（Eq.13，对应 C4，画为旁路监督）**：
- 由 `S` 计算 `u_m = (1/N)Σ_n S_{n,m}` → 归一化熵 `H(u)` → `L_bal = λ(1 - H(u))`（虚线指向 loss）。
- 标注 "λ=0 disables balancing (ablation A3)"。

**图例（必画）**：
- 实线=张量流；虚线=门控/正则旁路。
- 三个消融开关（A1 refine / A2 三 flag / A3 balance）用统一小标签样式标在对应位置，
  与 Table III/IV/V 呼应。

**配色**：四个 stage 用四种邻近色块分区；门控元素（γ/β/α/τ）用统一强调色的小圆。

---

## 备注

- 两图均为结构示意，不含任何数值/指标，符合 method 草稿"无指标断言"边界。
- 出图后核对：箭头方向与 Eq. 编号一一对应；开关标签与 protocol §10.C / traceability A1–A3 一致。
- 落地工具建议：Fig.1 用 draw.io（块状清晰）；Fig.2 若投 IEEE/Elsevier 可用 TikZ 保证矢量与字体统一。
