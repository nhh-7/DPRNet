# CATANet x2 scratch 日志分析交接记录（供后续 AI 继续追加）

实验目录：`CATANet/experiments/train_CATANet_x2_scratch`

> 用途：
> - 这份文件用于给后续 AI 快速接手实验。
> - 默认按“追加记录”方式维护，不要覆盖历史结论。
> - 后续如果有新日志分析，直接在文末追加一个新小节即可。

---

## 1. 实验背景与总原则

这是 DPRNet / CATANet 里的一个超分实验，当前主实验目录是：

- `CATANet/experiments/train_CATANet_x2_scratch`

原始目标不是只看短程点，而是沿着长程训练路线继续推进，目标尺度是：

- 长程训练到 `800k iter`

这个实验此前的主线问题一直是：

- **router collapse（路由塌缩）**，尤其是深层 block 的局部塌缩

到目前为止形成的总体原则是：

1. **不要轻易从头训练**；
2. **不要因为 50k / 90k / 100k 这类短程平台感就否定长程方案**；
3. 优先采用**保守修正**，避免一次性大改全局配置；
4. 重点盯深层 `b6 / b7`，而不是把 `b1 / b3` 当成当前主矛盾；
5. 先观察 milestone 之后的真实 post-decay 行为，再讨论是否要改 scheduler。

---

## 2. 早期已做过的修正

此前围绕 router collapse 已经做过这些修改：

- 限制 `router_scale`
- `router_scale_init = 6.0`
- `max_router_logit_scale = 10.0`
- `router_scale_lr_mult = 0.1`
- 加入 `route_balance_weight`
- 加入 `l_route`
- 修复多验证集 warning

后续又做过一次**只加强深层 b6 / b7 的保守配置调整**：

```yaml
route_balance_weight: [0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0007, 0.0007]
```

关键点：

- 前 6 个 block 仍然保持 `0.0005`
- 只把 `b6 / b7` 提到 `0.0007`
- 没有做全局上调到 `0.001`
- 没有改 milestones

---

## 3. 截至目前的主时间线

### 3.1 0 -> 50k

对应日志：

- `train_train_CATANet_x2_scratch_20260518_120949.log`

阶段结论：

- 旧版那种灾难性塌缩已经明显缓解；
- `router_scale` 基本稳定在 6 左右；
- 训练重新回到可用状态。

关键指标（best up to 50k）：

- Set5：`38.0297 / 0.9607 @ 50000`
- Set14：`33.5746 @ 45000`，`0.9182 @ 50000`

路由状态（50k 末尾）：

- Set5 `b6`: active `53.8`, usage_max `0.2815`, entropy `0.5816`, scale `5.9961`
- Set5 `b7`: active `84.2`, usage_max `0.2723`, entropy `0.5360`, scale `5.9981`
- Set14 `b6`: active `59.1429`, usage_max `0.3039`, entropy `0.5938`, scale `5.9961`
- Set14 `b7`: active `93.8571`, usage_max `0.2043`, entropy `0.6087`, scale `5.9981`

判断：

- 早期 anti-collapse 修正是有效的。

---

### 3.2 90k（第一次明确暴露深层问题）

对应日志：

- `train_train_CATANet_x2_scratch_20260519_105034.log`

阶段结论：

- 指标在 90k 仍在增长，说明训练没有停；
- 真正暴露的问题是深层路由，主问题集中在 `b6`，`b7` 次之；
- `b1 / b3` 不是当前主矛盾。

关键指标：

- Set5：best `38.1362 @ 90000`，best SSIM `0.9612 @ 80000`
- Set14：best `33.8214 / 0.9200 @ 90000`

典型路由现象（90k 点附近）：

- Set5 `b6`: active `33.6`, usage_max `0.5265`, entropy `0.2891`, scale `5.9932`
- Set14 `b6`: active `38.5714`, usage_max `0.5088`, entropy `0.3081`, scale `5.9932`

判断：

- 90k 的核心不是“指标不涨”，而是**深层局部塌缩开始明显化**。

---

### 3.3 100k（高 LR 短程检查）

对应日志：

- `train_train_CATANet_x2_scratch_20260520_101755.log`

阶段结论：

- 到 100k 仍然处于高学习率阶段；
- 平台/震荡感是合理的，不能因此否定 800k 长程方案；
- 但 `b6 / b7` 的约束偏弱，说明需要对深层做小幅增强。

关键指标：

- Set5：final `38.0628 / 0.9608 @ 100000`
- Set5：best `38.1457 / 0.9612 @ 92500`
- Set14：final `33.7621 / 0.9194 @ 100000`
- Set14：best PSNR `33.7711 @ 92500`，best SSIM `0.9194 @ 100000`

100k 末尾路由状态：

- Set5 `b6`: active `28.2`, usage_max `0.5204`, entropy `0.3659`, scale `5.9481`
- Set5 `b7`: active `39.8`, usage_max `0.3152`, entropy `0.3852`, scale `6.0030`
- Set14 `b6`: active `32.7857`, usage_max `0.5807`, entropy `0.3472`, scale `5.9481`
- Set14 `b7`: active `49.9286`, usage_max `0.3628`, entropy `0.3843`, scale `6.0030`

判断：

- 不足以支持“停止长程训练”；
- 但足以支持做一次**只加强 b6 / b7** 的保守修正。

---

### 3.4 150k（只增强 b6 / b7 的保守分支）

对应日志：

- `train_train_CATANet_x2_scratch_20260520_141054.log`

当时实际配置：

- `resume_state = experiments/train_CATANet_x2_scratch/training_states/90000.state`
- `total_iter = 150000`
- `route_balance_weight = [0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0007, 0.0007]`

阶段结论：

- 修改有效；
- `b6 / b7` 的局部塌缩明显缓解；
- 没有看到旧版那种继续恶化趋势。

关键指标：

- Set5 best：`38.1760 / 0.9613 @ 135k`
- Set14 best：`33.8553 / 0.9204 @ 127.5k`

150k 末尾路由状态：

- Set5 `b6`: active `44.8`, usage_max `0.3819`, entropy `0.4308`, scale `5.9165`
- Set5 `b7`: active `54.6`, usage_max `0.2898`, entropy `0.4747`, scale `5.9397`
- Set14 `b6`: active `53.0714`, usage_max `0.4280`, entropy `0.4485`, scale `5.9165`
- Set14 `b7`: active `65.2143`, usage_max `0.2804`, entropy `0.5078`, scale `5.9397`

当时形成的建议：

- 不要重新开新配置；
- 不要从头训练；
- 直接基于 `150000.state` 继续往下跑；
- 暂时不要动 LR / milestones / 全局 route balance。

---

### 3.5 300k（长程阶段确认修正仍成立）

对应日志：

- `train_train_CATANet_x2_scratch_20260522_000211.log`

确认配置：

- `resume_state = experiments/train_CATANet_x2_scratch/training_states/150000.state`
- `pretrain_network_g = experiments/train_CATANet_x2_scratch/models/net_g_150000.pth`
- `total_iter = 300000`
- `milestones = [300000, 500000, 650000, 700000, 750000]`
- `route_balance_weight = [0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0007, 0.0007]`

阶段结论：

- 相比 150k，Set5 / Set14 还有进一步提升；
- `b6 / b7` 没有重新塌回去，整体明显好于旧 90k / 100k；
- `router_scale` 仍然稳定；
- 说明前面的修正策略到 300k 依然成立。

关键指标：

- Set5 best PSNR：`38.2119 @ 287500`
- Set5 best SSIM：`0.9616 @ 280000`
- Set14 best PSNR：`33.9140 @ 300000`
- Set14 best SSIM：`0.9212 @ 255000`

300k 点路由状态：

- Set5 `b6`: active `54.6`, usage_max `0.3223`, entropy `0.5942`, scale `5.8860`
- Set5 `b7`: active `83.4`, usage_max `0.2068`, entropy `0.5783`, scale `5.9875`
- Set14 `b6`: active `58.0`, usage_max `0.3634`, entropy `0.5403`, scale `5.8860`
- Set14 `b7`: active `103.2857`, usage_max `0.2557`, entropy `0.5734`, scale `5.9875`

判断：

- `150k -> 300k` 不是空跑；
- “只小幅提高 b6 / b7” 的保守策略到 300k 仍然成立且有效。

---

### 3.6 400k（post-decay 阶段）

对应日志：

- `train_train_CATANet_x2_scratch_20260523_135513.log`
- `train_train_CATANet_x2_scratch_20260524_005918.log`

确认配置延续：

- 仍然沿用：
  - `route_balance_weight = [0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0007, 0.0007]`
- 后续存在一次中途恢复：
  - `resume_state = experiments/train_CATANet_x2_scratch/training_states/327500.state`

关键结论：

- `300k` milestone 后的 LR decay **已经生效**；
- 主 LR 从 `2e-4` 进入 `1e-4`，router group 从 `2e-5` 进入 `1e-5`；
- `300k -> 400k` 期间指标仍有提升；
- 没有出现新的致命问题；
- `b6 / b7` 总体仍健康，没有回到旧版严重塌缩状态。

关键指标：

- Set5 best PSNR：`38.2484 @ 392500`
- Set5 best SSIM：`0.9616 @ 380000`
- Set14 best PSNR：`34.0270 @ 392500`
- Set14 best SSIM：`0.9212 @ 345000`

400k 点路由状态：

- Set5 `b6`: active `55.8`, usage_max `0.2607`, entropy `0.6270`, scale `5.8553`
- Set5 `b7`: active `89.0`, usage_max `0.2572`, entropy `0.5564`, scale `6.0317`
- Set14 `b6`: active `56.2143`, usage_max `0.3604`, entropy `0.5369`, scale `5.8553`
- Set14 `b7`: active `104.5`, usage_max `0.2584`, entropy `0.5774`, scale `6.0317`

额外判断：

- Set14 的 `b6` 在约 `365k` 一带出现过一次局部偏紧区间；
- 但后续恢复了，没有演化成结构性恶化；
- 因此不把它判断成新的系统性问题。

---

## 4. 到目前为止的总判断（截至 400k）

可以认为，目前已有较强证据支持以下判断：

1. **早期 router collapse 的主问题已经被修正到可接受范围**；
2. **只小幅加强 `b6 / b7` 的保守路线是成功的**；
3. 训练从 `150k -> 300k -> 400k` 都还有收益；
4. `300k` milestone 之后的 decay 行为已经生效，并非 scheduler 没起作用；
5. 目前没有充分理由：
   - 从头训练；
   - 全局提高 route balance；
   - 继续上调 `b6 / b7` 到 `0.0008 / 0.001`；
   - 立刻修改 milestones；
   - 围绕 `b1` 再开新干预分支。

---

## 5. 当前默认建议

截至当前，这个实验的默认推进策略应该是：

- **继续基于最新 state 往下跑，不要从头来**；
- 保持当前配置不变：

```yaml
route_balance_weight: [0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0007, 0.0007]
```

- 暂时不要：
  - 全局上调 route balance；
  - 再次抬高 `b6 / b7`；
  - 修改 milestones；
  - 为 `b1` 做额外干预。

下一步最合理的观察窗口：

- `400k -> 500k`

重点观察三件事：

1. decay 后，PSNR / SSIM 是否继续更平滑地爬升；
2. `b6 / b7` 是否继续保持健康；
3. 是否开始出现真正意义上的收敛平台，而不是过渡期震荡平台。

---

## 6. 已知日志 / 工程细节注意事项

以下现象已经出现过，默认视为**日志或计数 artifact**，不要过度解读为训练故障：

- `50002`
- `100004`
- `300005`
- 日志尾部重复 validation

当前判断：

- 这些更像收尾阶段的重复验证 / 计数偏移；
- **不像训练发散、NaN 或数值不稳定**。

---

## 7. 当前相关文件

### 7.1 主要日志

- `train_train_CATANet_x2_scratch_20260517_224304.log`
- `train_train_CATANet_x2_scratch_20260518_120949.log`
- `train_train_CATANet_x2_scratch_20260519_105034.log`
- `train_train_CATANet_x2_scratch_20260520_101755.log`
- `train_train_CATANet_x2_scratch_20260520_141054.log`
- `train_train_CATANet_x2_scratch_20260522_000211.log`
- `train_train_CATANet_x2_scratch_20260523_135513.log`
- `train_train_CATANet_x2_scratch_20260524_005918.log`

---

## 8. 给后续 AI 的建议工作流

如果后续继续接这个实验，建议按下面顺序处理：

1. **先看最新日志是否有新阶段**
   - 重点确认最新 `resume_state`、`total_iter`、LR 是否发生新变化。

2. **先和这几个历史关键点对比**
   - 50k
   - 90k
   - 100k
   - 150k
   - 300k
   - 400k

3. **优先判断是否出现新的结构性问题**
   - 看 `b6 / b7` 是否重新恶化；
   - 看 router scale 是否失控；
   - 看是否出现发散、NaN、loss 异常。

4. **如果只是小震荡，不要马上改配置**
   - 先区分“高 LR / 过渡期震荡”与“真实塌缩 / 真正收敛平台”。

5. **默认优先继续已有 state，而不是另起炉灶**
   - 除非出现新的强证据证明当前路线失败。

---

## 9. 后续追加模板（直接复制到文末）

后续有新分析时，直接在文末追加一个类似小节：

```md
---

## Update YYYY-MM-DD / XXXk -> YYYk

### 日志 / checkpoint
- 日志：
- model：
- state：
- resume_state：
- total_iter：
- milestones：
- route_balance_weight：

### 关键观察
- LR 是否变化：
- Set5 best / final：
- Set14 best / final：
- b6 状态：
- b7 状态：
- 是否出现新问题：

### 结论
- 

### 下一步建议
- 
```

---

## 10. 本次交接落点（截至 2026-05-24）

本次交接完成时，最新稳定结论是：

- 训练已经顺利从 `300k` 推进到 `400k`；
- `300k` 后 LR decay 已生效；
- `300k -> 400k` 仍有收益；
- `b6 / b7` 总体健康，没有重新塌回旧版严重状态；
- 当前默认建议仍是：**继续从最新 state 往下跑，先不要改配置**。

---

## Update 2026-05-25 / 400k -> 550k

### 日志 / checkpoint
- 日志：
  - `train_train_CATANet_x2_scratch_20260524_115647.log`
  - `train_train_CATANet_x2_scratch_20260524_231938.log`
- model：
  - 中间关键点：`net_g_500000.pth`、`net_g_527500.pth`、`net_g_530000.pth`
  - 最新点：`net_g_550000.pth`
- state：
  - 中间关键点：`500000.state`、`527500.state`、`530000.state`
  - 最新点：`550000.state`
- resume_state：
  - 第一段：`experiments/train_CATANet_x2_scratch/training_states/400000.state`
  - 第二段：`experiments/train_CATANet_x2_scratch/training_states/475000.state`
- total_iter：
  - 第一段：`650000`
  - 第二段：`550000`
- milestones：`[300000, 500000, 650000, 700000, 750000]`
- route_balance_weight：`[0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0007, 0.0007]`

### 关键观察
- LR 是否变化：
  - `500000` 时训练打印仍是 `1e-4 / 1e-5`；
  - `500200` 起切到 `5e-5 / 5e-6`；
  - 说明 **500k milestone 的 decay 正常生效**。
- Set5 best / final：
  - 400k 之后 best PSNR：`38.2649 @ 530000`
  - 400k 之后 best SSIM：`0.9616`（多点并列，按当前汇总规则 latest best iter 记到 `550000`）
  - final：`38.2582 / 0.9616 @ 550000`
- Set14 best / final：
  - 400k 之后 best PSNR：`34.0286 @ 527500`
  - 400k 之后 best SSIM：`0.9216 @ 500000`
  - final：`33.9606 / 0.9207 @ 550000`
- 400k -> 550k 阶段性表现：
  - `402.5k -> 447.5k`：Set5 / Set14 均值约 `38.2338 / 33.9467`
  - `450k -> 497.5k`：Set5 / Set14 均值约 `38.2376 / 33.9478`
  - `500k -> 550k`：Set5 / Set14 均值约 `38.2536 / 33.9827`
  - 说明 post-decay 阶段整体均值**略优于**前两段，不是 scheduler 带来的负收益。
- b6 状态：
  - `500k` Set5 `b6`: active `55.4`, usage_max `0.2917`, entropy `0.6106`, scale `5.8344`
  - `500k` Set14 `b6`: active `54.7857`, usage_max `0.4027`, entropy `0.5098`, scale `5.8344`
  - `550k` Set5 `b6`: active `54.8`, usage_max `0.2408`, entropy `0.6308`, scale `5.8291`
  - `550k` Set14 `b6`: active `54.0`, usage_max `0.3688`, entropy `0.5366`, scale `5.8291`
  - 判断：Set14 的 `b6` 仍然是相对最紧点，但整体仍在健康区间，没有重新演化成系统性塌缩。
- b7 状态：
  - `500k` Set5 `b7`: active `87.0`, usage_max `0.2410`, entropy `0.5686`, scale `6.0714`
  - `500k` Set14 `b7`: active `104.5`, usage_max `0.2338`, entropy `0.5952`, scale `6.0714`
  - `550k` Set5 `b7`: active `87.4`, usage_max `0.2196`, entropy `0.5840`, scale `6.0837`
  - `550k` Set14 `b7`: active `101.2857`, usage_max `0.2030`, entropy `0.6178`, scale `6.0837`
  - 判断：`b7` 整体稳定，甚至比早期深层问题阶段更健康，没有新风险迹象。
- 是否出现新问题：
  - 没有出现 NaN、发散、router scale 失控；
  - 没有看到 `b6 / b7` 重新 collapse；
  - 当前更像是**高位平台区中的边际刷新**，而不是训练失败。

### 结论
- `400k -> 550k` 不是空跑，仍然有收益，但收益已经明显进入边际区；
- **500k decay 是正确且有效的**，没有证据支持现在去修改 milestones；
- **只小幅加强 `b6 / b7` 的保守策略到 550k 依然成立**；
- 当前 best checkpoint 更适合从 `500000 / 527500 / 530000` 中选，而不是默认直接拿 `550000`；
- 目前仍然没有足够理由：
  - 从头训练；
  - 全局提高 route balance；
  - 再抬高 `b6 / b7`；
  - 立刻调整 scheduler。

### 下一步建议
- 如果主线目标仍是长程训练到更高 iter，建议：
  - **直接基于 `550000.state` 继续向下跑**；
  - 暂时保持当前配置不变；
  - 下一观察窗口重点看 `550k -> 650k`。
- 如果现在需要挑测试 checkpoint，优先建议：
  1. `net_g_527500.pth`（综合最平衡）
  2. `net_g_530000.pth`（Set5 PSNR 最优）
  3. `net_g_500000.pth`（Set14 SSIM 最优，且是关键 milestone 点）

