# DPR 模块独立实现方案

## 1. 文档目标

本文只讨论 `DPR` 模块的实现思路

目标只有一个：

```text
给定一段输入 token 序列，独立实现一个“基于内容相似性而不是空间邻近性”的动态语义原型路由模块。
```

---

## 2. 模块职责与边界

`DPR` 的职责不是做注意力，也不是做重建，而是做下面四件事：

1. 从当前输入 token 中生成一组动态语义原型。
2. 计算每个 token 属于哪个原型，以及归属置信度有多高。
3. 按“原型标签 + 归属置信度”对 token 重新排序。
4. 输出一组可复用的路由元信息。

因此，`DPR` 的核心问题可以表述为：

```text
如何把空间上分散、但内容上相似的 token 动态组织到相邻位置，
以便后续模块在更干净的候选集合上继续处理。
```

`DPR` 不负责：

1. 稀疏连接筛选。
2. 多级重建。
3. 最终图像恢复。
4. 原始二维空间上的局部卷积建模。

---

## 3. 输入输出接口

### 3.1 输入

输入为已经展平后的 token 序列：

$$
X \in \mathbb{R}^{B \times N \times C}
$$

其中：

- `B`：batch size
- `N`：token 数量
- `C`：token 通道维度

通常这一步来自图像特征：

```text
B x C x H x W -> B x N x C
N = H x W
```

### 3.2 输出

建议 `DPR` 输出以下 5 个张量：

```text
sorted_x
idx_last
sorted_belong_idx
sorted_scores
prototypes
```

分别含义如下：

- `sorted_x`：按语义原型和置信度重排后的 token，形状 `B x N x C`
- `idx_last`：恢复原始顺序所需的索引，形状 `B x N x 1`
- `sorted_belong_idx`：排序后每个 token 对应的原型编号，形状 `B x N`
- `sorted_scores`：排序后每个 token 的归属置信度，形状 `B x N`
- `prototypes`：最终动态语义原型，形状 `B x M x C`

其中 `M` 是原型数量，是可配置超参数。

---

## 4. 总体实现流程

推荐将 `DPR` 的计算过程拆成 5 步：

```text
输入 token X
-> token embedding
-> soft assignment 聚合内容原型
-> prototype query refine
-> token-to-prototype 反向匹配
-> 内容排序与元信息输出
```

进一步展开后可写成：

```text
X
-> E = Embed(X)
-> S0 = Softmax(Assign(E))
-> P_content = WeightedAggregate(X, S0)
-> P = Refine(P_content, E, X)
-> S = Softmax(phi(E) · psi(P)^T / sqrt(d))
-> belong_idx, x_scores = argmax/max(S)
-> sorted_idx = argsort(sort_key)
-> gather 得到排序后的 token 和元信息
```

---

## 5. 动态语义原型生成

### 5.1 目标

这一步要解决的问题是：

```text
原型必须来自当前输入内容，而不是只依赖一组固定参数。
```

因此，推荐使用“软分配聚合”的方式先生成内容原型，再进行轻量细化。

### 5.2 Token 嵌入

先对输入 token 做一层轻量嵌入：

```python
embed = LayerNorm(C) -> Linear(C, C) -> GELU
```

得到：

$$
E \in \mathbb{R}^{B \times N \times C}
$$

这一层的作用不是改变结构，而是给后续分配头提供更稳定的内容表征。

### 5.3 Soft Assignment 生成内容原型

使用线性层预测每个 token 到 `M` 个原型槽位的初始分配：

$$
L = W_{assign}(E), \quad L \in \mathbb{R}^{B \times N \times M}
$$

对原型维做 Softmax：

$$
S_0 = \text{Softmax}(L, \text{dim}=M)
$$

其中：

- `S0[b, n, m]` 表示第 `b` 个样本中，第 `n` 个 token 属于第 `m` 个原型槽位的软分配权重

然后用 `S0` 对原始 token `X` 做加权聚合，得到内容原型：

$$
P_{content}[b,m,:] =
\frac{\sum_{n=1}^{N} S_0[b,n,m] \cdot X[b,n,:]}
{\sum_{n=1}^{N} S_0[b,n,m] + \varepsilon}
$$

矩阵实现可以写成：

```python
proto_content = einsum("bnm,bnc->bmc", assignment, x)
proto_weight = assignment.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
prototypes = proto_content / proto_weight
```

这里有两个关键点：

1. 聚合时使用的是原始 `X`，不是 `E`。
2. 分母必须 `clamp_min(eps)`，否则某些原型槽位几乎没有分配到 token 时会数值不稳定。

### 5.4 归一化

初始内容原型生成后，建议立刻做一次：

```python
prototypes = normalize(LayerNorm(prototypes), dim=-1)
```

这样做的原因是：

1. 降低不同原型范数差异造成的匹配偏置。
2. 让后续 token 与 prototype 的相似度更接近方向相似度。
3. 提高排序和置信度估计的稳定性。

---

## 6. Prototype Query Refine

### 6.1 为什么还要 refine

只靠软分配聚合虽然已经能得到输入自适应的内容原型，但有两个问题：

1. 不同原型槽位可能出现分工不稳定。
2. 某些噪声 token 可能把内容中心拉偏。

因此推荐再加一个轻量的 `prototype query refine`。

### 6.2 可学习槽位查询

维护一组可学习参数：

$$
Q_{proto} \in \mathbb{R}^{M \times C}
$$

它们不是最终原型本身，而是每个原型槽位的稳定锚点。

构造 refine 查询：

```python
query_seed = prototypes + prototype_queries.unsqueeze(0)
```

即把“当前图像中的内容中心”和“全局可学习槽位身份”相加。

### 6.3 轻量 cross-attention refine

设路由空间维度为 `d_r`，一般可取与后续注意力 `qk_dim` 相同或接近。

做三组线性映射：

```python
q_proto = Linear(C, d_r)(query_seed)
k_tokens = Linear(C, d_r)(embed)
v_tokens = Linear(C, C)(x)
```

然后计算原型对全部 token 的 refine 注意力：

$$
A_{refine} = \text{Softmax}\left(\frac{Q_{proto}K_{token}^{T}}{\sqrt{d_r}}\right)
$$

得到 refine 结果：

$$
P_{refine} = A_{refine}V_{token}
$$

最终做残差式更新：

$$
P = \text{Norm}(P_{content} + \gamma \cdot P_{refine})
$$

其中：

- `gamma` 建议做成可学习门控
- 实现上可维护一个标量参数 `refine_gate`
- 使用 `sigmoid(refine_gate)` 将其约束到 `(0, 1)` 区间

对应实现：

```python
gamma = torch.sigmoid(self.refine_gate)
prototypes = F.normalize(self.prototype_norm(prototypes + gamma * proto_refine), dim=-1)
```

### 6.4 这一设计的本质

这一步不是让 learnable query 直接替代动态原型，而是做“弱约束”：

```text
动态内容中心负责输入自适应
+ 槽位 query 负责中心分工稳定
= 更可靠的动态语义原型
```

---

## 7. Token 到原型的反向匹配

### 7.1 目的

原型生成完成后，需要重新计算每个 token 到最终原型的归属关系。

注意，这一步不能直接复用前面的 `S0`。

原因是：

1. `S0` 只是生成内容原型时的初始软分配。
2. refine 之后的原型已经发生变化。
3. 最终路由必须基于 refine 后的最终原型重新确认。

### 7.2 双投影匹配

分别对 token 和 prototype 做投影：

```python
token_features = normalize(Linear(C, d_r)(embed), dim=-1)
prototype_features = normalize(Linear(C, d_r)(prototypes), dim=-1)
```

然后计算 token 到 prototype 的归属概率：

$$
S = \text{Softmax}\left(
\frac{\phi(E)\psi(P)^T}{\sqrt{d_r}}
\right)
$$

这里：

- `phi(E)` 是 token 投影特征，形状 `B x N x d_r`
- `psi(P)` 是 prototype 投影特征，形状 `B x M x d_r`
- `S` 的形状为 `B x N x M`

### 7.3 取主归属标签与归属置信度

对每个 token，取最大概率对应的原型编号和概率值：

```python
x_scores, belong_idx = torch.max(scores, dim=-1)
```

得到：

- `belong_idx`：每个 token 最终属于哪个原型，形状 `B x N`
- `x_scores`：该 token 对所属原型的归属置信度，形状 `B x N`

`x_scores` 的含义很重要：

1. 值越高，说明该 token 与所属原型越一致。
2. 值越低，说明该 token 靠近边界或语义更混杂。
3. 后续如果其他模块要利用路由纯度，通常就是使用这个值。

---

## 8. Token 排序与候选集合构建

### 8.1 为什么要排序

`DPR` 的关键收益不是只得到 `belong_idx`，而是把属于同一原型的 token 排到一起。

这样做之后：

1. 同类 token 在序列上变得相邻。
2. 后续分组处理时更容易构造语义一致的候选集合。
3. 即使不使用二维窗口，也能形成基于内容相似性的局部邻域。

### 8.2 当前实现采用的排序键

建议直接使用如下排序键：

$$
\text{sort\_key} = belong\_idx + 0.5 \cdot (1 - x\_scores)
$$

然后：

```python
sorted_idx = torch.argsort(sort_key, dim=-1)
```

这条规则非常关键，它等价于以下两层排序逻辑：

1. 先按 `belong_idx` 排序，使同一原型编号的 token 聚到一起。
2. 在同一原型组内部，高置信 token 具有更小的附加项，因此会更靠前。

为什么是 `0.5 * (1 - x_scores)`：

1. `belong_idx` 是整数标签，主导原型级排序。
2. `x_scores ∈ (0, 1)`，因此附加项落在 `(0, 0.5)`。
3. 附加项不会跨越相邻整数标签的边界，因此不会打乱不同原型之间的主排序。
4. 同一原型组内部又能形成“高置信在前，低置信在后”的自然次序。

### 8.3 Gather 得到排序结果

```python
gather_idx = sorted_idx.unsqueeze(-1).expand(B, N, C)
sorted_x = torch.gather(x, dim=1, index=gather_idx)
sorted_belong_idx = torch.gather(belong_idx, dim=1, index=sorted_idx)
sorted_scores = torch.gather(x_scores, dim=1, index=sorted_idx)
idx_last = sorted_idx.unsqueeze(-1)
```

这里要注意：

`idx_last` 在这个实现中保存的是：

```text
排序后第 j 个 token，在原始序列中的位置
```

也就是“sorted position -> original position”的映射。

### 8.4 恢复原始顺序

如果后续模块在排序后的序列上完成计算，需要恢复回原始位置，可以直接用：

```python
def restore_sorted_tokens(sorted_x, idx_last):
    out = torch.zeros_like(sorted_x)
    return out.scatter(dim=1, index=idx_last.expand_as(sorted_x), src=sorted_x)
```

这一点必须保证正确，否则后续恢复时会出现 token 空间错位。

---

## 9. 推荐的 PyTorch 实现骨架

下面给出一个与当前实现思路一致的独立骨架：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_logit(prob: float) -> torch.Tensor:
    prob = min(max(prob, 1e-4), 1.0 - 1e-4)
    return torch.logit(torch.tensor(prob, dtype=torch.float32))


class DPR(nn.Module):
    def __init__(self, dim: int, router_dim: int, num_prototypes: int):
        super().__init__()
        self.dim = dim
        self.router_dim = router_dim
        self.num_prototypes = num_prototypes

        self.embed = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
        )

        self.assign = nn.Linear(dim, num_prototypes, bias=False)
        self.prototype_queries = nn.Parameter(torch.randn(num_prototypes, dim) * 0.02)

        self.refine_q = nn.Linear(dim, router_dim, bias=False)
        self.refine_k = nn.Linear(dim, router_dim, bias=False)
        self.refine_v = nn.Linear(dim, dim, bias=False)

        self.token_proj = nn.Linear(dim, router_dim, bias=False)
        self.prototype_proj = nn.Linear(dim, router_dim, bias=False)

        self.prototype_norm = nn.LayerNorm(dim)
        self.refine_gate = nn.Parameter(init_logit(0.25))
        self.scale = router_dim ** -0.5

    def forward(self, x):
        b, n, c = x.shape
        embed = self.embed(x)

        assignment = F.softmax(self.assign(embed), dim=-1)
        proto_content = torch.einsum("bnm,bnc->bmc", assignment, x)
        proto_weight = assignment.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        prototypes = proto_content / proto_weight
        prototypes = F.normalize(self.prototype_norm(prototypes), dim=-1)

        query_seed = prototypes + self.prototype_queries.unsqueeze(0)
        q_proto = self.refine_q(query_seed)
        k_tokens = self.refine_k(embed)
        v_tokens = self.refine_v(x)
        refine_attn = F.softmax(torch.matmul(q_proto, k_tokens.transpose(-2, -1)) * self.scale, dim=-1)
        proto_refine = torch.matmul(refine_attn, v_tokens)

        gamma = torch.sigmoid(self.refine_gate)
        prototypes = F.normalize(self.prototype_norm(prototypes + gamma * proto_refine), dim=-1)

        token_features = F.normalize(self.token_proj(embed), dim=-1)
        prototype_features = F.normalize(self.prototype_proj(prototypes), dim=-1)
        scores = F.softmax(torch.matmul(token_features, prototype_features.transpose(-2, -1)) * self.scale, dim=-1)

        x_scores, belong_idx = torch.max(scores, dim=-1)
        sort_key = belong_idx.to(scores.dtype) + 0.5 * (1.0 - x_scores)
        sorted_idx = torch.argsort(sort_key, dim=-1)

        gather_idx = sorted_idx.unsqueeze(-1).expand(b, n, c)
        sorted_x = torch.gather(x, dim=1, index=gather_idx)
        sorted_belong_idx = torch.gather(belong_idx, dim=1, index=sorted_idx)
        sorted_scores = torch.gather(x_scores, dim=1, index=sorted_idx)
        idx_last = sorted_idx.unsqueeze(-1)

        return sorted_x, idx_last, sorted_belong_idx, sorted_scores, prototypes
```

---

## 10. 模块初始化建议

### 10.1 关键超参数

建议至少暴露以下参数：

- `dim`：输入 token 通道维
- `router_dim`：路由相似度计算维度
- `num_prototypes`：原型个数 `M`
- `use_prototype_query_refine`：是否启用 refine
- `prototype_mode`：动态原型或固定原型

### 10.2 默认建议

对于独立实现，推荐初始设置：

- `router_dim = dim` 或略小于 `dim`
- `num_prototypes` 取 `8 / 16 / 32` 中的一档起步
- `prototype_query` 使用 `N(0, 0.02)` 初始化
- `refine_gate` 初始概率可设为 `0.25`

如果 token 数较多，可以让 `num_prototypes` 随序列长度增大而增加，但不建议一开始设置过大，否则：

1. 原型之间会变稀疏。
2. 单个原型收到的 token 变少。
3. 路由稳定性反而可能下降。

---

## 11. 可选消融实现

为了后续验证设计是否有效，建议保留两个消融开关。

### 11.1 固定原型模式

可以直接维护：

```python
self.fixed_prototypes = nn.Parameter(torch.randn(M, C) * 0.02)
```

前向时不做 soft assignment 聚合，而是直接：

```python
prototypes = self.fixed_prototypes.unsqueeze(0).expand(B, -1, -1)
```

这可以用于验证“动态原型是否真的比固定原型更有效”。

### 11.2 去掉 prototype-query refine

只保留：

```text
soft assignment 聚合 -> normalize -> 直接反向匹配
```

这可以用于验证 refine 是否提升了原型槽位稳定性。

---

## 12. 复杂度与开销分析

`DPR` 的主要开销来自三部分：

1. `assign(embed)` 的 `B x N x M`
2. 内容原型聚合 `einsum("bnm,bnc->bmc")`
3. refine 阶段的原型对 token 的注意力 `B x M x N`

相较于直接对全部 token 做全局自注意力，`DPR` 的计算重点是：

```text
token -> prototype
```

而不是：

```text
token -> token
```

因此只要 `M << N`，这一模块通常是比较轻量的。

---

## 13. 工程实现细节与坑点

### 13.1 `idx_last` 的语义必须统一

本文推荐的 `idx_last` 语义是：

```text
排序后位置 -> 原始位置
```

如果你在其他工程中把它实现成“原始位置 -> 排序后位置”，恢复函数就必须跟着一起改，不能混用。

### 13.2 排序必须逐 batch 独立进行

不能把不同 batch 的 token 混在一起排序。

### 13.3 `sorted_belong_idx` 和 `sorted_scores` 必须跟随 `sorted_idx` 一起 gather

不能只排序 `x`，否则后续标签和分数会错位。

### 13.4 原型归一化建议保留

不建议直接拿未归一化的 prototype 做匹配，否则：

1. 大范数原型会更容易抢占 token。
2. 归属概率更不稳定。
3. 不同 batch 之间分布波动更大。

### 13.5 低分配槽位的数值稳定性

某些图像内容简单时，可能只有少数几类语义明显存在，这会导致部分原型槽位收到极少 token。

因此聚合分母必须：

```python
clamp_min(1e-6)
```

### 13.6 `sort_key` 的次级排序不要破坏主标签顺序

如果你用：

```text
belong_idx - x_scores
```

这类范围过大的排序键，就可能让某些高分 token 跳到相邻原型组前面，破坏“同类先聚集”的主目标。

`0.5 * (1 - x_scores)` 的价值就在于只做组内细排，不打乱组间主排序。

---

## 14. 独立实现时的最小交付标准

如果你只想先做一个可用版本，最少需要满足下面这些条件：

1. 输入 `B x N x C`，输出 `sorted_x / idx_last / labels / scores / prototypes`
2. 原型来自当前输入 token 的 soft assignment 聚合
3. 生成后的原型再做一次 token 反向匹配
4. 排序规则至少满足“同标签聚集，高置信优先”
5. 可以无误恢复到原始 token 顺序

如果以上 5 点都满足，这个 `DPR` 就已经具备独立可复用性。

---

## 15. 一句话总结

`DPR` 的本质不是“把 token 分组”，而是：

```text
先从当前输入中动态生成语义中心，
再让每个 token 向这些语义中心显式归属，
最后按归属结果重新组织序列。
```

也就是说，它把原本基于空间邻近的组织方式，改造成了基于内容相似性的动态语义路由方式。这就是该模块能够独立成立、并且适合迁移到其他项目中的根本原因。
