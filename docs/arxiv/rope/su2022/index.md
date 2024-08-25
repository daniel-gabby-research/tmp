---
layout: default
parent: Positional Embedding
grand_parent: ArXiv
title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
nav_order: 1
discuss: true
math: katex
---

# RoPE及其变体的全面分析：Enhanced Transformer with Rotary Position Embedding

Paper link: [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)

## 1. RoPE的引入与基本原理

### 1.1 背景

Transformer模型的成功很大程度上依赖于其捕捉序列中位置信息的能力。早期的方法，如正弦位置编码，虽然有效但存在一些局限性。Rotary Position Embedding (RoPE) 作为一种新的位置编码方法被引入，旨在解决这些问题并提供更强大的位置表示。

### 1.2 RoPE的核心思想

RoPE的核心思想是将位置信息编码为旋转操作。对于位置m处的向量x，RoPE定义了一个旋转函数：

$$f_\theta(x, m) = x e^{im\theta} = (x_1 + ix_2)(\cos(m\theta) + i\sin(m\theta))$$

这里，θ是一个预定义的角频率，通常定义为：

$$\theta_j = \frac{1}{10000^{2j/d}}$$

其中d是嵌入维度，j是维度索引。

### 1.3 RoPE的矩阵形式

在实际应用中，RoPE可以表示为矩阵乘法。对于2d维的向量x = [x_1, x_2, ..., x_2d]，RoPE操作可以表示为：

$$
\begin{bmatrix}
\cos(m\theta_0) & -\sin(m\theta_0) & 0 & 0 & \cdots \\
\sin(m\theta_0) & \cos(m\theta_0) & 0 & 0 & \cdots \\
0 & 0 & \cos(m\theta_1) & -\sin(m\theta_1) & \cdots \\
0 & 0 & \sin(m\theta_1) & \cos(m\theta_1) & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
x_4 \\
\vdots
\end{bmatrix}
$$

这个矩阵形式清楚地展示了RoPE如何对每对相邻维度应用旋转。

## 2. RoPE在Attention中的应用

### 2.1 Attention机制回顾

在自注意力机制中，我们计算查询(Q)、键(K)和值(V)之间的关系。attention分数通常通过Q和K的点积来计算。

### 2.2 RoPE与Attention的结合

将RoPE应用到attention机制中，我们得到：

$$\langle f_\theta(q, m), f_\theta(k, n) \rangle = \langle q, k \rangle \cos((m-n)\theta) + \langle q_\perp, k \rangle \sin((m-n)\theta)$$

这里，$q⊥ = [-q_1, q_0, -q_3, q_2, ...]$，是$q$的一个特殊置换。

### 2.3 数学推导

让我们详细推导这个公式：

1) 首先，展开 f_θ(q, m) 和 f_θ(k, n)：

$$
\begin{align*}
f_θ(q, m) &= q e^{imθ} = (q_1 + iq_2)(\cos(mθ) + i \sin(mθ)) \\
f_θ(k, n) &= k e^{inθ} = (k_1 - ik_2)(\cos(nθ) - i \sin(nθ))
\end{align*}
$$

2) 计算它们的内积：

$$
\begin{align*}
⟨f_θ(q, m), f_θ(k, n)⟩ &= (q_1 + iq_2)(\cos(mθ) + i \sin(mθ))(k_1 - ik_2)(\cos(nθ) - i \sin(nθ))
&= [(q_1k_1 + q_2k_2) + i(q_2k_1 - q_1k_2)](\cos((m-n)θ) + i \sin((m-n)θ)) \\
&= (q_1k_1 + q_2k_2)\cos((m-n)θ) - (q_2k_1 - q_1k_2)\sin((m-n)θ) + i[...]
\end{align*}
$$

3) 取实部：

$$
\begin{align*}
\text{Re}[⟨f_θ(q, m), f_θ(k, n)⟩] &= (q_1k_1 + q_2k_2)\cos((m-n)θ) + (q_1k_2 - q_2k_1)\sin((m-n)θ)
\end{align*}
$$

4) 注意到 $q_⊥ = [-q_2, q_1]$，因此：

$$
\begin{align*}
⟨q, k⟩ &= q_1k_1 + q_2k_2 \\
⟨q_⊥, k⟩ &= q_1k_2 - q_2k_1
\end{align*}
$$

这就得到了我们的最终公式。

## 3. RoPE的变体：引入Receptive Field

### 3.1 变体的动机

尽管原始RoPE表现优秀，研究者们仍在探索如何进一步增强其性能。一个主要的想法是引入可学习的参数，使模型能够根据任务需求自适应地调整位置编码。

### 3.2 新的角频率定义

在这个变体中，新的角频率θ*定义为：

$$\theta^*_j = \frac{\alpha}{10000^{\rho}} + (1 - \alpha)\theta_j$$

这里，ρ 是一个新引入的"receptive field"参数，α 是一个可学习的权重。

### 3.3 参数解释

- α：控制新引入的全局频率成分和原始RoPE频率的平衡。
- ρ：调整新引入的全局频率成分的尺度。

## 4. 变体RoPE的详细数学推导

### 4.1 新的attention score公式

使用新的 $θ^*$，attention score可以表示为：

$$a^*(m-n) = \text{Re}\left[\sum_{j=0}^{d/2-1} (q_{2j} + iq_{2j+1})(k_{2j} - ik_{2j+1})e^{i(m-n)\theta^*_j}\right]$$

### 4.2 展开 θ*

将 θ* 的定义代入上述表达式：

$$\begin{align*}
a^*(m-n) &= \text{Re}\left[\sum_{j=0}^{d/2-1} (q_{2j} + iq_{2j+1})(k_{2j} - ik_{2j+1})e^{i(m-n)(\frac{\alpha}{10000^{\rho}} + (1 - \alpha)\theta_j)}\right] \\
&= \text{Re}\left[\sum_{j=0}^{d/2-1} (q_{2j} + iq_{2j+1})(k_{2j} - ik_{2j+1})e^{i(m-n)\frac{\alpha}{10000^{\rho}}}e^{i(m-n)(1 - \alpha)\theta_j}\right]
\end{align*}$$

### 4.3 引入 φ

根据您的建议，我们定义：

$$\phi = \frac{\alpha}{10000^{\rho}}$$

那么，我们可以将表达式重写为：

$$a^*(m-n) = \text{Re}\left[e^{i(m-n)\phi} \sum_{j=0}^{d/2-1} (q_{2j} + iq_{2j+1})(k_{2j} - ik_{2j+1})e^{i(m-n)(1 - \alpha)\theta_j}\right]$$

### 4.4 分离实部和虚部

为了更好地理解这个表达式，我们可以将其分解为实部和虚部：

$$\begin{align*}
a^*(m-n) &= \cos((m-n)\phi) \cdot \text{Re}\left[\sum_{j=0}^{d/2-1} (q_{2j} + iq_{2j+1})(k_{2j} - ik_{2j+1})e^{i(m-n)(1 - \alpha)\theta_j}\right] \\
&- \sin((m-n)\phi) \cdot \text{Im}\left[\sum_{j=0}^{d/2-1} (q_{2j} + iq_{2j+1})(k_{2j} - ik_{2j+1})e^{i(m-n)(1 - \alpha)\theta_j}\right]
\end{align*}$$

### 4.5 引入 a' 和 b'

定义：

$$a'(m-n) = \text{Re}\left[\sum_{j=0}^{d/2-1} (q_{2j} + iq_{2j+1})(k_{2j} - ik_{2j+1})e^{i(m-n)(1 - \alpha)\theta_j}\right]$$

$$b'(m-n) = \text{Im}\left[\sum_{j=0}^{d/2-1} (q_{2j} + iq_{2j+1})(k_{2j} - ik_{2j+1})e^{i(m-n)(1 - \alpha)\theta_j}\right]$$

那么，最终的表达式可以简化为：

$$a^*(m-n) = \cos((m-n)\phi) \cdot a'(m-n) - \sin((m-n)\phi) \cdot b'(m-n)$$

## 5. 变体RoPE的特性分析

### 5.1 Hilbert变换对

$a'(m-n)$ 和 $b'(m-n)$ 构成了一个Hilbert变换对。这意味着它们捕捉了信号的不同方面，提供了更丰富的表示。

### 5.2 局部性与全局性的平衡

- $φ$ 项引入了一个全局的周期调制，其频率由 $ρ$ 控制。
- $(1-α)θ_j$ 项保留了原始RoPE的局部敏感性，但其影响被 $α$ 调节。

### 5.3 θ 对attention分布的影响

通过数学分析，我们发现：

1. θ 较大时，attention更集中在对角线附近，呈现更局部化的特征。
2. θ 较小时，attention分布更平坦，体现更全局的特征。

这可以通过cosine函数的Taylor展开来理解：

$$\cos((m-n)\theta) \approx 1 - \frac{1}{2}((m-n)\theta)^2 + O((m-n)^4\theta^4)$$

## 6. RoPE变体的潜在应用和优势

1. **可调节的感受野**：通过调整 $\alpha$ 和 $\rho$，模型可以在局部精确性和全局上下文之间取得平衡。
2. **增强的表达能力**：复值表示允许模型捕捉更复杂的attention模式。
3. **多尺度表示**：不同的 $\theta_j$ 允许模型同时捕捉不同尺度的依赖关系。
4. **任务适应性**：可学习参数使模型能够根据具体任务需求调整位置编码特性。

## 7. 结论与未来方向

RoPE及其变体为位置编码提供了一个强大而灵活的框架。通过引入可调节的参数，RoPE变体有潜力在保持原始RoPE优势的同时，提供更精细的位置编码控制。

未来的研究方向可能包括：
1. 探索自动调整 $\alpha$ 和 $\rho$ 的方法。
2. 研究RoPE在不同类型任务中的最佳配置。
3. 将RoPE的思想扩展到其他模态，如图像或时间序列数据。
4. 深入研究Hilbert变换对在注意力机制中的作用及其理论意义。