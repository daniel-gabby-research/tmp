---
layout: default
parent: Positional Embedding
grand_parent: ArXiv
title: "Gaussian Kernel-enhanced Rotary Position Embedding"
nav_order: 4
discuss: true
math: katex
---

## **Gaussian Kernel-enhanced Rotary Position Embedding**

### **引言**

在自然语言处理（NLP）中，Transformer 模型中的自注意力机制依赖于输入序列中的位置信息。这些位置信息通过某种位置编码（Positional Encoding）方式被融入模型，从而使模型能够理解不同序列元素之间的相对位置。Rotary Position Embedding (RoPE) 是一种通过旋转变换嵌入相对位置信息的方式。然而，RoPE 在处理不同长度的依赖关系时，可能无法灵活地平衡局部和全局信息。

为了解决这个问题，我们可以将 **多尺度高斯核函数** 引入到 RoPE 中，使其在处理短距离（局部）和长距离（全局）依赖时能够更加自适应，从而提升模型对不同信息范围的感知能力。

### **背景：什么是 RoPE？**

RoPE 的设计思想是通过旋转操作，将位置信息嵌入到查询（query）和键（key）向量的不同维度中。与传统的正弦/余弦位置编码相比，RoPE 通过旋转嵌入方式保留了相对位置信息，因此特别适用于自注意力机制。

#### **RoPE 的基本公式**

在 RoPE 中，每个查询和键向量的偶维度和奇维度分别应用以下旋转变换：

$$
f(x, m) = [(x_0 + ix_1) e^{im\theta_0}, (x_2 + ix_3) e^{im\theta_1}, \dots, (x_{d-2} + ix_{d-1}) e^{im\theta_{d/2-1}}]
$$

其中：
- $x$ 是输入向量，$m$ 是位置信息，$d$ 是嵌入维度。
- $\theta_j = 10000^{-2j/d}$，用于控制旋转角度。

### **问题：RoPE 对局部与全局信息的局限性**

虽然 RoPE 保留了相对位置信息，但其旋转变换在处理不同距离的依赖时是**固定的**，无法根据序列的距离自适应地调整聚焦点。这种固定的旋转变换可能在局部信息（例如短距离依赖）和全局信息（长距离依赖）之间无法做到良好的平衡。

**问题要点**：
1. **局部信息**需要模型更加精细地捕捉短距离依赖，而 RoPE 默认的旋转频率可能不足以应对。
2. **全局信息**则需要模型在长距离依赖中保留上下文，但固定的旋转幅度可能导致对全局信息的关注不足。

我们可以把高斯核 $K(m)$ 作为 **旋转嵌入操作的整体缩放因子**，作用在整个旋转嵌入矩阵外。这意味着高斯核控制 RoPE 在不同相对位置 $r$ 的感知能力。

因此，RoPE 和多尺度高斯核结合后的完整公式应该是：

$$
f(x, m) = K(m) \cdot \left[ \left( x_{2i}, x_{2i+1} \right) \cdot \begin{pmatrix} \cos(m \cdot \theta_i) & -\sin(m \cdot \theta_i) \\ \sin(m \cdot \theta_i) & \cos(m \cdot \theta_i) \end{pmatrix} \right]
$$

### **公式解释**

- **$m$**：表示查询向量 $q$ 和键向量 $k$ 的绝对位置。
- **旋转矩阵**：$\begin{pmatrix} \cos(m \cdot \theta_i) & -\sin(m \cdot \theta_i) \\ \sin(m \cdot \theta_i) & \cos(m \cdot \theta_i) \end{pmatrix}$ 负责对输入向量的偶数维度和奇数维度进行旋转操作，编码绝对位置信息。
- **高斯核 $K(m)$**：乘在旋转操作的外部，作用是动态调整整个旋转操作的幅度，根据绝对位置的距离来控制局部和全局信息的感知。

### **多尺度高斯核的定义**

高斯核 $K(m)$ 的定义保持不变，用于根据绝对位置 $m$ 动态调整 RoPE 的旋转幅度：

$$
K(m) = \alpha_1 \exp\left(-\frac{m^2}{2\sigma_1^2}\right) + \alpha_2 \exp\left(-\frac{m^2}{2\sigma_2^2}\right)
$$

- **$\alpha_1$** 和 **$\alpha_2$**：分别控制不同尺度的权重，分别对应局部和全局信息的感知。
- **$\sigma_1$** 和 **$\sigma_2$**：控制局部和全局信息的感知范围。

### **解释如何工作**

1. **旋转矩阵**：使用标准的 RoPE 旋转矩阵，基于绝对位置 $m$ 和频率 $\theta_i$ 对偶数维度和奇数维度的元素进行旋转，嵌入绝对位置信息。

2. **高斯核 $K(m)$**：高斯核通过控制 RoPE 的旋转频率，起到调整模型感知范围的作用。当 $m$ 较小时，高斯核将加强局部信息的处理能力；当 $m$ 较大时，高斯核将增强对全局信息的处理能力。

### **为什么这样处理更合理**

1. **高斯核调节整体旋转操作**：把高斯核 $K(m)$ 作用在旋转嵌入的外部，相当于在不同绝对位置下整体调节模型对局部和全局信息的聚焦程度，而不是直接影响旋转频率本身。这使得模型在处理不同距离的依赖时更加灵活。

2. **局部和全局信息自适应调节**：通过高斯核的两项 $\alpha_1$ 和 $\alpha_2$，我们可以为局部和全局信息定义不同的感知尺度，进而让模型在不同的绝对位置下自动切换。

### **实现 Python 代码**

根据这个新的公式，我们可以更新之前的 Python 实现，将高斯核 $K(r)$ 作用在 `cos` 和 `sin` 的外部。

```python
import torch
import torch.nn as nn
import math

# 定义高斯核函数
def gaussian_kernel(r, alpha1, alpha2, sigma1, sigma2):
    """通过多尺度高斯核调整旋转幅度"""
    return alpha1 * torch.exp(-r**2 / (2 * sigma1**2)) + alpha2 * torch.exp(-r**2 / (2 * sigma2**2))

# 定义RoPE旋转位置嵌入的辅助函数
def apply_rotary_pos_emb(x, cos, sin, K_r):
    """对输入向量x应用旋转位置嵌入，基于cos和sin表进行旋转，并乘以高斯核"""
    return K_r * ((x * cos) + (rotate_half(x) * sin))

# 用于处理偶数维度与奇数维度的旋转
def rotate_half(x):
    """旋转一半维度，用于RoPE的奇偶维度交换"""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

# Rotary Position Embedding 实现
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, alpha1: float, alpha2: float, sigma1: float, sigma2: float):
        super().__init__()
        # 生成反频率（inverse frequency）
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # 初始化多尺度高斯核参数
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        # 获取序列长度，生成cos和sin表
        seq_len = k.shape[-2]
        t = torch.arange(seq_len, device=k.device).type_as(self.inv_freq)

        # 计算旋转角度频率
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]

        # 计算cos和sin表
        cos = emb.cos()[None, :, :]  # 添加batch维度
        sin = emb.sin()[None, :, :]

        # 计算相对位置 r
        r = torch.arange(seq_len, device=k.device).type_as(freqs)

        # 使用高斯核计算K(r)
        K_r = gaussian_kernel(r, self.alpha1, self.alpha2, self.sigma1, self.sigma2)

        # 将旋转嵌入应用到查询和键，并乘以高斯核
        return apply_rotary_pos_emb(q, cos, sin, K_r), apply_rotary_pos_emb(k, cos, sin, K_r)

# 示例使用
if __name__ == "__main__":
    # 假设嵌入维度为64，输入的序列长度为128
    dim = 64
    seq_len = 128
    batch_size = 2

    # 高斯核参数
    alpha1 = 0.7
    alpha2 = 0.3
    sigma1 = 5.0  # 局部尺度
    sigma2 = 20.0  # 全局尺度

    # 随机生成查询和键向量
    query = torch.randn(batch_size, seq_len, dim)
    key = torch.randn(batch_size, seq_len, dim)

    # 实例化RotaryEmbedding并应用到查询和键
    rotary_emb = RotaryEmbedding(dim, alpha1, alpha2, sigma1, sigma2)
    query_rotated, key_rotated = rotary_emb(query, key)

    print("查询嵌入后的结果：", query_rotated)
    print("键嵌入后的结果：", key_rotated)
```

总结

通过将高斯核 $K(m)$ 乘在 `cos` 和 `sin` 的外面，我们可以在旋转操作之后调整整体的旋转幅度，从而让 RoPE 更加灵活地处理局部和全局信息。这种方法让模型能够根据相对位置信息自动调节感知范围，增强了模型在不同位置依赖下的表现。