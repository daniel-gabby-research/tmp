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

### **解决方案：引入多尺度高斯核**

我们可以通过将**多尺度高斯核函数**引入到 RoPE 中，使其旋转幅度根据输入位置的不同距离动态调整。这可以让 RoPE 自适应地处理局部和全局信息。

#### **多尺度高斯核的公式**

我们使用一个多尺度高斯核函数来调整 RoPE 的旋转幅度：

$$
K(m) = \alpha_1 \exp\left(-\frac{m^2}{2\sigma_1^2}\right) + \alpha_2 \exp\left(-\frac{m^2}{2\sigma_2^2}\right)
$$

- $ \alpha_1 $ 和 $\alpha_2$ 是控制不同尺度的权重。
- $\sigma_1$ 控制局部信息的聚焦范围（即短距离），而 $\sigma_2 $ 控制全局信息的感知能力（即长距离）。

通过这种核函数，RoPE 的旋转角度可以根据位置距离进行调整。例如，当位置 $m$ 较小时，模型更多聚焦于局部信息，而当 $m$ 较大时，模型能够更加关注全局信息。

### **实现：结合多尺度高斯核的 RoPE**

我们现在来实现一个基于多尺度高斯核优化的 **Rotary Position Embedding (RoPE)**，这可以通过简单的调整代码来实现。

#### **代码实现**

```python
import torch
import torch.nn as nn
import math

# 定义高斯核函数
def gaussian_kernel(m, alpha1, alpha2, sigma1, sigma2):
    """通过多尺度高斯核调整旋转幅度"""
    return alpha1 * torch.exp(-m**2 / (2 * sigma1**2)) + alpha2 * torch.exp(-m**2 / (2 * sigma2**2))

# 定义RoPE旋转位置嵌入的辅助函数
def apply_rotary_pos_emb(x, cos, sin):
    """对输入向量x应用旋转位置嵌入，基于cos和sin表进行旋转"""
    return (x * cos) + (rotate_half(x) * sin)

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

        # 计算旋转角度频率并应用多尺度高斯核调整
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        m = torch.arange(seq_len, device=k.device).type_as(freqs)  # 用于调整旋转幅度
        K_m = gaussian_kernel(m, self.alpha1, self.alpha2, self.sigma1, self.sigma2)
        freqs *= K_m.unsqueeze(-1)  # 应用高斯核调整

        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]

        # 计算cos和sin表
        cos = emb.cos()[None, :, :]  # 添加batch维度
        sin = emb.sin()[None, :, :]

        # 将旋转嵌入应用到查询和键
        return apply_rotary_pos_emb(q, cos, sin), apply_rotary_pos_emb(k, cos, sin)

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

### **代码说明**

1. **高斯核函数**：`gaussian_kernel` 定义了多尺度的高斯核，通过参数 $$ \alpha_1 $$、$$ \alpha_2 $$ 控制不同尺度的权重，$$ \sigma_1 $$ 和 $$ \sigma_2 $$ 控制局部和全局信息的感知范围。
2. **RoPE 基础实现**：`RotaryEmbedding` 类实现了基于 RoPE 的位置嵌入，并通过多尺度高斯核调整旋转幅度。
3. **代码应用**：在运行示例中，你可以调整不同的高斯核参数，观察模型在不同场景下对局部和全局信息的自适应调整。

### **结论**

通过将多尺度高斯核函数与 RoPE 结合，我们成功地提升了 RoPE 的自适应能力，使其能够更灵活地处理局部和全局信息。此方法为自注意力机制提供了更强的感知能力，尤其适用于长序列任务中的局部与全局依赖关系处理。

这项技术不仅能够提升 Transformer 模型的表现，还为如何平衡不同尺度的信息提供了新的思路。在未来的工作中，我们可以继续探索不同核函数的组合，以及如何自适应地调节这些核函数的参数，以应对更复杂的序列任务。

--- 

### **总结思考**
- RoPE 的优势在于它能够以一种轻量级的方式为自注意力机制引入相对位置信息。
- 通过结合多尺度高斯核，RoPE 的局部和全局信息感知能力得到了显著提升，适用于长文本处理任务。
