---
layout: default
parent: Positional Embedding
grand_parent: ArXiv
title: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
nav_order: 1
discuss: true
math: katex
---

# Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation

Paper link: [https://arxiv.org/abs/2108.12409](https://arxiv.org/abs/2108.12409)

## Extending ALiBi with a Local Receptive Field

### 1. Motivation for Introducing a Receptive Field into ALiBi

**Attention with Linear Biases (ALiBi)** is a positional encoding method that adds a linear bias to the attention scores based on the relative positions of tokens. This approach encourages the model to focus more on nearby tokens, which is beneficial for capturing local dependencies. However, the linear nature of the bias means that it applies uniformly across all positions, which may not be optimal for tasks that require a balance between local and global context.

To address this limitation, we can introduce a **local receptive field** into ALiBi, allowing the model to adjust the emphasis on local versus global information. This concept is inspired by the receptive field parameter introduced in the RoPE variant, which provides more control over the positional encoding.

### 2. Revisiting the ALiBi Bias Term

In the standard ALiBi method, the attention scores are computed as:

$$
A_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}} + B_{ij}
$$

Where:

- $ Q_i $ and $ K_j $ are the query and key vectors at positions $ i $ and $ j $.
- $ d $ is the dimension of the key/query vectors.
- $ B_{ij} $ is the ALiBi bias term, defined as:

  $$
  B_{ij} = m_h (i - j)
  $$

- $ m_h $ is the negative slope for head $ h $:

  $$
  m_h = -2^{-8(\frac{h}{H} - 1)}
  $$

- $ H $ is the total number of attention heads.

This bias term decreases linearly with the increase in relative distance $ i - j $, promoting attention to closer positions.

### 3. Introducing a Receptive Field Parameter

To incorporate a receptive field into ALiBi, we can modify the bias term to include a decay function that diminishes the bias effect as the distance between tokens increases beyond a certain threshold. One way to achieve this is by introducing an exponential decay controlled by a **receptive field parameter** $ \rho $:

$$
B_{ij} = m_h (i - j) \cdot f_\rho(i - j)
$$

Where $ f_\rho(i - j) $ is a decay function defined as:

$$
f_\rho(i - j) = \exp\left(-\frac{\vert i - j \vert}{\rho}\right)
$$

- $ \rho $ controls the rate of decay:
  - Smaller $ \rho $: Faster decay, more focus on local context.
  - Larger $ \rho $: Slower decay, allowing attention to consider longer-range dependencies.

### 4. Modified Attention Scores with Receptive Field

The attention scores now become:

$$
A_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}} + m_h (i - j) \exp\left(-\frac{\vert i - j \vert}{\rho}\right)
$$

This modification ensures that the bias term's influence reduces for tokens beyond the receptive field determined by $ \rho $.

### 5. Analysis of the Modified ALiBi Bias

#### 5.1 Effect of the Receptive Field Parameter $ \rho $

- **Local Attention**: With a small $ \rho $, the bias term becomes negligible for larger $ \vert i - j \vert $, making the model focus on nearby tokens.
- **Global Attention**: A large $ \rho $ maintains the bias term's influence over longer distances, enabling the model to capture global dependencies.

#### 5.2 Comparison with Original ALiBi

The original ALiBi can be viewed as a special case where $ \rho \to \infty $, resulting in $ \exp\left(-\frac{\vert i - j \vert}{\rho}\right) \approx 1 $, and the bias term does not decay with distance.

### 6. Alternative Decay Functions

Depending on the desired receptive field characteristics, other decay functions can be used:

#### 6.1 Gaussian Decay

$$
f_\rho(i - j) = \exp\left(-\frac{(i - j)^2}{2 \rho^2}\right)
$$

- Provides a symmetric decay around zero, focusing strongly on immediate neighbors.

#### 6.2 Reciprocal Decay

$$
f_\rho(i - j) = \frac{\rho}{\rho + \vert i - j \vert}
$$

- Offers a slower decay than the exponential function, balancing between local and global attention.

### 7. Learnable Receptive Field Parameter

To make the model adaptable, $ \rho $ can be treated as a learnable parameter:

$$
\rho = \text{softplus}(\theta_\rho)
$$

- The softplus function ensures $ \rho $ remains positive.
- $ \theta_\rho $ is learned during training, allowing the model to determine the optimal receptive field size.

### 8. Head-Specific Receptive Fields

Similar to the head-specific slopes $ m_h $, we can define head-specific receptive fields $ \rho_h $:

$$
B_{ij} = m_h (i - j) \exp\left(-\frac{\vert i - j \vert}{\rho_h}\right)
$$

- Different heads can focus on different ranges, enhancing the model's capacity to capture various positional dependencies.

### 9. Theoretical Justification

#### 9.1 Capturing Local Dependencies

Introducing a decay function aligns with the intuition that dependencies between tokens diminish with distance. By controlling the decay rate, the model can better capture local patterns essential for tasks like language modeling and machine translation.

#### 9.2 Flexibility and Adaptability

- **Task-Specific Tuning**: Learnable $ \rho $ allows the model to adjust the receptive field based on the task's requirements.
- **Multi-Scale Attention**: Head-specific $ \rho_h $ enables the model to simultaneously capture local and global patterns.

### 10. Potential Advantages

1. **Enhanced Locality**: Improves the model's ability to focus on relevant local context.
2. **Reduced Noise**: Diminishes the influence of distant tokens that may introduce irrelevant information.
3. **Adaptability**: Learns optimal receptive fields for different datasets and tasks.
4. **Complementarity**: Can be combined with other positional encoding methods for further performance gains.

### 11. Implementation Considerations

#### 11.1 Computational Efficiency

- The decay function $ \exp\left(-\frac{\vert i - j \vert}{\rho}\right) $ is computationally inexpensive and can be precomputed.
- Efficient implementation ensures minimal impact on training and inference speed.

#### 11.2 Initialization and Regularization

- Proper initialization of $ \theta_\rho $ is crucial for stable training.
- Regularization techniques may be applied to prevent overfitting.

### 12. Experimental Validation

To validate the effectiveness of the modified ALiBi with a receptive field, the following steps can be undertaken:

- **Benchmarking**: Compare performance on tasks sensitive to local and global dependencies.
- **Ablation Studies**: Analyze the impact of different decay functions and receptive field sizes.
- **Visualization**: Examine attention patterns to confirm that the model focuses appropriately on relevant positions.

### 13. Conclusion and Future Directions

By incorporating a local receptive field into ALiBi, we enhance the model's ability to control the focus between local and global contexts. This modification provides a flexible and powerful approach to positional encoding in transformer models.

**Future research may explore**:

- **Dynamic Receptive Fields**: Adjusting $ \rho $ based on the input sequence or layer depth.
- **Hybrid Positional Encodings**: Combining modified ALiBi with RoPE or other positional encodings.
- **Theoretical Analysis**: Further study of the mathematical properties and potential benefits.