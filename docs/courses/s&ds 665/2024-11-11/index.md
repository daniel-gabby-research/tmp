---
layout: default
parent: S&DS 665 Intermediate Machine Learning
grand_parent: Courses at Yale
title: "Classical Sequence Models and Recurrent Neural Networks"
nav_order: 10
discuss: true
math: katex
---

# Classical Sequence Models and Recurrent Neural Networks

## 1. Hidden Markov Models

The graphical model looks like this:

![hmm](image.png)

Here $x_t$ is observable (word) at time $t$ and $s_t$ is unobserved state.


- Probability of word sequence:

$$p(x_1, \ldots, x_n) = \sum_{s_1, \ldots, s_n} \prod_{t=1}^n p(s_t | s_{t-1}) p(x_t | s_t)$$

$\leadsto$ Can be efficiently computed using the “forward-backward algorithm” which is a type of dynamic programming algorithm.

- For topic models:

$$p(x_1, \ldots, x_n) = \int p(\theta | \alpha) \prod_{t=1}^n p(s_t | \theta) p(x_t | s_t) d\theta$$

$\leadsto$ Word order doesn’t matter

- Estimation

> Algorithm:
> 
> 1. Initialize state probabilities $p(s_t \mid x_1, \ldots, x_t)$
> 2. Iterate until convergence:
> 3. Use forward-backward algorithm to update $p(s_t \mid x_1, \ldots, x_t)$
>    - Forward:
>       $$p(s_t \mid x_1, \ldots, x_t) = p(s_t \mid s_{t-1}) p(x_t \mid s_t)$$
>    - Backward:
>       $$p(x_{t+1}, \ldots, x_n \mid s_t) = \sum_{s_{t+1}} p(s_{t+1} \mid s_t) p(x_{t+1}, \ldots, x_n \mid s_{t+1})$$
> 4. Return final state probabilities
> 5. Total probability of word sequence: $p(x_1, \ldots, x_n) = \sum_{s_1, \ldots, s_n} \prod_{t=1}^n p(s_t \mid s_{t-1}) p(x_t \mid s_t)$

![forward-backward](image-1.png)

## 2. Kalman Filters


### 2.1 Gaussian Kalman Filter

$$S_t \mid S_{t-1} \sim N(A S_{t-1}, \Gamma)$$
$$X_t \mid S_t \sim N(B S_t, \Sigma)$$

where $A$ and $B$ are matrices for the means; $\Gamma$ and $\Sigma$ for the covariances. Everything is linear.

- State is distributed (real vector)
- State evolves stochastically
- Schur complements and forward-backward used to compute conditional probabilities; similar to Gaussian processes

### 2.2 Discrete Kalman Filter

We can also work with state space models for discrete data, like documents.

$$S_t \mid S_{t-1} \sim N(A S_{t-1}, \Gamma)$$
$$W_t \mid S_t \sim \text{Softmax}(B S_t)$$

where

$$\text{Softmax}(B S_t)_j = \frac{\exp(B^T_j S_t)}{\sum_k \exp(B^T_k S_t)}$$

## 3. Recurrent Neural Networks

- Similar to hidden Markov models and Kalman filters, but *the hidden layer is not stochastic*.

- The state is *distributed* (a vector), as in Kalman filters, not categorical as for HMMs.

![rnn](image-2.png)

## 3.1 RNN

This means
$$
\begin{aligned}
h_t &= \tanh (W_{hh} h_{t-1} + W_{xh} x_t) \\
y_t &= W_{hy} h_t \\
x_{t+1} &\sim \text{Multinomial}(\pi(y_t))
\end{aligned}
$$
where $\pi(\cdot)$ is the soft-max function.

In this illustration, $x_t$ is the “1-hot” representation of a character, $W_{xh} \in \mathbb{R}^{3 \times 4}$, $W_{hh} \in \mathbb{R}^{3 \times 3}$ and $W_{hy} \in \mathbb{R}^{4 \times 3}$.

**Loss function:**

The model is trained to assign high probability to the word (or
character) that appears next:

$$
\text{Loss} = -\frac{1}{T} \sum_{t=1}^T \log p(x_t | h_t)
$$

If the last layer assigns a score to word $v$ as

$$
f(v, h_t) = \beta^T_v h_t
$$

then this is given by

$$
\text{Loss} = -\frac{1}{T} \sum_{t=1}^T \log p(x_t | h_t)
$$

## 3.2 Memory circuits

A simpler alternative to the LSTM circuit is called the *Gated Recurrent Unit (GRU)*.

- In principle the state $h_t$ can carry information from far in the past.
- In practice, the gradients vanish (or explode) so this doesn’t really happen.
- We need other mechanisms to “remember” information from far away and use it to predict future words.
- Both LSTMs and GRUs have longer-range dependencies than vanilla RNNs.

> *Hadamard product:* We’ll need to use pointwise products. This is given the fancy name “Hadamard product” and written $\odot$:
>
> $$
> (a \odot b)_i = a_i b_i
> $$

### 3.2.1 Gated Recurrent Unit (GRU)

- **High level idea:**
  - Learn when to update hidden state to “remember” important pieces of information
  - Keep them in memory until they are used
  - Reset or “forget” this information when no longer useful

- GRUs make use of “gates” denoted by $\Gamma$ (Greek G for “Gate”)
  - $\Gamma = 1$: “the gate is open” and information flows through
  - $\Gamma = 0$: “the gate is closed” and information is blocked

- Two types of gates are used:
  - $\Gamma_u$: When open, information from long-term memory is propagated.
  - When closed, information from local state is used.
  - $\Gamma_r$: When closed, the local state is reset. When open, the state is updated as in a “vanilla” RNN.

- GRU state update:
![gru-state-update](image-3.png)
    - $c_t$ is the “candidate state” computed using the usual “vanilla RNN” state, after possibly resetting some components.
    - When the long-term memory gate is open ($\Gamma_u = 1$), the information gets sent through directly. This deals with vanishing gradients.
    - The gates are multi-dimensional, applied componentwise
    - Prediction of the next word is made using $h_t$.

![gru-overview](image-4.png)

### 3.2.2 LSTM

A variant called “Long Short-Term Memory” RNNs has a special context/hidden layer that “includes” or “forgets” information from the past.

- Forget:
  $$F_t = \sigma (W_{fh} h_{t-1} + W_{fx} x_t + b_f)$$
- Include:
  $$I_t = \sigma (W_{ih} h_{t-1} + W_{ix} x_t + b_i)$$
- “Memory cell” or “context” $c_t$ evolves according to
  $$c_t = F_t \odot c_{t-1} + I_t \odot \tilde{c}_t$$
  $$\tilde{c}_t = \tanh (W_{ch} h_{t-1} + W_{cx} x_t + b_c)$$

![lstm-overview](image-5.png)
