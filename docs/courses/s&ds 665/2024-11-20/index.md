---
layout: default
parent: S&DS 665 Intermediate Machine Learning
grand_parent: Courses at Yale
title: "Transformers and LLM Scaling"
nav_order: 10
discuss: true
math: katex
---

# Transformers and LLM Scaling

## 1. Preliminaries

### 1.1 Sequence to Sequence Models

Important in translation.Uses two RNNs (GRUs or LSTMs): Encoder and Decoder

![seq2seq](image.png)

The goal of Seq2seq is to estimate the conditional probability

$$
p(y_1, \ldots, y_T \mid x_1, \ldots, x_S)
$$

- Encoder RNN computes the fixed dimensional representation $h(x_1:S)$
- Decoder RNN then computes

$$
\prod_{t=1}^T p(y_t \mid y_{<t}, h(x_1:S))
$$


## 2. Attention

On each step of decoding, directly connect to the encoder, and focus on a particular part of the source sequence.

![attention](image-1.png)

### 2.1 Attention in terms of query/key/values

Basic idea behind attention mechanisms:

- Neural networks so far: Hidden activations are linear combinations of input activations, followed by nonlinearity:

$$
h = \varphi(Wu)
$$

- A more flexible model:
    - We have a set of $m$ feature vectors or values $V \in \mathbb{R}^{m \times v}$
    - The model dynamically chooses which to use based on how similar a query vector $q \in \mathbb{R}^q$ is to a set of $m$ keys $K \in \mathbb{R}^{m \times k}$.
    - If $q$ is most similar to key $i$, then we use value (feature) $v_i$.

![attention](image-2.png)

- We can write the attention mechanism as:
$$
\begin{aligned}
\text{Attn}(q, \{(k_1, v_1), \ldots, (k_m, v_m)\}) &= \text{Attn}(q, (k_1:m, v_1:m)) \\
&= \sum_{i=1}^m w_i(q, k_1:m) v_i
\end{aligned}
$$

where weights $w_i$ are softmax of attention scores $a(q, k)$:

$$
w_i(q, k_1:m) = \frac{\exp(a(q, k_i))}{\sum_{j=1}^m \exp(a(q, k_j))}
$$

### 2.2 Kernel regression as attention
Recall the kernel regression estimator:

$$
\hat{m}(x) = \sum_{i=1}^n w(x, x_1:n) y_i
$$

Here the attention scores (for Gaussian kernel) are

$$
a(x, x_i) = - \frac{1}{2h^2} \lVert x - x_i \rVert^2
$$

- query: test point $x$
- keys: data $x_1, \ldots, x_n$
- values: responses $y_1, \ldots, y_n$

### 2.3 Attention in Transformers

Note that if $x_i$ and $x$ have a fixed norm then the attention scores arejust scaled dot products:

$$
a(x, x_i) = \frac{1}{h^2} x^T x_i
$$

If query and keys have same dimension $d$, dot product attention is

$$
a(q, k) = \frac{q^T k}{\sqrt{d}} \in \mathbb{R}
$$

and

$$
\text{Attn}(Q, K, V) = \text{Softmax} \left( \frac{Q K^T}{\sqrt{d}} \right) V
$$

## 3. Transformer Architecture

### 3.1 Key Steps in the Transformer

- *Tokenize* the text into a fixed vocabulary
- *Embed* the tokens into high-dimensional vectors
- *Mix* the embeddings in the context using “Attention”
- *Map* the resulting vectors using a neural network

### 3.2 Tokenization

- Tokenizer properties
    - *Reversible*: Can convert from tokens back to original text
    - *General*: Works on arbitrary text, even very different from training
    - *Compressive*: Each token is about 4 bytes
    - *Statistical*: Will break strings into frequently occurring pieces

- The BPE algorithm

![bpe](image-3.png)


- Increasing complexity of attention patterns
![transformer-layers](image-4.png)
![transformer-layers](image-5.png)

### 3.3 Positional encoding

Positional encoding is an interesting way of getting position information into the model. For the $t$-th word in the sequence, components are

$$
PE(t ,2i ) = \sin \left( \frac{t}{C^{2i/d}} \right)
$$

$$
PE(t ,2i +1) = \cos \left( \frac{t}{C^{2i/d}} \right)
$$

where $d$ is the embedding dimension and $C$ is a maximum sequence length

![positional-encoding](image-6.png)

### 3.4 Multi-head attention

Multiple attention vectors are computed, then merged (concatenated)

![multi-head-attention](image-7.png)
![multi-head-attention](image-8.png)

## 4. Generating new tokens

Transformer assigns a score to each token
- Converted to a probability
- Generating next token equivalent to rolling weighted 50,257-sided die

Generation algorithm has two “knobs” to control:

1. Top-$k$ : Restricts to just the top $k$ tokens
    - Low $k$ : Low variability
    - High $k$ : Higher variability

2. Temperature $T$ : Lower temperature favors more likely tokens
    - Low temperature: Low variability
    - High temperature: High variability
    - At temperature $T$, if the scores of the top $k$ tokens are
    $$
    \text{score}_1, \text{score}_2, \ldots, \text{score}_k
    $$
    the weights on the die are proportional to
    $$
    \exp(\text{score}_1 / T), \exp(\text{score}_2 / T), \ldots, \exp(\text{score}_k / T)
    $$

![temperature](image-9.png)


## 5. How large are the models?

Design choices: number of tokens $V$, dimension $d$ of embeddings, and number of Transformer layers $L$

- Embeddings have $d \cdot V$ parameters
    - $d$ numbers for each token, $V$ tokens

- Self-Attention has $\approx 4d^2$ parameters
    - $d \times d$ matrix for queries (across heads)
    - $d \times d$ matrix for keys (across heads)
    - $d \times d$ matrix for values (across heads)
    - $d \times d$ matrix for combining heads

- MLP applied after has $\approx 8d^2$ parameters
    - Two layers
    - Hidden layer has $4 \cdot d$ neurons
    - Output layer has $d$ neurons
    - $4d^2$ weights in each layer; $8d^2$ weights overall

- In each Transformer layer:
    - Attention has about $4d^2$ parameters
    - MLP has about $8d^2$ parameters

- Total trainable parameters: $12d^2 \cdot L + d \cdot V$

![model-size-gpt-2](image-10.png)
![model-size-gpt-3](image-11.png)

- Scaling behavior of LLM models
![scaling-law](image-12.png)
![scaling-law](image-13.png)

> Sutton’s “Bitter Lesson” (2019)
> “The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin.”
> http://www.incompleteideas.net/IncIdeas/BitterLesson.html

