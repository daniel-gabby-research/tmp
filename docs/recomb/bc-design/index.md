---
layout: default
parent: RECOMB
title: "BC-Design: A Biochemistry-Aware Framework for High-Precision Inverse Protein Folding"
nav_order: 1
discuss: true
math: katex
---

# BC-Design: A Biochemistry-Aware Framework for High-Precision Inverse Protein Folding

## Mathematical Notation Details

### Graph and Set Notation
| Symbol | Description | Comment |
|--------|------------|-------------|
| $G(V, E, F_V, F_E)$ | Structure graph with nodes $V$, edges $E$, node features $F_V$, edge features $F_E$ | Clear |
| $V$ | Set of nodes (residues) in the structure graph | Clear |
| $E$ | Set of edges in the structure graph | Clear |
| $V'$ | Augmented node set including aggregator nodes | Clear |
| $E'$ | Augmented edge set including aggregator edges | Clear |
| $V_c$ | Set of structure aggregator nodes | Clear |
| `k` | Number of nearest neighbors in k-NN graph (k=30) | Good |
| $v_i^L$ | Local structure aggregator node | Should not have both superscript and subscript |
| $v^G$ | Global structure aggregator node | Should not have superscript |
| $Q$ | Local coordinate system for residues | Should be $\mathbf{Q} \in \mathbb{R}^{3 \times 3}$ |

- Comment:
  - The annotation of additional nodes and edges should be clear. Otherwise, it will be hard to read.
    - Should not have both superscript and subscript: $v_i^L$
    - Should not have superscript: $v^G$
  - Should be $\mathbf{Q} \in \mathbb{R}^{3 \times 3}$: $Q$. Quaternion system is $\mathbf{q} \in \mathbb{H}$. Need to clarify.

### Point Cloud and Biochemical Features
| Symbol | Description | Comment |
|--------|------------|-------------|
| $b(x, y, z) = \mathbf{b}$ | Continuous mapping of biochemical properties in 3D space | We have $x, y, z$ and $\mathbf{x}$ for coordinates at the same time. |
| $\mathcal{P}_S$ | Surface point cloud | Clear |
| $\mathcal{P}_I$ | Internal point cloud | Clear |
| $\mathcal{P}$ | Combined point cloud | Clear |
| $P_i$ | Point in the point cloud with coordinates and features |
| $\mathbf{x}_i$ | Spatial coordinates of point i | Should use $\mathbf{x}_i \in \mathbb{R}^3$ for coordinates. |
| $\mathbf{b}_i = (h_i, c_i)$ | Biochemical features (hydrophobicity, charge) of point i | Would it be better to use $\mathbf{b}_i = b(\mathbf{x}_i) \in \mathbb{R}^2$? |
| $P^G$ | Global biochemical aggregator point | Same problem as $v^G$. |
| $P_i^L$ | Local biochemical aggregator point | Same problem as $v_i^L$. |

- Comment:
  - What is $\mathbf{b}$? Is it a constant? If not, why we use $b(x, y, z) = \mathbf{b}$ as an implicit definition?
  - Would it be better to use $\mathbf{b}_i = b(\mathbf{x}_i) \in \mathbb{R}^2$?
  - The definition of $P^G$ and $P_i^L$ should be clear. Same as $v^G$ and $v_i^L$.

### Neural Network Components
| Symbol | Description | Comment |
|--------|------------|-------------|
| $H_V$ | Node embeddings | Clear |
| $w_E$ | Edge weights | Why $w_E \in \mathbb{R}^{|E| \times d}$ while we can have $w_E[i,j] \in \mathbb{R}$? |
| $H_{V_c}$ | Structure aggregator node embeddings | Clear |
| $H_{V'}$ | Concatenated node embeddings | Clear |
| $W_E$ | Weight matrix for edges | What is the difference between $w_E$ and $W_E$? |
| $GPE$ | Graph Positional Encodings | Should be $\text{GPE} \in \mathbb{R}^{|V| \times d}$ |
| $Q, K, V$ | Query, Key, Value matrices in attention | Clear |
| $S$ | Attention score matrix | Clear |
| $S'$ | Modified attention score matrix | Clear |

- Comment:
  - What is the difference between $w_E$ and $W_E$?
    - It is not clear to me why we need $W_E$ for $w_E$.
    - It is also not clear why $w_E$ is a vector while it is used as a matrix in the equation.
    - Is the following correct?
$$
W_E[i,j] = \begin{cases}
w_E[i,j] & \text{if } e_{ij} \in E \\
1 & \text{if } e_{ij} \in E_c \\
0 & \text{otherwise},
\end{cases}
$$
  - Should be $\text{GPE} \in \mathbb{R}^{|V| \times d}$.

### Loss Functions
| Symbol | Description | Comment |
|--------|------------|-------------|
| $\mathcal{L}_{\text{CE}}$ | Cross-entropy loss | Clear |
| $\mathcal{L}_{\text{GCL}}$ | Global contrastive loss | Clear |
| $\mathcal{L}_{\text{LCL}}$ | Local contrastive loss | Clear |
| $\mathcal{L}$ | Combined loss function | Clear |
| $\lambda_1, \lambda_2$ | Loss weights (both set to 1) | Clear |

### Key Equations
| Equation | Description | Comment |
|----------|-------------|-------------|
| $S' = S \odot W_E + GPE$ | Attention score modification | Clear |
| $\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_1\mathcal{L}_{\text{GCL}} + \lambda_2\mathcal{L}_{\text{LCL}}$ | Combined loss function | Clear |
| $\mathcal{N}_r(P'_i) = \{P'_j \mid \|P'_i - P'_j\| \leq r\}$ | Multi-scale neighborhood definition | Clear |
| $K_{BC} = \max(1, \lfloor 1400/\|V\| \rfloor)$ | Dynamic connection parameter for BC-Graph | Clear |

### Other Mathematical Notation
| Symbol | LaTeX Code | Description |
|--------|------------|-------------|
| $\odot$ | Hadamard (element-wise) product | Clear |
| $\vert \cdot \vert$ | Absolute value or set cardinality | Clear |
| $\lfloor x \rfloor$ | Floor function | Clear |
| $\in$ | Element of | Clear |
| $\cup$ | Set union | Clear |
| $\leq$ | Less than or equal to | Clear |
| $\times$ | Cross product or multiplication | Clear |
