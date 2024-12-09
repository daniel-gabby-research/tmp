---
layout: default
parent: S&DS 665 Intermediate Machine Learning
grand_parent: Courses at Yale
title: "Discrete Data Graphs and Graph Neural Networks"
nav_order: 2
discuss: true
math: katex
---

# Discrete Data Graphs and Graph Neural Networks

## 1. Markov Property


![graph](image.png)

This encodes the independence relation

$$
X \perp Z \mid Y
$$

which means that X and Z are independent conditioned on Y.



A probability distribution $P$ satisfies the *global Markov property* with respect to a graph $G$ if:

for any disjoint vertex subsets $A$, $B$, and $C$ such that $C$ separates $A$ and $B$,

$$
X_A \perp X_B \mid X_C
$$

- $X_A$ are the random variables $X_j$ with $j \in A$.
- $C$ separates $A$ and $B$ means that there is no path from $A$ to $B$ that does not pass through $C$.

> The blue node is independent of the red nodes, given the white nodes.
>![alt text](image-1.png)

## 2. Graph Estimation

- A graph $G$ represents the class of distributions, $\mathcal{P}(G)$, the distributions that are Markov with respect to $G$.

- Graph estimation: Given $n$ samples $X_1, \ldots, X_n \sim P$, estimate the graph $G$.

**Theorem 2.1 (Hammersley, Clifford, Besag)** A positive distribution over random variables $X_1, \ldots, X_p$ satisfies the Markov properties of graph $G$ if and only if it can be represented as

$$
p(X) \propto \prod_{c \in \mathcal{C}} \psi_c(X_c)
$$

where $\mathcal{C}$ is the set of cliques in the graph $G$.

> ![alt text](image-2.png)

