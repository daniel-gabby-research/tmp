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

for any disjoint vertex subsets $A, B, and C$ such that $C$ separates $A$ and $B$,
$$
X_A \perp X_B \mid X_C
$$

- $X_A$ are the random variables $X_j$ with $j \in A$.
- $C$ separates $A$ and $B$ means that there is no path from $A$ to $B$ that does not pass through $C$.

> The blue node is independent of the red nodes, given the white nodes.
>![alt text](image-1.png)