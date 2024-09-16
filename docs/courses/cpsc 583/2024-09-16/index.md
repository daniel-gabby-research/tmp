---
layout: default
parent: CPSC 583 Deep Learning on Graph-Structured Data
grand_parent: Courses at Yale
title: "Inductive Representation Learning on Large Graphs"
nav_order: 3
discuss: true
math: katex
render_with_liquid: false
---

# Inductive Representation Learning on Large Graphs

## Introduction

Graphs are fundamental data structures that model relationships between entities in various domains such as social networks, biological systems, and recommendation engines. Learning meaningful representations (embeddings) of nodes within these graphs is crucial for tasks like node classification, link prediction, and clustering. Traditional approaches often struggle with scalability and generalization, especially when dealing with large, dynamic graphs where new nodes continuously appear.

**GraphSAGE** (Graph SAmple and aggreGatE) is an inductive framework proposed by Hamilton, Ying, and Leskovec to address these challenges. Unlike transductive methods that require retraining when new nodes are added, GraphSAGE learns a function that generates embeddings by aggregating feature information from a node's local neighborhood. This allows for efficient and scalable representation learning on large, evolving graphs.

---

## The Challenge of Inductive Representation Learning

In many real-world applications, graphs are not static. New nodes and edges are frequently introduced, rendering transductive methods inadequate due to their inability to generalize without retraining. The key challenges are:

1. **Scalability**: Handling large graphs with millions of nodes and edges is computationally intensive.
2. **Generalization**: Learning representations that can be applied to unseen nodes without retraining the entire model.
3. **Feature Utilization**: Effectively incorporating node features (attributes) into the embedding process.

GraphSAGE tackles these issues by employing an inductive approach that leverages node feature information and neighborhood structures to generate embeddings on-the-fly.

---

## Mathematical Formulation of GraphSAGE

### Notation and Definitions

Let $ G = (V, E) $ be an undirected graph where $ V $ is the set of nodes and $ E $ is the set of edges. Each node $ v \in V $ is associated with a feature vector $ \mathbf{x}_v \in \mathbb{R}^F $, where $ F $ is the dimensionality of the input features.

### Neighborhood Sampling

To ensure scalability, GraphSAGE samples a fixed-size set of neighbors for each node. Let $ \mathcal{N}(v) $ denote the set of neighbors of node $ v $. For computational efficiency, we sample a subset $ \mathcal{N}_k(v) \subseteq \mathcal{N}(v) $ with a fixed size at each layer $ k $.

### Aggregation Function

At each layer $ k $, we compute the hidden representation $ \mathbf{h}_v^k $ of node $ v $ by aggregating the representations of its sampled neighbors from the previous layer $ k-1 $:

$$
\mathbf{h}_v^k = \sigma \left( \mathbf{W}^k \cdot \text{AGGREGATE}_k \left( \left\{ \mathbf{h}_u^{k-1}, \forall u \in \mathcal{N}_k(v) \right\} \right) + \mathbf{b}^k \right),
$$

where:

- $ \sigma $ is an activation function (e.g., ReLU).
- $ \mathbf{W}^k $ and $ \mathbf{b}^k $ are the weight matrix and bias vector for layer $ k $.
- $ \text{AGGREGATE}_k $ is a differentiable aggregation function.

### Initialization

The initial hidden representations are set to the input features:

$$
\mathbf{h}_v^0 = \mathbf{x}_v.
$$

### Aggregation Functions

Several aggregation functions can be employed:

1. **Mean Aggregator**:

$$
\text{AGGREGATE}_\text{mean} = \frac{1}{|\mathcal{N}_k(v)|} \sum_{u \in \mathcal{N}_k(v)} \mathbf{h}_u^{k-1}.
$$

2. **LSTM Aggregator**:

An LSTM is applied to the sequence of neighbor embeddings.

3. **Pooling Aggregator**:

$$
\text{AGGREGATE}_\text{pool} = \text{max} \left( \left\{ \sigma \left( \mathbf{W}_\text{pool} \mathbf{h}_u^{k-1} + \mathbf{b}_\text{pool} \right), \forall u \in \mathcal{N}_k(v) \right\} \right).
$$

### Multi-hop Aggregation

By stacking $ K $ layers, GraphSAGE captures information from $ K $-hop neighborhoods. The final embedding of node $ v $ is:

$$
\mathbf{z}_v = \mathbf{h}_v^K.
$$

---

## Training Objectives

GraphSAGE can be trained under both supervised and unsupervised settings.

### Supervised Learning

In supervised node classification, we optimize the cross-entropy loss:

$$
\mathcal{L} = -\sum_{v \in V_\text{train}} \sum_{i=1}^{C} y_{v,i} \log \left( \text{softmax}(\mathbf{z}_v)_i \right),
$$

where:

- $ V_\text{train} $ is the set of training nodes.
- $ y_{v,i} $ is the binary indicator (0 or 1) if class label $ i $ is the correct classification for node $ v $.
- $ C $ is the number of classes.

### Unsupervised Learning

In the unsupervised setting, GraphSAGE employs a **negative sampling** strategy inspired by skip-gram models. The objective is to maximize the similarity between a node and its neighbors while minimizing it with random nodes:

$$
\mathcal{L} = -\sum_{(v,u) \in E} \log \left( \sigma \left( \mathbf{z}_v^\top \mathbf{z}_u \right) \right) - \sum_{(v,u') \notin E} \log \left( \sigma \left( -\mathbf{z}_v^\top \mathbf{z}_{u'} \right) \right),
$$

where $ \sigma $ is the sigmoid function.

---

## Advantages of GraphSAGE

1. **Inductive Capability**: By learning aggregation functions rather than specific embeddings, GraphSAGE can generate embeddings for unseen nodes.

2. **Scalability**: Fixed-size neighborhood sampling reduces computational overhead, making it suitable for large graphs.

3. **Flexibility in Aggregation**: Different aggregation functions allow customization based on the specific characteristics of the data.

4. **Feature Incorporation**: Effectively utilizes node features, which is particularly beneficial when structural information is insufficient.

---

## Empirical Results

### Datasets

- **Citation Networks**: Predicting paper categories in citation graphs like Cora and PubMed.
- **Reddit Posts**: Classifying Reddit posts into communities.
- **Protein-Protein Interaction (PPI)**: Multi-label node classification across various graphs representing different human tissues.

### Performance Metrics

- **Micro-averaged F1 Score**: Used for multi-label classification tasks.
- **Accuracy**: For single-label classification.

![alt text](image.png)

![alt text](image-1.png)

### Observations

1. **Improved Accuracy**: GraphSAGE outperforms baseline methods like DeepWalk and node2vec, especially in inductive settings.

2. **Aggregation Function Impact**: The choice of aggregator affects performance. The pooling aggregator often yields the best results due to its ability to capture complex neighborhood features.

3. **Scalability**: Demonstrated the ability to scale to graphs with millions of nodes and edges.

---

## Theoretical Analysis: Expressive Capabilities of GraphSAGE

In addition to its practical advantages, GraphSAGE has been the subject of theoretical analysis to understand its expressive power in capturing graph structural properties. A key question is whether GraphSAGE, which inherently relies on node features and neighborhood aggregation, can learn complex structural patterns such as the **clustering coefficient** of a node—a measure of how close its neighbors are to being a complete graph (i.e., how tightly knit a node's local neighborhood is).

### Clustering Coefficient

For a given node $ v $, the clustering coefficient $ c_v $ is defined as:

$$
c_v = \frac{2T_v}{k_v (k_v - 1)},
$$

where:

- $ T_v $ is the number of triangles through node $ v $.
- $ k_v $ is the degree of node $ v $.

The clustering coefficient quantifies the likelihood that a node's neighbors are also connected to each other, reflecting the local cohesiveness of the graph around $ v $.

### Theorem 1: Approximating Clustering Coefficients with GraphSAGE

**Statement of Theorem 1:**

*Let $ G = (V, E) $ be an undirected graph where each node $ v \in V $ has a feature vector $ \mathbf{x}_v \in U \subset \mathbb{R}^d $, with $ U $ being a compact subset of $ \mathbb{R}^d $. Suppose there exists a fixed positive constant $ C \in \mathbb{R}^+ $ such that $ \| \mathbf{x}_v - \mathbf{x}_{v'} \|_2 > C $ for all pairs of nodes $ v \neq v' $. Then, for any $ \varepsilon > 0 $, there exists a parameter setting $ \Theta^* $ for GraphSAGE (Algorithm 1) such that after $ K = 4 $ iterations, the output $ z_v \in \mathbb{R} $ satisfies:*

$$
| z_v - c_v | < \varepsilon, \quad \forall v \in V,
$$

*where $ c_v $ is the clustering coefficient of node $ v $.*

**Interpretation:**

Theorem 1 asserts that GraphSAGE can approximate the clustering coefficients of nodes in a graph to any desired degree of accuracy, provided certain conditions are met:

- **Distinct Node Features**: The condition $ \| \mathbf{x}_v - \mathbf{x}_{v'} \|_2 > C $ ensures that each node has a unique feature representation, preventing ambiguity during the learning process.
- **Sufficient Depth**: With at least $ K = 4 $ layers, GraphSAGE can capture the necessary neighborhood information to compute clustering coefficients.
- **Parameter Existence**: There exists a parameter setting $ \Theta^* $ (weights and biases in the aggregation functions) that enables this approximation.

### Implications of Theorem 1

- **Expressive Power**: The theorem demonstrates that GraphSAGE is not limited to propagating and transforming node features but can also infer and encode structural properties of the graph.
- **Feature Dependency**: The reliance on distinct node features highlights the importance of informative input features. In cases where node features are not inherently unique or informative, additional preprocessing or feature engineering may be necessary.
- **Model Depth**: The requirement of at least four aggregation layers suggests a trade-off between model depth and computational efficiency. Deeper models can capture more complex structures but may introduce challenges such as overfitting or increased training time.

### Discussion

The proof of Theorem 1 leverages properties of the **pooling aggregator**, which is capable of capturing complex, non-linear relationships in the aggregated neighborhood information. Specifically, the pooling operation (e.g., max or average pooling after a non-linear transformation) allows the model to distinguish between different neighborhood configurations.

This theoretical result provides insight into why GraphSAGE with pooling aggregators often outperforms models using simpler aggregators like mean or GCN-based methods. The pooling aggregator's ability to capture higher-order structural motifs contributes to a richer and more expressive node representation.

### Practical Considerations

- **Node Feature Design**: In practice, ensuring that node features are distinct and informative may not always be feasible. Augmenting features with unique identifiers or positional encodings could help satisfy the conditions of the theorem.
- **Aggregation Function Choice**: Selecting an appropriate aggregator is crucial. The pooling aggregator's success in approximating clustering coefficients suggests it may be more effective for tasks requiring sensitivity to complex neighborhood structures.
- **Model Complexity**: While deeper models can capture more intricate patterns, they also require more computational resources and may be prone to overfitting. Techniques like dropout, regularization, and careful hyperparameter tuning become important.

---

## Key Takeaways from the Theoretical Analysis

- **Graph Structural Learning**: GraphSAGE is theoretically capable of learning complex structural properties of graphs, not just propagating node features.
- **Expressive Aggregators**: The choice of aggregator function significantly impacts the model's ability to capture graph topology.
- **Feature Uniqueness**: Distinct and well-separated node features enhance the model's capacity to learn structural patterns.
- **Model Depth Matters**: Sufficient aggregation layers are necessary to capture higher-order structures, but this comes with increased computational costs.

---

By understanding the theoretical underpinnings of GraphSAGE's expressive capabilities, practitioners can make informed decisions about model architecture, aggregator selection, and feature design to maximize performance on their specific tasks.

---

## Insights and Discussion

### Importance of Node Features

GraphSAGE's reliance on node features is both a strength and a limitation. While it excels when rich feature information is available, its performance may degrade in feature-sparse scenarios. Incorporating methods to generate or infer features could mitigate this issue.

### Aggregation Depth vs. Breadth

There's a trade-off between the depth (number of layers $ K $) and the breadth (neighborhood size). Deeper models capture more global information but may suffer from over-smoothing, where node embeddings become indistinguishable. Careful tuning is required to balance local and global information.

### Computational Efficiency

By sampling neighborhoods, GraphSAGE reduces the computational burden. However, this sampling introduces stochasticity, which can affect the stability of the embeddings. Techniques like importance sampling or adaptive sampling strategies could enhance performance.

---

## Conclusion

GraphSAGE represents a significant advancement in graph representation learning by introducing an inductive framework capable of generalizing to unseen nodes in large, dynamic graphs. Its use of neighborhood aggregation functions allows it to capture both local and global structural information while effectively utilizing node features.

The framework's flexibility and scalability make it applicable to a wide range of domains, from social network analysis to biological systems. Future work could explore extensions like incorporating edge features, dynamic graphs, and unsupervised pre-training to further enhance its capabilities.

---

## References

Hamilton, W. L., Ying, R., & Leskovec, J. (2017). **Inductive Representation Learning on Large Graphs**. *Advances in Neural Information Processing Systems*, 30.

---