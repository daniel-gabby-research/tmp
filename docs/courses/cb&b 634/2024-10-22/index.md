---
layout: default
parent: CB&B 634 Computational Methods in Informatics
grand_parent: Courses at Yale
title: "2024-10-22 Iterative algorithms"
nav_order: 28
discuss: true
math: katex
---

## 1. Definitions
An algorithm is said to be an **iterative algorithm** if it is an implementation of a method that:
– Starts with an **initial guess**,
– Produces a sequence of **approximate solutions** where each approximation depends on the previous ones, and
– Has a **termination condition**.

**Direct methods**, by contrast, perform a **finite** sequence of operations to generate an **exact** solution.
- The solution is only exact given **exact arithmetic**, which does not happen on a computer.

## 2. Convergence
An iterative algorithm is said to **converge** if the sequence of approximate solutions it generates converges...

That is, there is some value $S$ such that for any error tolerance $\epsilon > 0$, there is some number of iterations such that after that, the iterative algorithm’s approximate solutions are always within $\epsilon$ of $S$.

## 2.1 Example: Newton's method
Newton's method is an iterative algorithm for **finding roots** of a function $f(x)$.

- $x_{n+1} \leftarrow x_n - \frac{f(x_n)}{f'(x_n)}$

![alt text](image.png)

```python
g = lambda x: (x ** 4 / 4 - 2 * x ** 3 / 3 - x ** 2 / 2 + 2 * x + 2)
f = lambda x: g(x - 2)
h = 1e-4
f_prime = lambda x: (f(x + h) - f(x)) / h
gamma = 0.01
guess = 7   # initial guess
for _ in range(10):
    print(guess)
    guess = guess - gamma * f_prime(guess)
```

## 2.2 Example: Gradient descent

### 2.2.1 Local and global optimality

![alt text](image-1.png)

- **Local Minima**: A point $x$ is a local minimum of $f$ if $f(x) \leq f(y)$ for all $y$ in some neighborhood of $x$.
- **Global Minima**: A point $x$ is a global minimum of $f$ if $f(x) \leq f(y)$ for all $y$ in the domain of $f$.

### 2.2.2 Gradient descent
In higher dimensional space, the gradient points in the direction of the most rapid increase in the function value, so moving in the opposite direction (red line) heads toward smaller values.

$x_{n+1} \leftarrow x_n - \eta \nabla f(x_n),$
where $\eta$ is the **learning rate**, and $\nabla f(x_n)$ is the gradient of the function $f$ at $x_n$.

![alt text](image-2.png)

Under certain conditions on $f$ and $\eta$ sufficiently small, the gradient descent algorithm converges to a local minimum of $f$.

### 2.2.3 Escaping local minima
- If gradient descent approaches a **local (non-global) minimum**, it will never escape to find the **global minimum**.
- There are various strategies for avoiding local extrema.
    - Naively, one can simply start with **many guesses** and let each evolve.
    - Can allow **stochastic jumps** beyond the local area, typically with probability that decreases with time.
    - Can **stochastically accept increases** to the function values, generally with probability that decreases with time.

# 3. Stopping criteria
- A **maximum number** of iterations?
    - Allows detecting a failure to converge... may want to **raise an exception** if this happens.
- When the **guessed values** do not change very much.
- When the **function value** does not change very much.


# 4. Example: K-means clustering

- Suppose we have a dataset and we wish to group it into $k$ clusters.
    - Elements within clusters should be *similar*, and *dissimilar* to elements outside the clusters.

- k-means divides data points into $k$ clusters $S_1, S_2, ..., S_k$ to minimize the **Within Clusters Sum of Squares (WCSS)**:
    $\text{WCSS} = \sum_{i=1}^k \sum_{x \in S_i} \|x - \mu_i\|^2,$
    where $\mu_i$ is the center of cluster $S_i$.

## 4.1 Lloyd's algorithm
The Lloyd's algorithm proceeds as follows:
- Given data, pick $k$ points at random to serve as initial cluster centers.
- Assign points to cluster defined by nearest center.
- Compute actual centers and repeat until convergence.

![alt text](image-3.png)

> $k$-means will only ever create **convex** clusters, so it will not capture non-convex relationships.

## 4.2 $k$-means limitations
- Lloyd’s algorithm is not guaranteed to find the optimal solution.
    - It can get stuck in a local minimum of the WCSS.
    - Run multiple times.
- $k$-means incorporates all the data into a cluster.
    - Outliers could become clusters of size 1.
- $k$-means will only ever create convex clusters.

## 4.3 Python implementation
```python
import pandas as pd
import plotnine as p9
import random
import numpy as np

k = 3
df = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/iris.csv')

def normalize(series):
    return (series - series.mean()) / series.std()

df['petal.length.normalized'] = normalize(df['petal.length'])
df['petal.width.normalized'] = normalize(df['petal.width'])

pts = [np.array(pt) for pt in zip(df['petal.length.normalized'], df['petal.width.normalized'])]
centers = random.sample(pts, k)
old_cluster_ids, cluster_ids = None, [] # arbitrary but different
while cluster_ids != old_cluster_ids:
    old_cluster_ids = list(cluster_ids)
    cluster_ids = []
    for pt in pts:
        min_cluster = -1
        min_dist = float('inf')
        for i, center in enumerate(centers):
            dist = np.linalg.norm(pt - center)
            if dist < min_dist:
                min_cluster = i
                min_dist = dist
        cluster_ids.append(min_cluster)
    df['cluster'] = cluster_ids
    cluster_pts = [[pt for pt, cluster in zip(pts, cluster_ids) if cluster == match] for match in range(k)]
    centers = [sum(pts)/len(pts) for pts in cluster_pts]

(p9.ggplot(df, p9.aes(x="petal.length.normalized", y="petal.width.normalized", color="cluster"))
+ p9.geom_point()).draw()
```

