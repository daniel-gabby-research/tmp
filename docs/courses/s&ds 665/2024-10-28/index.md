---
layout: default
parent: S&DS 665 Intermediate Machine Learning
grand_parent: Courses at Yale
title: "Reinforcement Learning"
nav_order: 7
discuss: true
math: katex
---

# Reinforcement Learning

## 1. Introduction

![alt text](image.png)

- The environment is in state $s$ at a given time
- The agent takes action $a$
- The environment transitions to state $s' = \text{next}(s, a)$
- The agent receives reward $r = \text{reward}(s, a)$

This is said to be a *Markov decision process*. It’s “Markov” because the next state only depends on the current state and the action selected. It’s a “decision process” because the agent is making choices of actions in a sequential manner.

**Policy**: A mapping from states to actions. An algorithm/rule to make decisions at each time step, designed to maximize the long-term reward

**Value function**: A mapping from states to total reward. The total reward the agent can expect to accumulate in the future, starting from that state.

Rewards are short term. Values are expectations of future rewards.

**Model**: Used for planning to mimic the behavior of the environment to predict rewards and next states.

**Model-free approach**: Directly estimates a value function, without modeling the environment.
Analogous to distinction between generative and discriminative
classification models


## 2. Q-learning

- **Update**:

$$Q(s,a) \leftarrow Q(s,a) + \alpha(\text{reward}(s,a) + \gamma \max_{a'} Q(\text{next}(s,a),a') - Q(s,a))$$

![q-learning](image-2.png)

## 3. Bellman equation

- **Value function optimality**:

    The optimality condition for the value function $v_*$ is:

    $$v_*(s) = \max_a \{\text{reward}(s,a) + \gamma v_*(\text{next}(s,a))\}$$

    The optimality condition for the Q-function is:

    $$Q_*(s,a) = \text{reward}(s,a) + \gamma \max_{a'} Q_*(\text{next}(s,a), a')$$

    Then, the value function $v_*(s) = \max_a Q(s,a)$


- **Deterministic Case**:
    If we know $Q_*$, we know $v_*$:

    $$
    \begin{aligned}
    v_*(s) &= \max_a Q_*(s,a) \\
    &= \max_a \lbrace \text{reward}(s,a) + \gamma \max_{a'} Q_*(next(s,a),a') \rbrace \\
    &= \max_a \lbrace \text{reward}(s,a) + \gamma v_*(next(s,a)) \rbrace
    \end{aligned}
    $$

    which is the Bellman equation for the value function.

- **Random Environment**:
> ![value-function-optimality](image-3.png)
> ![q-function-optimality](image-4.png)

- Q-learning is an example of temporal difference (TD) learning
- It is an “off-policy” approach that is practical if the space of actions is small


![ontology](image-5.png)

# Supplementary Materials

## CS 224 Stanford
![alt text](image-1.png)

### Markov Reward Process(MRP)
A Markov Reward Process (MRP) is a tuple $(S,P,R,\gamma)$ where:
- $S$ is a finite set of states
- $P$ is the state transition probability matrix, $P(s'\mid s)$
- $R$ is the reward function, $R(s)$
- $\gamma$ is the discount factor, $\gamma \in [0,1]$

The value function $V(s)$ of an MRP is the expected return starting from state $s$:

$$V(s) = R(s) + \gamma \sum_{s' \in S} P(s'\mid s)V(s')$$

This recursive equation is called the Bellman equation for MRP.


**Solution**
1. The Bellman equation can be written in matrix form as:

$$
\begin{pmatrix} 
V(s_1) \\ \vdots \\ V(s_N) 
\end{pmatrix} = \begin{pmatrix} 
R(s_1) \\ \vdots \\ R(s_N) 
\end{pmatrix} + \gamma \begin{pmatrix} 
P(s_1 \mid s_1) & \cdots & P(s_N \mid s_1) \\ 
P(s_1 \mid s_2) & \cdots & P(s_N \mid s_2) \\ 
\vdots & \ddots & \vdots \\ 
P(s_1 \mid s_N) & \cdots & P(s_N \mid s_N) 
\end{pmatrix} \begin{pmatrix} 
V(s_1) \\ \vdots \\ V(s_N) 
\end{pmatrix}
$$

Which can be written more compactly as:

$$V = R + \gamma PV$$

Rearranging:

$$V = (I - \gamma P)^{-1}R$$

This gives us a direct solution for computing the value function, *assuming the inverse exists*.


2. The value function can be computed iteratively using the Bellman equation:

$$V_k(s) = R(s) + \gamma \sum_{s' \in S} P(s'\mid s)V_{k-1}(s')$$

where:
- $V_k(s)$ is the value function at iteration $k$
- $V_{k-1}(s')$ is the value function from the previous iteration
- The iteration continues until convergence, i.e., $\lvert V_k(s) - V_{k-1}(s) \rvert < \epsilon$ for some small $\epsilon$



### Markov Decision Process(MDP) + Policy $\pi(a\mid s)$ = MRP
Given a MDP $(S,A,P,R,\gamma)$ and a policy $\pi$, the induced MRP is $(S,R^\pi,P^\pi,\gamma)$, where:

$$R^\pi(s) = \sum_{a\in A} \pi(a\mid s)R(s,a)$$

$$P^\pi(s'\mid s) = \sum_{a\in A} \pi(a\mid s)P(s'\mid s,a)$$

The value function for a policy $\pi$ in an MDP can be written as:

$$V^\pi(s) = \sum_{a\in A} \pi(a\mid s)\left[R(s,a) + \gamma \sum_{s'\in S} P(s'\mid s,a)V^\pi(s')\right]$$

This is the Bellman equation for the MDP policy evaluation problem. 

**Solution**

Similar to the MRP case, it can be solved iteratively:

$$V^\pi_k(s) = \sum_{a\in A} \pi(a\mid s)\left[R(s,a) + \gamma \sum_{s'\in S} P(s'\mid s,a)V^\pi_{k-1}(s')\right]$$

where $V^\pi_k(s)$ is the value function estimate at iteration $k$.

**Optimal Policy**

The optimal policy $\pi^*$ is the policy that maximizes the value function for all states:

$$\pi^*(s) = \arg\max_\pi V^\pi(s)$$

The optimal value function $V^*(s)$ satisfies the Bellman optimality equation:

$$V^*(s) = \max_{a\in A}\left[R(s,a) + \gamma \sum_{s'\in S} P(s'\mid s,a)V^*(s')\right]$$

This equation can be solved iteratively using value iteration:

$$V_{k+1}(s) = \max_{a\in A}\left[R(s,a) + \gamma \sum_{s'\in S} P(s'\mid s,a)V_k(s')\right]$$

Once we have the optimal value function, we can extract the optimal policy:

$$\pi^*(s) = \arg\max_{a\in A}\left[R(s,a) + \gamma \sum_{s'\in S} P(s'\mid s,a)V^*(s')\right]$$

### State-Action Value Function (Q-function)

The state-action value function $Q^\pi(s,a)$ represents the expected return starting from state $s$, taking action $a$, and thereafter following policy $\pi$:

$$Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'\in S} P(s'\mid s,a)V^\pi(s')$$

The relationship between the Q-function and value function is:

$$V^\pi(s) = \sum_{a\in A} \pi(a\mid s)Q^\pi(s,a)$$

For the optimal policy, we have:

$$Q^*(s,a) = R(s,a) + \gamma \sum_{s'\in S} P(s'\mid s,a)V^*(s')$$

And:

$$V^*(s) = \max_{a\in A} Q^*(s,a)$$

$$\pi^*(s) = \arg\max_{a\in A} Q^*(s,a)$$

