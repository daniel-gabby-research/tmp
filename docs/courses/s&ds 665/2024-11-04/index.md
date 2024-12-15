---
layout: default
parent: S&DS 665 Intermediate Machine Learning
grand_parent: Courses at Yale
title: "Reinforcement Learning Policy Methods"
nav_order: 8
discuss: true
math: katex
---

# Reinforcement Learning Policy Methods

## 1. Concepts of RL

*Policy*: A mapping from states to actions. An algorithm/rule to make decisions at each time step, designed to maximize the long-term
reward.

*Value function*: A mapping from states to total reward. The total reward the agent can expect to accumulate in the future, starting from that state (if they are making optimal decisions).

*Q-function*: A mapping from states and actions to total reward.

## 2. Principle: Bellman Equation

- **Value function optimality**:

$$v_*(s) = \max_a \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a]$$

- **Q-function optimality**:

$$Q_*(s,a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} Q_*(S_{t+1}, a') \mid S_t = s, A_t = a]$$

![q-learning](image-8.png)

## 3. Deep Q-Learning

- **Strategy**:

    - **Objective**:

    $$Q(s, a; \theta) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta) \mid S_t = s, A_t = a]$$

    - Let $y_t$ be one step of "play":

    $$y_t = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta_{old})$$

    - Adjust the parameters $\theta$ to make the squared error small (SGD):

    $$(y_t - Q(s, a; \theta))^2$$

    - Conduct SGD using backpropagation:

    $$\theta \leftarrow \theta + \eta (y_t - Q(s,a;\theta)) \nabla_\theta Q(s,a;\theta)$$

- **Replay buffer**:
    - Prevent “forgetting” how to play early parts of a game
    - Remove correlations between nearby state transitions
    - Prevent cycling behavior, due to target changing

> Learning takes place when expectations are violated. The receipt of the reward itself does not cause changes.

- *Automatic differentiation*: Gradient Collection

## 4. Multi-Armed Bandits

- The rewards are independent and noisy
- Arm k has expected payoff $\mu_k$ with variance $\sigma^2_k$ on each pull
- Each time step, pull an arm and observe the resulting reward
- Played often enough, can estimate mean reward of each arm

![alt text](image-7.png)

## 5. Policy Iteration

- **Policy evaluation**:

![policy_evaluation](image-1.png)

- **Policy improvement**:

![policy_improvement](image.png)

## 6. Policy Gradient Methods

- **Parameterize the policy**: $\pi_\theta(s)$ 

    Policy is probability distribution $\pi_\theta(a \vert s)$ over action $a$ given state $s$

- **Loss function**: Expected reward $\mathbb{J}(\theta) = \mathbb{E}[R]$

    Let $\tau$ be a trajectory sequence:

    $(s_0,a_0) \rightarrow (s_1,r_1,a_1) \rightarrow (s_2,r_2,a_2) \rightarrow \cdots \rightarrow (s_T,r_T,a_T) \rightarrow s_{T+1}$, where $s_{T+1}$ is a terminal state.

    The objective function $\mathcal{J}(\theta)$ is defined as:

    $$\mathcal{J}(\theta) = \mathbb{E}_\theta[R(\tau)] = \mathbb{E}_\theta\left[\sum_{t=1}^T r_t\right]$$


- **Calculating the gradient**:

    Using Markov property, calculate $E_\theta(R(\tau))$ as

    $$E_\theta(R(\tau)) = \int p(\tau \mid \theta) R(\tau) d\tau$$

    $$p(\tau \mid \theta) = \prod_{t=0}^T \pi_\theta(a_t \mid s_t) p(s_{t+1}, r_{t+1} \mid s_t, a_t)$$

    It follows that

    $$\nabla_\theta \log p(\tau \mid \theta) = \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t) = \sum_{t=0}^T \dfrac{\nabla_\theta \pi_\theta(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)}$$

    Now we use

    $$\nabla_\theta J(\theta) = \nabla_\theta E_\theta R(\tau) = \nabla_\theta \int R(\tau) p(\tau \mid \theta) d\tau = \int R(\tau) \nabla_\theta p(\tau \mid \theta) d\tau$$

    $$= \int R(\tau) \dfrac{\nabla_\theta p(\tau \mid \theta)}{p(\tau \mid \theta)} p(\tau \mid \theta) d\tau = E_\theta \left[ R(\tau) \nabla_\theta \log p(\tau \mid \theta) \right]$$

- **Approximating the gradient**:

    Since it’s an expectation, can approximate by sampling:

    $$
    \begin{aligned}
    \nabla_\theta J(\theta) &\approx \frac{1}{N} \sum_{i=1}^N R(\tau^{(i)}) \nabla_\theta \log p(\tau^{(i)} \mid \theta) \\
    &= \frac{1}{N} \sum_{i=1}^N R(\tau^{(i)}) \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a^{(i)}_t \mid s^{(i)}_t) \\
    &\equiv \widehat{\nabla_\theta J(\theta)}
    \end{aligned}
    $$

    The policy gradient algorithm is then

    $$\theta \leftarrow \theta + \eta \widehat{\nabla_\theta J(\theta)}$$