# Lab 1: Value-Based Deep Reinforcement Learning

## TDDE78 — Deep Reinforcement Learning, Linköping University

---

## Overview

In this lab you will implement **Deep Q-Networks (DQN)** and two of its most important variants: **Double DQN** and **Dueling DQN**. These are the foundational algorithms for value-based deep RL — they use neural networks to approximate the optimal action-value function Q*(s, a) and learn policies directly from high-dimensional observations.

### Why DQN Matters

DQN (Mnih et al., 2015) was the first deep RL algorithm to demonstrate superhuman performance on Atari games, learning directly from raw pixel inputs. It introduced two key innovations that made deep RL stable:

1. **Experience Replay** — stores transitions in a buffer and samples random mini-batches, breaking temporal correlations
2. **Target Network** — a slowly-updated copy of the Q-network used to compute stable TD targets

You will implement these from scratch, understand why they are necessary, and then explore improvements that address DQN's known weaknesses.

---

## Learning Objectives

After completing this lab, you will be able to:

- [ ] Explain why naive Q-learning with neural networks is unstable and how experience replay and target networks address this
- [ ] Implement a complete DQN agent in PyTorch, including the training loop, epsilon-greedy exploration, and loss computation
- [ ] Implement Double DQN and explain how it reduces overestimation bias
- [ ] Implement Dueling DQN and explain the advantage of separating value and advantage streams
- [ ] Design and run systematic experiments to compare algorithm variants
- [ ] Analyze learning curves, hyperparameter sensitivity, and performance across environments

---

## Background

### Q-Learning Recap

In reinforcement learning, an agent interacts with an environment by observing states \( s \in \mathcal{S} \), taking actions \( a \in \mathcal{A} \), receiving rewards \( r \), and transitioning to new states \( s' \). The goal is to find a policy \( \pi(a|s) \) that maximizes the expected cumulative discounted reward:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

The **action-value function** (Q-function) represents the expected return when taking action \( a \) in state \( s \) and following policy \( \pi \) thereafter:

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid s_t = s, a_t = a \right]
$$

The **optimal Q-function** \( Q^*(s, a) \) satisfies the Bellman optimality equation:

$$
Q^*(s, a) = \mathbb{E} \left[ r + \gamma \max_{a'} Q^*(s', a') \mid s, a \right]
$$

Classical Q-learning updates a table of Q-values using this equation. However, when the state space is large or continuous (e.g., images), a table is infeasible — we need function approximation.

---

### Deep Q-Networks (DQN)

**The Problem:** Naive Q-learning with neural networks is unstable for two reasons:
1. **Correlated samples** — consecutive transitions \( (s_t, a_t, r_t, s_{t+1}) \) are highly correlated, violating the i.i.d. assumption of stochastic gradient descent
2. **Moving target** — the target \( r + \gamma \max_{a'} Q(s', a'; \theta) \) changes with every update to \( \theta \), creating a "chasing your own tail" effect

**DQN's Solution:** Mnih et al. (2015) introduced two stabilization techniques:

#### Experience Replay

Instead of training on consecutive transitions, DQN stores all transitions \( (s, a, r, s', \text{done}) \) in a **replay buffer** \( \mathcal{D} \) of fixed capacity. During training, random mini-batches are sampled from \( \mathcal{D} \):

- **Breaks temporal correlations** — random sampling decorrelates the training data
- **Improves sample efficiency** — each transition can be reused multiple times
- **Smooths over data distribution** — averages over many past behaviors

#### Target Network

A separate **target network** \( Q(s, a; \theta^-) \) with parameters \( \theta^- \) is used to compute the TD target. The target network is a periodic copy of the online network — its parameters \( \theta^- \) are only updated every \( C \) steps by copying \( \theta^- \leftarrow \theta \):

- **Stabilizes the target** — the target doesn't change on every gradient step
- **Prevents oscillations** — reduces the feedback loop between Q-values and targets

#### DQN Loss Function

The network \( Q(s, a; \theta) \) is trained by minimizing the mean squared temporal difference (TD) error:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( y - Q(s, a; \theta) \right)^2 \right]
$$

where the target \( y \) is:

$$
y = r + \gamma \max_{a'} Q(s', a'; \theta^-)
$$

Note that \( \theta^- \) (target network) is held fixed during the gradient computation — gradients only flow through \( Q(s, a; \theta) \).

#### Epsilon-Greedy Exploration

DQN uses an \( \varepsilon \)-greedy policy for exploration:

$$
a = \begin{cases}
\text{random action} & \text{with probability } \varepsilon \\
\arg\max_a Q(s, a; \theta) & \text{with probability } 1 - \varepsilon
\end{cases}
$$

Typically, \( \varepsilon \) starts at 1.0 (fully random) and decays linearly to a small value (e.g., 0.01) over training, gradually shifting from exploration to exploitation.

#### DQN Algorithm Summary

```
Initialize replay buffer D with capacity N
Initialize Q-network with random weights θ
Initialize target network with weights θ⁻ = θ

For each episode:
    Observe initial state s
    For each step:
        Select action a using ε-greedy policy based on Q(s, ·; θ)
        Execute a, observe reward r, next state s', done
        Store (s, a, r, s', done) in D
        Sample random mini-batch of transitions from D
        Compute target: y = r + γ · max_a' Q(s', a'; θ⁻) · (1 - done)
        Update θ by gradient descent on (y - Q(s, a; θ))²
        Every C steps: θ⁻ ← θ
        s ← s'
```

---

### Double DQN (van Hasselt et al., 2016)

**The Problem:** Standard DQN uses the max operator to both **select** and **evaluate** the best action in the target:

$$
y = r + \gamma \max_{a'} Q(s', a'; \theta^-)
$$

This leads to **overestimation bias**: because \( \max \) always picks the highest Q-value (including those inflated by noise), the targets are systematically too high. Over time, this can cause poor policy performance even when Q-values appear to be improving.

**Why it happens:** Consider noisy Q-value estimates. Even if the true Q-values are similar for all actions, noise will make some estimates higher than others. The max operator always selects the noisy overestimate, leading to a positive bias:

$$
\mathbb{E}\left[\max_a Q(s, a)\right] \geq \max_a \mathbb{E}\left[Q(s, a)\right]
$$

**Double DQN's Solution:** Decouple action selection from action evaluation by using two networks:

$$
y = r + \gamma Q\left(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-\right)
$$

- The **online network** \( \theta \) selects the best action: \( a^* = \arg\max_{a'} Q(s', a'; \theta) \)
- The **target network** \( \theta^- \) evaluates that action: \( Q(s', a^*; \theta^-) \)

This is a minimal change — only 2 lines of code differ from standard DQN — but it significantly reduces overestimation and often improves final performance.

**Implementation:** In the `compute_loss()` method, the only difference is:

```python
# Standard DQN
next_q = target_network(next_states).max(dim=1)[0]

# Double DQN
best_actions = q_network(next_states).argmax(dim=1)           # select with online
next_q = target_network(next_states).gather(1, best_actions)   # evaluate with target
```

---

### Dueling DQN (Wang et al., 2016)

**The Insight:** For many states, the value of being in that state matters more than which action you take. For example, if the agent is about to crash, no action will help — the state itself is bad. Conversely, in a safe state, most actions are equally good.

**Dueling Architecture:** Instead of directly estimating \( Q(s, a) \), the network separates it into two streams:

- **Value stream** \( V(s; \theta) \) — how good is this state overall?
- **Advantage stream** \( A(s, a; \theta) \) — how much better is this action compared to the average?

These are combined as:

$$
Q(s, a; \theta) = V(s; \theta) + A(s, a; \theta) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \theta)
$$

The subtraction of the mean advantage is critical for **identifiability** — without it, \( V \) and \( A \) are not uniquely determined (you could shift value between them without changing \( Q \)).

**Network Architecture:**

```
                                    ┌─ FC(128) → ReLU → FC(1)           → V(s)
State → FC(128) → ReLU → [features]│
                                    └─ FC(128) → ReLU → FC(action_dim)  → A(s,a)

Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
```

**Benefits:**
- The value stream can learn the state value without having to learn the effect of every action
- More efficient learning when many actions have similar effects
- Particularly helpful in environments with large action spaces

---

## Lab Structure

### Part A — Implementation (~60% of effort)

You will implement the following components in the starter notebook `starter_code/lab1_dqn.ipynb`:

| Component | Description |
|-----------|-------------|
| **Replay Buffer** | Store and sample transitions `(s, a, r, s', done)` |
| **DQN Network** | Fully-connected Q-network |
| **Double DQN** | Modified target computation to reduce overestimation |
| **Dueling DQN** | Separate value and advantage streams |
| **DQN Agent** | Epsilon-greedy action selection, training loop, target network updates |
| **Training Loop** | Train the agent, log metrics, evaluate periodically |

### Part B — Experiments (~40% of effort)

After implementing the algorithms, run the following experiments:

| Experiment | What to Investigate |
|------------|-------------------|
| **1. DQN on CartPole-v1** | Verify your implementation works. Plot learning curves (reward vs episodes). |
| **2. DQN on LunarLander-v3** | Test on a harder environment. How many episodes to solve? |
| **3. DQN vs Double DQN** | Compare Q-value estimates. Does Double DQN reduce overestimation? |
| **4. DQN vs Dueling DQN** | Compare learning speed and final performance. |
| **5. Ablation: Replay Buffer Size** | Try sizes: 1000, 10000, 100000. How does it affect stability? |
| **6. Ablation: Target Network Update** | Try update frequencies: every 1, 10, 100, 1000 steps. |
| **7. Ablation: Epsilon Decay** | Compare linear vs exponential decay schedules. |
| **8. (Bonus) Atari Pong** | Scale DQN to pixel-based observations with a CNN. |

**For all experiments:** Run at least 3 seeds and plot mean ± std deviation.

---

## Environments

| Environment | Observation | Actions | Solved At |
|-------------|------------|---------|-----------|
| `CartPole-v1` | 4-dim vector (position, velocity, angle, angular velocity) | 2 (left, right) | Average reward ≥ 475 over 100 episodes |
| `LunarLander-v3` | 8-dim vector (position, velocity, angle, leg contact) | 4 (noop, left engine, main engine, right engine) | Average reward ≥ 200 over 100 episodes |
| `ALE/Pong-v5` (bonus) | 210×160×3 RGB pixels | 6 (movement + fire) | Average reward ≥ 18 |

---

## Getting Started

```bash
# Make sure you're in the project root with the venv activated
cd /path/to/tdde78labsolution
source .venv/bin/activate

# Navigate to Lab 1
cd labs/lab1_value_based

# Launch the starter notebook
jupyter lab starter_code/lab1_dqn.ipynb
```

---

## Files

```
lab1_value_based/
├── README.md                          # This file
├── starter_code/
│   ├── lab1_dqn.ipynb                 # Starter notebook — fill in the TODOs
│   ├── replay_buffer.py               # Experience replay buffer
│   └── networks.py                    # DQN network architectures
└── experiments/
    └── (your experiment results go here)
```

---

## Deliverables

Submit the following:

1. **Completed notebook** (`lab1_dqn.ipynb`) with all TODO sections filled in and cells executed
2. **Experiment report** — either inline in the notebook or as a separate document, containing:
   - Learning curves for all experiments (mean ± std over 3+ seeds)
   - Analysis of Double DQN vs DQN overestimation
   - Ablation study results with discussion
   - Conclusions about what matters most for DQN performance

---

## Grading Criteria

| Grade | Requirements |
|-------|-------------|
| **3** | DQN implemented correctly, solves CartPole. Basic experiments with learning curves. |
| **4** | Double DQN and Dueling DQN implemented. LunarLander solved. Ablation studies with analysis. |
| **5** | All variants implemented correctly. Thorough experimental analysis with statistical rigor. Clear discussion of results and insights. Bonus Atari task attempted. |

---

## References

1. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529-533.
2. van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep Reinforcement Learning with Double Q-learning*. AAAI.
3. Wang, Z., et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*. ICML.
4. Plaat, A. (2022). *Deep Reinforcement Learning*. Springer. Chapters 3-5.
5. Lin, L.-J. (1992). *Self-improving reactive agents based on reinforcement learning, planning and teaching*. Machine Learning, 8(3-4), 293-321.

---

**Lab designed by Amath Sow:** [amath.sow@liu.se](mailto:amath.sow@liu.se)
