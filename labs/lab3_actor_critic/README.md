# Lab 3: Actor-Critic Methods

## TDDE78 — Deep Reinforcement Learning, Linköping University

---

## Overview

In this lab you will implement two fundamental **actor-critic** algorithms:
- **A2C (Advantage Actor-Critic with GAE)** — an on-policy method that combines value-based and policy gradient ideas using synchronous parallel rollouts and Generalized Advantage Estimation
- **SAC (Soft Actor-Critic)** — a state-of-the-art off-policy algorithm for continuous control with entropy regularization and twin Q-networks

Actor-critic methods sit at the intersection of value-based and policy gradient approaches: a **critic** learns to estimate the value function (reducing variance), while an **actor** uses those estimates to update the policy directly (avoiding value-function policy extraction).

### Why Actor-Critic Matters

1. **Lower variance than REINFORCE** — the critic's value estimates replace noisy Monte Carlo returns
2. **Lower bias than pure value methods** — the actor directly optimizes the policy without a greedy max operator
3. **SAC achieves state-of-the-art** — SAC (Haarnoja et al., 2018) is one of the best off-policy algorithms for continuous control, combining sample efficiency with stability

---

## Learning Objectives

After completing this lab, you will be able to:

- [ ] Explain the actor-critic framework: how the critic reduces variance and how the actor uses advantage estimates
- [ ] Implement **A2C** in PyTorch with Generalized Advantage Estimation (GAE) for continuous control
- [ ] Implement **SAC** with twin Q-networks, entropy regularization, and automatic temperature tuning
- [ ] Train agents on **LunarLander-v3** (discrete/continuous) and **BipedalWalker-v3** (continuous)
- [ ] Compare A2C vs SAC empirically: sample efficiency, stability, and final performance
- [ ] Analyze the effect of entropy temperature and GAE lambda on performance

---

## Background

### The Actor-Critic Framework

Actor-critic methods maintain two function approximators:

1. **Actor** \( \pi_\theta(a|s) \) — the policy network, updated via policy gradient
2. **Critic** \( V_\phi(s) \) or \( Q_\phi(s,a) \) — the value network, trained with TD learning

The key insight is to replace the high-variance Monte Carlo return \( G_t \) in the policy gradient with a low-variance **advantage estimate** \( A_t = Q(s_t, a_t) - V(s_t) \):

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t \right]
$$

---

### Advantage Actor-Critic (A2C)

A2C (Mnih et al., 2016) is a synchronous version of the asynchronous A3C algorithm. It collects rollouts using the current policy, computes GAE advantages, and simultaneously updates both actor and critic.

#### Generalized Advantage Estimation (GAE)

A2C uses **GAE** (Schulman et al., 2016) for advantage estimation — the same technique as PPO:

$$
\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)
$$

$$
A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

Computed backwards:

$$
A_t = \delta_t + \gamma \lambda (1 - d_t) \cdot A_{t+1}
$$

#### A2C Network Architecture

For **continuous control** (e.g. LunarLander-v3 continuous, BipedalWalker-v3):

```
state_dim → 256 → Tanh → 256 → Tanh
                                    ├── → action_dim   [actor: mean μ(s)]
                                    ├── → action_dim   [log std: log σ (learned parameter)]
                                    └── → 1            [critic: V(s)]
```

The policy outputs a **Gaussian distribution**: \( \pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta^2) \)

Actions are sampled as: \( a = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) \)

#### A2C Loss

$$
\mathcal{L}(\theta) = \underbrace{-\mathbb{E}\left[\log \pi_\theta(a_t|s_t) \cdot A_t\right]}_{\text{policy loss}} + \underbrace{c_V \cdot \mathbb{E}\left[(V_\phi(s_t) - R_t)^2\right]}_{\text{value loss}} - \underbrace{c_H \cdot \mathcal{H}[\pi_\theta(\cdot|s_t)]}_{\text{entropy bonus}}
$$

where \( R_t = A_t + V(s_t) \) is the GAE return target, \( c_V \approx 0.5 \), and \( c_H \approx 0.01 \).

#### A2C Algorithm Summary

```
Initialize actor-critic network π_θ / V_φ

For each update:
    Collect n_steps transitions using π_θ
    Get bootstrap value V(s_T) for the last state
    Compute GAE advantages and returns
    Normalize advantages: Â_t = (A_t - mean) / (std + ε)

    Compute policy loss:  L_π = -mean(log π_θ(a_t|s_t) · Â_t)
    Compute value loss:   L_V = mean((V_φ(s_t) - R_t)²)
    Compute entropy loss: L_H = -mean(H[π_θ(·|s_t)])
    Total loss: L = L_π + c_V · L_V + c_H · L_H

    Update θ by gradient descent on L
```

---

### Soft Actor-Critic (SAC)

SAC (Haarnoja et al., 2018) is an off-policy actor-critic algorithm for **continuous action spaces** that maximizes a **maximum entropy objective**:

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_t \left( r_t + \alpha \mathcal{H}[\pi(\cdot|s_t)] \right) \right]
$$

The entropy term \( \alpha \mathcal{H}[\pi] \) encourages the policy to be as stochastic as possible while still collecting high rewards. This leads to **better exploration** and **robustness** compared to standard actor-critic.

#### Twin Q-Networks (Reducing Overestimation)

SAC uses **two separate Q-networks** \( Q_{\phi_1}, Q_{\phi_2} \) and takes the minimum when computing targets — this is the "clipped double-Q" trick from TD3 (Fujimoto et al., 2018):

$$
y = r + \gamma \cdot \min_{i=1,2} Q_{\phi_i^-}(s', \tilde{a}') - \alpha \log \pi_\theta(\tilde{a}'|s')
$$

where \( \tilde{a}' \sim \pi_\theta(\cdot|s') \) is a freshly sampled action (with its log probability subtracted as the entropy term).

#### Reparameterization Trick

SAC uses the **reparameterization trick** to make the policy gradient through stochastic actions differentiable:

$$
\tilde{a} = \tanh(\mu_\theta(s) + \sigma_\theta(s) \cdot \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)
$$

The \( \tanh \) squashes actions to \([-1, 1]\). The log probability must account for this change of variables:

$$
\log \pi_\theta(a|s) = \log \mathcal{N}(\text{arctanh}(a); \mu, \sigma) - \sum_i \log(1 - a_i^2 + \varepsilon)
$$

#### SAC Losses

**Critic update (for each Q-network):**

$$
\mathcal{L}_{Q_i}(\phi_i) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ (Q_{\phi_i}(s, a) - y)^2 \right]
$$

**Actor update:**

$$
\mathcal{L}_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D}, \epsilon \sim \mathcal{N}} \left[ \alpha \log \pi_\theta(\tilde{a}|s) - \min_{i} Q_{\phi_i}(s, \tilde{a}) \right]
$$

**Temperature update (automatic tuning):**

$$
\mathcal{L}_\alpha = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\theta} \left[ -\alpha \left( \log \pi_\theta(a|s) + \bar{\mathcal{H}} \right) \right]
$$

where \( \bar{\mathcal{H}} = -|\mathcal{A}| \) is the target entropy (negative action dimension).

#### SAC Algorithm Summary

```
Initialize actor π_θ, twin critics Q_{φ1}, Q_{φ2}, target critics Q_{φ1^-}, Q_{φ2^-}
Initialize replay buffer D
Initialize log_alpha (temperature parameter)

For each environment step:
    Select action ã ~ π_θ(·|s) (reparameterization)
    Execute ã, observe r, s', done
    Store (s, ã, r, s', done) in D
    s ← s'

    For each gradient step:
        Sample mini-batch (s, a, r, s', d) from D

        # Critic update (both networks)
        ã' ~ π_θ(·|s') (reparameterization)
        y = r + γ · (min_i Q_{φi^-}(s', ã') - α · log π_θ(ã'|s')) · (1-d)
        L_Q = mean((Q_{φ1}(s,a) - y)² + (Q_{φ2}(s,a) - y)²)
        Update φ1, φ2 by gradient descent on L_Q

        # Actor update
        ã ~ π_θ(·|s) (reparameterization)
        L_π = mean(α · log π_θ(ã|s) - min_i Q_{φi}(s, ã))
        Update θ by gradient descent on L_π

        # Temperature update (auto-tuning)
        L_α = mean(-exp(log_α) · (log π_θ(ã|s) + H̄))
        Update log_α by gradient descent on L_α

        # Soft target update
        φi^- ← τ · φi + (1-τ) · φi^- for i=1,2
```

---

## Lab Structure

### Part A — Implementation (~60% of effort)

| Component | Description |
|-----------|-------------|
| **ContinuousActorCritic** | Shared MLP with Gaussian policy head and value head — for A2C |
| **SACActorNetwork** | Gaussian policy with reparameterization trick and tanh squashing |
| **SACCriticNetwork** | Twin Q-networks Q(s,a) for SAC |
| **A2C Agent** | Rollout collection, GAE computation, actor-critic update |
| **SAC Agent** | Replay buffer, off-policy updates for Q-networks, actor, and temperature |
| **Training Loops** | `train_a2c` and `train_sac` functions |
| **Evaluation** | Greedy evaluation, learning curves, return statistics |

### Part B — Experiments (~40% of effort)

| Experiment | What to Investigate |
|------------|-------------------|
| **1. A2C on LunarLander-v3** | Verify A2C implementation. Does it solve the environment? |
| **2. SAC on LunarLander-v3** | Compare sample efficiency and final performance vs A2C |
| **3. SAC on BipedalWalker-v3** | Scale to a harder continuous control task |
| **4. Ablation: Entropy Temperature** | Fixed α vs automatic tuning — effect on exploration and performance |
| **5. Ablation: GAE Lambda (A2C)** | Try λ ∈ {0, 0.95, 1.0} — how does bias-variance trade-off affect A2C? |

**For all experiments:** Run at least 3 seeds and plot mean ± std deviation.

---

## Environments

| Environment | Observation | Actions | Goal |
|-------------|------------|---------|------|
| `LunarLander-v3` (continuous) | 8-dim vector | 2 continuous (main, side engines) | Average reward ≥ 200 |
| `BipedalWalker-v3` | 24-dim vector | 4 continuous (joint torques) | Average reward ≥ 300 |

---

## Getting Started

```bash
# Make sure you're in the project root with the venv activated
cd /path/to/tdde78labsolution
source .venv/bin/activate

# Navigate to Lab 3
cd labs/lab3_actor_critic

# Launch the starter notebook
jupyter lab starter_code/lab3_ac.ipynb
```

---

## Files

```
lab3_actor_critic/
├── README.md                          # This file
├── starter_code/
│   ├── lab3_ac.ipynb                  # Starter notebook — fill in the TODOs
│   ├── networks.py                    # Actor-Critic network architectures
│   └── utils.py                       # GAE, RolloutBuffer, SAC ReplayBuffer
└── experiments/
    └── (your experiment results go here)
```

---

## Deliverables

Submit the following:

1. **Completed notebook** (`lab3_ac.ipynb`) with all TODO sections filled in and cells executed
2. **Experiment report** — either inline in the notebook or as a separate document, containing:
   - Learning curves for all experiments (mean ± std over 3+ seeds)
   - Comparison of A2C vs SAC (convergence speed, stability, final performance)
   - Entropy temperature ablation with analysis
   - Conclusions about when each algorithm is preferable

---

## Grading Criteria

| Grade | Requirements |
|-------|-------------|
| **3** | A2C implemented correctly, solves LunarLander. Basic A2C vs SAC comparison. |
| **4** | SAC implemented correctly with twin Q-networks and entropy regularization. BipedalWalker attempted. |
| **5** | All components correct including automatic temperature tuning. Thorough experimental analysis with statistical rigor. Clear discussion of entropy regularization and actor-critic trade-offs. |

---

## References

1. Mnih, V., et al. (2016). *Asynchronous Methods for Deep Reinforcement Learning*. ICML.
2. Schulman, J., et al. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. ICLR.
3. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. ICML.
4. Haarnoja, T., et al. (2018). *Soft Actor-Critic Algorithms and Applications*. arXiv:1812.05905.
5. Fujimoto, S., van Hoof, H., & Meger, D. (2018). *Addressing Function Approximation Error in Actor-Critic Methods*. ICML.
6. Plaat, A. (2022). *Deep Reinforcement Learning*. Springer. Chapters 8-9.

---

**Lab designed by Amath Sow:** [amath.sow@liu.se](mailto:amath.sow@liu.se)
