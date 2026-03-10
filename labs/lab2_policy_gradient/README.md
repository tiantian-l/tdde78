# Lab 2: Policy Gradient Methods

## TDDE78 — Deep Reinforcement Learning, Linköping University

---

## Overview

In this lab you will implement two fundamental **policy gradient** algorithms: **REINFORCE** (with a learned baseline) and **Proximal Policy Optimization (PPO)**. Unlike value-based methods (Lab 1), these algorithms directly optimize the policy parameters using gradient ascent on expected return — no Q-value table or function approximation required.

### Why Policy Gradient Matters

Policy gradient methods are essential because they:

1. **Directly optimize what we care about** — the expected cumulative reward, rather than approximating it through value functions
2. **Handle continuous action spaces naturally** — no need to maximize over actions at every step
3. **Support stochastic policies** — useful for exploration and in partially-observable environments
4. **Scale to modern deep RL** — PPO (Schulman et al., 2017) remains one of the most widely used deep RL algorithms

---

## Learning Objectives

After completing this lab, you will be able to:

- [ ] State and apply the **Policy Gradient Theorem** to derive a gradient estimate from sampled trajectories
- [ ] Implement **REINFORCE** in PyTorch, including the full training loop for episodic tasks
- [ ] Explain why REINFORCE has high variance, and reduce it using a **learned baseline**
- [ ] Implement **PPO** with the clipped surrogate objective, **GAE** for advantage estimation, and an entropy bonus
- [ ] Train agents on **CartPole-v1** (discrete actions) with both algorithms
- [ ] Compare REINFORCE and PPO empirically and analyze the effect of the value baseline

---

## Background

### The Policy Gradient Theorem

In reinforcement learning, we parameterize a policy $\pi_\theta(a \mid s)$ directly and seek parameters $\theta$ that maximize the expected discounted return:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]
$$

The **Policy Gradient Theorem** (Sutton et al., 1999) gives the gradient of $J(\theta)$:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]
$$

where $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ is the discounted return from step $t$. This is the foundation for all policy gradient algorithms — the gradient tells us to increase the probability of actions that led to high returns, and decrease those that led to low returns.

---

### REINFORCE (Williams, 1992)

REINFORCE is the simplest policy gradient algorithm. It collects a **complete episode**, computes $G_t$ for each timestep, and updates the policy with stochastic gradient ascent:

$$
\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t
$$

**Algorithm:**

```
Initialize policy π_θ with random weights θ

For each episode:
    Collect trajectory τ = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T)
    For each step t:
        Compute G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...
    Update θ ← θ + α · Σ_t ∇_θ log π_θ(a_t | s_t) · G_t
```

**Limitations:** REINFORCE has extremely high variance — different runs of the same policy produce very different $G_t$ values. This makes learning slow and unstable.

---

### Variance Reduction: The Baseline

A key insight is that we can subtract any **baseline** $b(s_t)$ from $G_t$ without introducing bias (as long as the baseline does not depend on the action):

$$
\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (G_t - b(s_t)) \right]
$$

The best baseline is the **value function** $V^\pi(s_t)$, which represents the expected return from state $s_t$ under the current policy. The difference $A_t = G_t - V^\pi(s_t)$ is called the **advantage** — it measures how much better (or worse) action $a_t$ was compared to the average action in state $s_t$.

**Implementation:** We add a **value head** to the policy network that predicts $V(s_t)$. It is trained with MSE against the Monte Carlo return $G_t$:

$$
\mathcal{L}_V(\phi) = \mathbb{E} \left[ (G_t - V_\phi(s_t))^2 \right]
$$

The total loss for REINFORCE with baseline is:

$$
\mathcal{L}(\theta, \phi) = - \sum_t \log \pi_\theta(a_t \mid s_t) \cdot \underbrace{(G_t - V_\phi(s_t))}_{\text{advantage}} + c_V \cdot \mathcal{L}_V(\phi)
$$

---

### Proximal Policy Optimization (PPO)

REINFORCE (and vanilla policy gradient) suffers from another problem: a large gradient step can collapse the policy and take it to a region from which recovery is slow. PPO (Schulman et al., 2017) addresses this by constraining how much the policy can change per update.

#### Surrogate Objective

PPO uses **importance sampling** to reuse data collected under an old policy $\pi_{\theta_\text{old}}$ to update a new policy $\pi_\theta$. Define the probability ratio:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}
$$

The unclipped surrogate objective is:

$$
L^{\text{CPI}}(\theta) = \mathbb{E}_t \left[ r_t(\theta) \cdot A_t \right]
$$

This is equivalent to a policy gradient step but allows multiple epochs of updates per rollout.

#### Clipped Objective

To prevent large policy updates, PPO clips the ratio:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \cdot A_t, \; \text{clip}(r_t(\theta), 1 - \varepsilon, 1 + \varepsilon) \cdot A_t \right) \right]
$$

The clip parameter $\varepsilon$ (typically 0.1–0.3) prevents the ratio from deviating too far from 1. The $\min$ ensures we never benefit from going beyond the clip — it is a lower bound on the unclipped objective.

#### Value Function Loss

PPO jointly trains a value function:

$$
L^V(\theta) = \mathbb{E}_t \left[ (V_\theta(s_t) - V_t^{\text{target}})^2 \right]
$$

#### Entropy Bonus

To encourage exploration, an entropy bonus is added:

$$
L^{\text{S}}(\theta) = \mathbb{E}_t \left[ \mathcal{H}[\pi_\theta(\cdot \mid s_t)] \right]
$$

#### Full PPO Objective

$$
L(\theta) = L^{\text{CLIP}}(\theta) - c_V \cdot L^V(\theta) + c_S \cdot L^{\text{S}}(\theta)
$$

where $c_V \approx 0.5$ and $c_S \approx 0.01$ are coefficients.

---

### Generalized Advantage Estimation (GAE)

PPO uses **GAE** (Schulman et al., 2016) instead of Monte Carlo returns for advantage estimation. GAE interpolates between one-step TD ($\lambda = 0$) and Monte Carlo ($\lambda = 1$):

Define the temporal difference error:

$$
\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)
$$

The GAE advantage is:

$$
A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

Computed efficiently backwards:

$$
A_t = \delta_t + \gamma \lambda (1 - d_t) \cdot A_{t+1}
$$

- $\lambda = 0$: $A_t = \delta_t$ — low variance, high bias (one-step TD)
- $\lambda = 1$: $A_t = G_t - V(s_t)$ — high variance, zero bias (Monte Carlo)
- Typical: $\lambda = 0.95$ — a good bias-variance trade-off

#### PPO Algorithm Summary

```
Initialize network π_θ (shared actor-critic)

For each update:
    Collect n_steps transitions using π_θ (may span multiple episodes)
    Compute GAE advantages and returns
    Normalize advantages: Â_t = (A_t - mean) / (std + ε)

    For k epochs:
        For each mini-batch:
            Compute log_prob ratio r = exp(log π_θ(a|s) - log π_old(a|s))
            Compute clipped policy loss
            Compute value loss
            Compute entropy bonus
            Update θ via gradient descent
```

---

## Lab Structure

### Part A — Implementation (~60% of effort)

| Component | Description |
|-----------|-------------|
| **DiscreteActorCritic** | Shared MLP with policy (logits) and value heads — for CartPole-v1 |
| **REINFORCE Agent** | Episodic collection, compute_returns, policy gradient update with optional baseline |
| **PPO Agent** | Rollout collection, GAE computation, clipped PPO update with entropy bonus |
| **Training Loops** | train_reinforce and train_ppo functions |
| **Evaluation** | Greedy evaluation, learning curves, return statistics |

### Part B — Experiments (~40% of effort)

| Experiment | What to Investigate |
|------------|-------------------|
| **1. REINFORCE vs PPO on CartPole** | Compare convergence speed, stability, and final reward |
| **2. Baseline Ablation** | REINFORCE with vs. without baseline — how much does variance reduction help? |
| **3. Ablation: GAE Lambda** | Try λ ∈ {0, 0.95, 1.0} — TD vs default vs Monte Carlo advantage estimation |

**For all experiments:** Run at least 3 seeds and plot mean ± std deviation.

---

## Environment

| Environment | Observation | Actions | Goal |
|-------------|------------|---------|------|
| `CartPole-v1` | 4-dim vector | 2 discrete | Average reward ≥ 475 over 100 episodes |

---

## Getting Started

```bash
# Make sure you're in the project root with the venv activated
cd /path/to/tdde78labsolution
source .venv/bin/activate

# Navigate to Lab 2
cd labs/lab2_policy_gradient

# Launch the starter notebook
jupyter lab starter_code/lab2_pg.ipynb
```

---

## Files

```
lab2_policy_gradient/
├── README.md                          # This file
├── starter_code/
│   ├── lab2_pg.ipynb                  # Starter notebook — fill in the TODOs
│   ├── networks.py                    # Actor-Critic network architectures
│   └── utils.py                       # compute_returns, compute_gae, RolloutBuffer
└── experiments/
    └── (your experiment results go here)
```

---

## Deliverables

Submit the following:

1. **Completed notebook** (`lab2_pg.ipynb`) with all TODO sections filled in and cells executed
2. **Experiment report** — either inline in the notebook or as a separate document, containing:
   - Learning curves for all experiments (mean ± std over 3+ seeds)
   - Comparison of REINFORCE vs PPO (convergence speed, variance, final performance)
   - Baseline ablation analysis and conclusions

---

## Grading Criteria

| Grade | Requirements |
|-------|-------------|
| **3** | REINFORCE with baseline implemented correctly, solves CartPole. Basic REINFORCE vs PPO comparison. |
| **4** | PPO implemented correctly, solves CartPole. Clear discussion of variance reduction and baseline effect. |
| **5** | All components correct. Thorough experimental analysis with statistical rigor (3+ seeds, mean ± std). Insightful discussion of the bias-variance trade-off and the PPO trust region. |

---

## References

1. Williams, R.J. (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning*. Machine Learning, 8(3-4), 229-256.
2. Sutton, R.S., McAllester, D., Singh, S., & Mansour, Y. (1999). *Policy Gradient Methods for Reinforcement Learning with Function Approximation*. NeurIPS.
3. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. ICLR.
4. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
5. Plaat, A. (2022). *Deep Reinforcement Learning*. Springer. Chapters 6-7.

---

**Lab designed by Amath Sow:** [amath.sow@liu.se](mailto:amath.sow@liu.se)
