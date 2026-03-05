# TDDE78: Deep Reinforcement Learning

## Linköping University — Spring 2026



## Course Overview

This course provides a foundational introduction to **deep reinforcement learning** from a computer science perspective. The focus is exclusively on **deep learning-based methods** — no tabular or dynamic programming approaches are used. Students implement state-of-the-art algorithms in PyTorch and systematically evaluate them on standard benchmark environments using Gymnasium and Atari.

### Prerequisites

- Practical programming skills in Python
- Working knowledge of deep learning, especially deep neural networks (e.g., from TDDC17 Artificial Intelligence)
- Understanding of common concepts in computer science and optimization

---

## Learning Outcomes

After completing all labs, students will be able to:

1. **Identify** basic concepts, terminology, theories, models, and methods in deep reinforcement learning
2. **Implement** core deep RL algorithms — including value-based, policy gradient, actor-critic, model-based, and multi-agent methods
3. **Design and execute** systematic experiments to evaluate and compare RL algorithms
4. **Analyze** the effect of architectural choices, hyperparameters, and algorithmic variants through ablation studies
5. **Interpret and document** experimental results with proper statistical methodology (multiple seeds, learning curves, performance metrics)
6. **Read and understand** current research literature in the field of deep reinforcement learning

---

## Lab Structure

The course consists of **5 labs**, each covering a major deep RL paradigm. Every lab is divided into two parts:

- **Part A — Implementation:** Build the algorithm(s) from scratch in PyTorch. Focus on understanding the core mechanics, loss functions, and training loops.
- **Part B — Experiments:** Run systematic experiments including ablation studies, hyperparameter sensitivity analysis, and cross-method comparisons. Produce a written report with visualizations and analysis.

### Lab Overview

| Lab | Topic | Core Methods | Environments |
|-----|-------|-------------|--------------|
| **Lab 1** | Value-Based Deep RL | DQN, Double DQN, Dueling DQN | CartPole-v1, LunarLander-v3, Atari (Pong) |
| **Lab 2** | Policy Gradient | REINFORCE (with baseline), PPO | CartPole-v1, Pendulum-v1, Atari |
| **Lab 3** | Actor-Critic | A2C (with GAE), SAC | LunarLander-v3, BipedalWalker-v3 |
| **Lab 4** | Model-Based Deep RL | Learned dynamics model, Dyna-style planning | CartPole-v1, Pendulum-v1 |
| **Lab 5** | Multi-Agent Deep RL | MADDPG / MAPPO | PettingZoo (simple_spread, simple_adversary) |

### Lab Details

#### Lab 1 — Value-Based Deep RL
> Learn to approximate Q-values with neural networks

- **Part A:** Implement vanilla DQN with experience replay and target networks. Extend to Double DQN (reducing overestimation bias) and Dueling DQN (separate value and advantage streams).
- **Part B:** Ablation studies on replay buffer size, target network update frequency, epsilon-decay schedules, and network architecture. Compare DQN variants on CartPole and LunarLander. Scale to an Atari game (e.g., Pong).

#### Lab 2 — Policy Gradient
> Learn policies directly through gradient ascent on expected return

- **Part A:** Implement REINFORCE with a learned baseline for variance reduction. Implement Proximal Policy Optimization (PPO) with clipped surrogate objective.
- **Part B:** Analyze REINFORCE vs PPO in terms of convergence speed and stability. Study the effect of the baseline, entropy bonus, GAE lambda, and clipping ratio. Test on both discrete and continuous action spaces.

#### Lab 3 — Actor-Critic
> Combine value estimation and policy optimization

- **Part A:** Implement Advantage Actor-Critic (A2C) with Generalized Advantage Estimation (GAE). Implement Soft Actor-Critic (SAC) for continuous control with entropy regularization and twin Q-networks.
- **Part B:** Compare A2C vs SAC on continuous control tasks. Study entropy temperature tuning in SAC. Analyze sample efficiency and final performance across environments.

#### Lab 4 — Model-Based Deep RL
> Learn a model of the environment and use it for planning

- **Part A:** Train a neural network to predict next-state and reward given current state and action. Integrate the learned model with a model-free method (Dyna-style: augment real experience with model-generated rollouts).
- **Part B:** Compare model-based vs model-free sample efficiency. Analyze how model prediction accuracy evolves during training. Study the effect of planning horizon (number of model rollouts). Identify when model-based approaches help vs hurt.

#### Lab 5 — Multi-Agent Deep RL
> Extend deep RL to settings with multiple interacting agents

- **Part A:** Implement Multi-Agent DDPG (MADDPG) or Multi-Agent PPO (MAPPO) with centralized training and decentralized execution (CTDE).
- **Part B:** Evaluate on cooperative (simple_spread) and competitive (simple_adversary) scenarios. Study how the number of agents affects learning. Analyze emergent communication and coordination strategies.

---

## Examination

All assessment is through a single examination component:

| Code | Component | Credits | Grading |
|------|-----------|---------|---------|
| LAB1 | Laboratory Work | 6 | U, 3, 4, 5 |

Each lab submission includes:
- Complete, runnable implementation code (Part A)
- Experiment results with plots, tables, and statistical analysis over multiple seeds (Part B)
- Written report documenting methodology, results, and discussion

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| RL Environments | Gymnasium (single-agent), Atari (via `gymnasium[atari]`) |
| Multi-Agent | PettingZoo |
| Logging | TensorBoard / Weights & Biases |
| Package Manager | uv |

---

## Setup

All dependencies are installed in an isolated virtual environment using `uv` — **no sudo required**.

```bash
# One-time setup
bash setup.sh

# Activate environment (every session)
source .venv/bin/activate

# Verify installation
python -c "import gymnasium; import torch; print('Ready!')"
```

See `setup.sh` for full details.

---

## Directory Structure

```
tdde78labsolution/
├── README.md                          # This file
├── setup.sh                           # Environment setup script
├── requirements.txt                   # Python dependencies
│
├── labs/
│   ├── lab1_value_based/              # Lab 1: DQN and variants
│   │   ├── starter_code/
│   │   ├── solution/
│   │   └── experiments/
│   │
│   ├── lab2_policy_gradient/          # Lab 2: REINFORCE, PPO
│   │   ├── starter_code/
│   │   ├── solution/
│   │   └── experiments/
│   │
│   ├── lab3_actor_critic/             # Lab 3: A2C, SAC
│   │   ├── starter_code/
│   │   ├── solution/
│   │   └── experiments/
│   │
│   ├── lab4_model_based/              # Lab 4: Learned dynamics, Dyna
│   │   ├── starter_code/
│   │   ├── solution/
│   │   └── experiments/
│   │
│   └── lab5_multi_agent/              # Lab 5: MADDPG, MAPPO
│       ├── starter_code/
│       ├── solution/
│       └── experiments/
│
├── utils/                             # Shared utilities
│   ├── replay_buffer.py
│   ├── networks.py
│   ├── plotting.py
│   └── metrics.py
│
└── results/                           # Experiment outputs
    ├── logs/
    ├── checkpoints/
    └── figures/
```

---

## Resources

- **Course Syllabus:** [studieinfo.liu.se/en/kurs/TDDE78](https://studieinfo.liu.se/en/kurs/TDDE78)
- **Gymnasium:** [gymnasium.farama.org](https://gymnasium.farama.org/)
- **PettingZoo:** [pettingzoo.farama.org](https://pettingzoo.farama.org/)
- **Textbook (open access):** [arxiv.org/abs/2201.02135](https://arxiv.org/abs/2201.02135)
- **PyTorch:** [pytorch.org](https://pytorch.org/)

---

*Department of Computer Science (IDA), Linköping University — Spring 2026*

---

**Lab designed by Amath Sow:** [amath.sow@liu.se](mailto:amath.sow@liu.se)
