# TDDE78 — Deep Reinforcement Learning
## Examination Instructions

**Linköping University — Department of Computer Science (IDA) — Spring 2026**

---

## Examination Structure

The examination consists of two components:

| Module | ECTS | Description |
|--------|------|-------------|
| **LAB** | 4 hp | 5 lab assignments — group code submission |
| **UPG** | 2 hp | Individual written report |

Both components must be completed to pass the course.

---

## General Rules

- Labs may be completed **individually or in groups of 2–3**.
- All group members must be able to explain and defend all submitted code and results.
- Each lab consists of two parts:
  - **Part A — Implementation:** Build the algorithm(s) from scratch in PyTorch following the TODO markers in the starter notebook.
  - **Part B — Experiments:** Run systematic ablation studies and comparisons, produce plots, and write a short analysis.
- Clone the course repository and follow the `README.md` for setup instructions:
  **[https://gitlab.liu.se/amaso88/tdde78lab](https://gitlab.liu.se/amaso88/tdde78lab)**
- Each lab's `starter_code/` directory contains:
  - A Jupyter notebook with TODO placeholders for the algorithm(s)
  - `networks.py` — fully implemented neural network architectures
  - `utils.py` — fully implemented helpers, replay buffers, and plotting functions

---

## Lab Overview

| Lab | Topic | Core Methods | Environment(s) |
|-----|-------|-------------|----------------|
| **Lab 1** | Value-Based Deep RL | DQN, Double DQN, Dueling DQN | CartPole-v1, LunarLander-v3 |
| **Lab 2** | Policy Gradient | REINFORCE (with/without baseline), PPO | CartPole-v1 |
| **Lab 3** | Actor-Critic | A2C (with GAE), SAC | LunarLanderContinuous-v3 |
| **Lab 4** | Model-Based Deep RL | Dyna-Q (neural WorldModel), MCTS | CliffWalking-v1 |
| **Lab 5** | Multi-Agent Deep RL | MAPPO, MADDPG | simple_spread_v3 (PettingZoo) |

---

## Code Submission

The code submission is **per group** — submit one **`.zip` archive** named `labX_[liu-id1]_[liu-id2]_[liu-id3].zip` containing:

```
labX_[liu-ids]/
├── labX_[name].ipynb      # Completed notebook, all cells executed
└── networks.py            # Your implementation
```

Submit via email to [amath.sow@liu.se](mailto:amath.sow@liu.se).

### Deadlines

All labs are available from the start of the course (Wed 1 Apr).

| Lab | Submission Deadline | Late Deadline (−1 grade) |
|-----|--------------------|-----------------------------|
| Lab 1 | Fri 17 Apr, 23:59 | Fri 24 Apr, 23:59 |
| Lab 2 | Fri 1 May, 23:59  | Fri 8 May, 23:59  |
| Lab 3 | Fri 8 May, 23:59  | Fri 15 May, 23:59 |
| Lab 4 | Fri 15 May, 23:59 | Fri 22 May, 23:59 |
| Lab 5 | Fri 29 May, 23:59 | Fri 5 Jun, 23:59  |

> Submissions more than one week late will **not** be accepted without prior approval from the course responsible.

### Grading

Each lab is graded **Pass / Fail**. You must pass all 5 labs.

A lab is **passed** when:
- Part A is correctly implemented (verified by running the provided test cells)
- Part B experiments are complete, reproducible, and correctly interpreted
- The passing reward threshold (stated in each lab's notebook) is reached by at least one seed

---

## UPG — Individual Written Report

The report is **individual** — every student submits their own report regardless of lab group.

Each student chooses **one** of the following two options:

### Option A — Lab Summary Report

Write a report summarising your work and findings across **all 5 labs**.

The report must be **at most 6 pages** (excluding figures and references) and cover:

1. **Overview** — briefly describe each algorithm implemented and the environment it was tested on
2. **Key results** — one or two main findings per lab (learning curves, ablation conclusions)
3. **Comparative analysis** — compare methods across labs (e.g., sample efficiency, stability, scalability)
4. **Reflection** — what worked well, what was challenging, and what you learned

### Option B — Paper Presentation: RL for LLMs

Select a published research paper on the application of **reinforcement learning to large language models** (e.g., RLHF, PPO for alignment, GRPO, reward modelling).

The report must be **at most 6 pages** (excluding figures and references) and include:

1. **Paper summary** — problem setting, proposed method, key contributions
2. **Connection to course content** — which algorithms from the labs are used or related
3. **Critical analysis** — strengths, limitations, and open questions
4. **References** — at least 3 published papers from established venues

> The paper must be approved by the course responsible **before** you start writing. Send a one-line proposal with the paper title to [amath.sow@liu.se](mailto:amath.sow@liu.se).

### Submission and Deadline

Submit your individual `report.pdf` via email to [amath.sow@liu.se](mailto:amath.sow@liu.se).

**Deadline:** Fri 5 Jun, 23:59

### Grading

The final course grade (3 / 4 / 5) is determined by the individual report:

| Grade | Requirement |
|-------|-------------|
| 3 | All labs passed + satisfactory report |
| 4 | All labs passed + good depth of analysis |
| 5 | All labs passed + thorough, insightful, and well-written report |

## Contact

For any questions, send an email to [amath.sow@liu.se](mailto:amath.sow@liu.se).

---

*Department of Computer Science (IDA), Linköping University — Spring 2026*
