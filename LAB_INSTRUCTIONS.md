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

Both components are graded on a scale of **U / 3 / 4 / 5**. The final course grade is computed in two steps:

1. **LAB grade** = mean of the 5 individual lab grades (rounded to nearest integer)
2. **Final grade** = mean of LAB grade and UPG report grade (rounded to nearest integer)

> **Important:** A grade of **U** in any single lab or the report results in a **U for the entire course**. All components must receive at least a 3 to pass.

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
  - `networks.py` — neural network architectures with TODO placeholders to implement
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

> **Important:** Before submitting, make sure the **Summary section** at the end of the notebook is fully filled in. All experiment questions from Part B must be answered there.

Submit via email to [amath.sow@liu.se](mailto:amath.sow@liu.se).

### Deadlines

All labs are available from the start of the course (Wed 1 Apr).

| Lab | Submission Deadline |
|-----|---------------------|
| Lab 1 | Fri 17 Apr, 23:59 |
| Lab 2 | Fri 1 May, 23:59  |
| Lab 3 | Fri 8 May, 23:59  |
| Lab 4 | Fri 15 May, 23:59 |
| Lab 5 | Fri 29 May, 23:59 |

> Late submissions will **not** be accepted without prior approval from the course responsible.

### Grading

Each lab is graded individually on a scale of **3 / 4 / 5**. The final LAB grade is the **mean** of all 5 lab grades.

| Grade | Criteria |
|-------|----------|
| 3 | Part A correctly implemented, Part B experiments complete and interpreted, reward threshold reached |
| 4 | Grade 3 criteria met + good experimental analysis and clear discussion of results |
| 5 | Grade 4 criteria met + thorough ablations, insightful analysis, and well-answered summary section |

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

The report is graded on a scale of **3 / 4 / 5** and counts as one component in the final mean.

| Grade | Criteria |
|-------|----------|
| 3 | Report submitted, options followed, findings clearly described |
| 4 | Grade 3 criteria met + good depth of analysis and well-structured writing |
| 5 | Grade 4 criteria met + critical insight, strong argumentation, and excellent writing |

## Contact

For any questions, send an email to [amath.sow@liu.se](mailto:amath.sow@liu.se).

---

*Department of Computer Science (IDA), Linköping University — Spring 2026*
