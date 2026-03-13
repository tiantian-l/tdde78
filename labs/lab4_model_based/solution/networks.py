"""
Neural Network Architectures for Model-Based RL — SOLUTION.

TDDE78 — Lab 4: Model-Based Deep RL
Linköping University, Spring 2026
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    DQN Q-network: maps one-hot state → Q-values for all discrete actions.

    Reused from Lab 1 — students do NOT re-implement this.

    Input is a one-hot encoded state vector (length = n_states), not a raw
    integer.  This makes the network architecture uniform across environments.

    Architecture: state_dim → 64 → 64 → action_dim  (ReLU activations)

    Args:
        state_dim  (int): Number of discrete states (one-hot input dimension).
        action_dim (int): Number of discrete actions.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class WorldModel(nn.Module):
    """
    Neural dynamics model for discrete state spaces: (s, a) → (s′, r, done).

    *** This is the main new component students implement in Lab 4. ***

    For discrete environments the next state is a distribution over all possible
    states, not a continuous vector. The model therefore predicts:
      - next_state_logits — unnormalised log-probabilities over all states
      - reward            — scalar reward prediction
      - done_logit        — binary terminal flag (sigmoid gives P(done))

    Input:  concat(one_hot(state, n_states), one_hot(action, n_actions))
    Output: (next_state_logits, reward, done_logit)

    Loss breakdown:
      - next state : cross-entropy(logits, true_next_state_index)
      - reward     : MSE
      - done       : binary cross-entropy

    Args:
        state_dim  (int): Number of discrete states  (= n_states).
        action_dim (int): Number of discrete actions (= n_actions).
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim
        in_dim = state_dim + action_dim

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.next_state_head = nn.Linear(128, state_dim)   # logits over next states
        self.reward_head     = nn.Linear(128, 1)
        self.done_head       = nn.Linear(128, 1)           # binary logit

    def forward(self, state_onehot, action_onehot):
        """
        Args:
            state_onehot  (FloatTensor): (batch, state_dim)
            action_onehot (FloatTensor): (batch, action_dim)

        Returns:
            next_state_logits (FloatTensor): (batch, state_dim)  — cross-entropy targets
            reward            (FloatTensor): (batch,)
            done_logit        (FloatTensor): (batch,)
        """
        x                 = torch.cat([state_onehot, action_onehot], dim=-1)
        h                 = self.trunk(x)
        next_state_logits = self.next_state_head(h)
        reward            = self.reward_head(h).squeeze(-1)
        done_logit        = self.done_head(h).squeeze(-1)
        return next_state_logits, reward, done_logit
