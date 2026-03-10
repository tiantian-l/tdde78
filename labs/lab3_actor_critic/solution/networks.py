"""
Neural Network Architectures for Actor-Critic methods — SOLUTION.

TDDE78 — Lab 3: Actor-Critic
Linköping University, Spring 2026
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -5
LOG_STD_MAX = 2


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal weight initialization (standard in A2C / CleanRL)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# =============================================================================
#  A2C Network — Continuous Control
# =============================================================================

class ContinuousActorCritic(nn.Module):
    """
    Actor-Critic network for continuous action spaces (A2C).

    Shared backbone with Gaussian policy head (state-independent log_std)
    and scalar value head.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super(ContinuousActorCritic, self).__init__()

        # SOLUTION: shared backbone + two heads + learnable log_std
        self.shared = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )
        self.actor_mean = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.critic     = layer_init(nn.Linear(256, 1),          std=1.0)
        self.log_std    = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        """Return (mean, value) given a batch of states."""
        features = self.shared(state)
        mean     = self.actor_mean(features)
        value    = self.critic(features)
        return mean, value

    def get_action(self, state, action=None):
        """
        Sample (or evaluate) a Gaussian action.

        Returns: action, log_prob (summed), entropy (summed), value
        """
        mean, value = self.forward(state)
        std         = self.log_std.exp().expand_as(mean)
        dist        = Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value


# =============================================================================
#  SAC Networks — Continuous Control
# =============================================================================

class SACActor(nn.Module):
    """
    SAC Actor: Gaussian policy with reparameterization trick and tanh squashing.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super(SACActor, self).__init__()

        # SOLUTION: 2-layer MLP + separate mean and log_std heads
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean_head    = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, state):
        """Return (mean, log_std) of the pre-squashing Gaussian."""
        h       = self.net(state)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_action(self, state):
        """
        Reparameterized sample with tanh squashing and log-prob correction.

        Returns: action (tanh-squashed), log_prob (with Jacobian correction)
        """
        mean, log_std = self.forward(state)
        std           = log_std.exp()
        dist          = Normal(mean, std)

        # Reparameterized sample (differentiable through std)
        x_t    = dist.rsample()
        action = torch.tanh(x_t)

        # Log prob with tanh change-of-variables correction (numerically stable)
        log_prob  = dist.log_prob(x_t).sum(dim=-1)
        log_prob -= (2 * (np.log(2) - x_t - F.softplus(-2 * x_t))).sum(dim=-1)
        return action, log_prob


class SACCritic(nn.Module):
    """
    SAC Critic: Twin Q-networks Q(s, a).
    """

    def __init__(self, state_dim: int, action_dim: int):
        super(SACCritic, self).__init__()

        # SOLUTION: two independent Q-networks
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),                    nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),                    nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        """Return (q1, q2) for the (state, action) pair."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)
