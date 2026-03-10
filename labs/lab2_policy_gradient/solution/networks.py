"""
Neural Network Architectures for Policy Gradient methods — SOLUTION.

TDDE78 — Lab 2: Policy Gradient
Linköping University, Spring 2026
"""

import numpy as np
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal weight initialization (standard in PPO / CleanRL)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class DiscreteActorCritic(nn.Module):
    """
    Actor-Critic network for discrete action spaces.

    Shared backbone with separate policy (actor) and value (critic) heads.
    Used for environments like CartPole-v1.

    Architecture:
        state_dim → 64 → 64  [shared, Tanh activations]
                            ├── → action_dim  [policy head: logits]
                            └── → 1           [value head: V(s)]

    Args:
        state_dim  (int): Dimension of the state/observation space.
        action_dim (int): Number of possible discrete actions.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super(DiscreteActorCritic, self).__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )

        # Policy head: outputs logits for a Categorical distribution
        self.policy_head = layer_init(nn.Linear(64, action_dim), std=0.01)

        # Value head: outputs scalar state value V(s)
        self.value_head = layer_init(nn.Linear(64, 1), std=1.0)

    def forward(self, state):
        """
        Forward pass through shared layers and both heads.

        Args:
            state: Tensor of shape (batch_size, state_dim)

        Returns:
            logits: Tensor of shape (batch_size, action_dim)
            value:  Tensor of shape (batch_size, 1)
        """
        features = self.shared(state)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

    def get_action(self, state, action=None):
        """
        Sample an action from the Categorical policy.

        Args:
            state:  Tensor of shape (batch_size, state_dim)
            action: Optional LongTensor of shape (batch_size,).
                    If provided, computes log_prob for these actions
                    (used during PPO update epochs).

        Returns:
            action:   LongTensor of shape (batch_size,)
            log_prob: Tensor of shape (batch_size,)
            entropy:  Tensor of shape (batch_size,) — policy entropy
            value:    Tensor of shape (batch_size, 1)
        """
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


