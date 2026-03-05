"""
Neural Network Architectures for DQN variants — SOLUTION.

TDDE78 — Lab 1: Value-Based Deep RL
Linköping University, Spring 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """
    Standard Deep Q-Network (CleanRL architecture).

    Architecture: state_dim -> 120 -> 84 -> action_dim
    Matches CleanRL's DQN implementation.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN Network (Wang et al., 2016).

    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    """

    def __init__(self, state_dim: int, action_dim: int):
        super(DuelingDQNNetwork, self).__init__()
        # Shared feature layer
        self.feature = nn.Linear(state_dim, 128)

        # Value stream
        self.value_fc = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)

        # Advantage stream
        self.advantage_fc = nn.Linear(128, 128)
        self.advantage = nn.Linear(128, action_dim)

    def forward(self, state):
        features = F.relu(self.feature(state))

        # Value stream
        v = F.relu(self.value_fc(features))
        v = self.value(v)

        # Advantage stream
        a = F.relu(self.advantage_fc(features))
        a = self.advantage(a)

        # Combine: Q = V + A - mean(A)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q
