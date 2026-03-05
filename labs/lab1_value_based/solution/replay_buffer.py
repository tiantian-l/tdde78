"""
Experience Replay Buffer for DQN — SOLUTION.

TDDE78 — Lab 1: Value-Based Deep RL
Linköping University, Spring 2026
"""

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Fixed-size replay buffer to store experience tuples.

    Each transition is stored as (state, action, reward, next_state, done).
    During training, random mini-batches are sampled to break temporal
    correlations between consecutive experiences.

    Args:
        capacity (int): Maximum number of transitions to store.
        seed (int): Random seed for reproducibility.
    """

    def __init__(self, capacity: int, seed: int = 42):
        self.buffer = deque(maxlen=capacity)
        self.seed = random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        """Add a new transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Sample a random mini-batch of transitions.

        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        """Return the current number of transitions stored."""
        return len(self.buffer)
