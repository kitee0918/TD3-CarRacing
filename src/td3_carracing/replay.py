"""Compact, sampled experience replay."""

from collections import deque
import random

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray,
            reward: float, done: bool) -> None:
        self.buffer.append((
            np.clip(state * 255.0, 0, 255).astype(np.uint8),
            np.asarray(action, dtype=np.float32),
            np.clip(next_state * 255.0, 0, 255).astype(np.uint8),
            float(reward),
            bool(done),
        ))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward, done = zip(*batch)
        return (
            torch.as_tensor(np.stack(state), device=device, dtype=torch.float32) / 255.0,
            torch.as_tensor(np.stack(action), device=device, dtype=torch.float32),
            torch.as_tensor(np.stack(next_state), device=device, dtype=torch.float32) / 255.0,
            torch.as_tensor(reward, device=device, dtype=torch.float32).unsqueeze(1),
            torch.as_tensor(done, device=device, dtype=torch.float32).unsqueeze(1),
        )
