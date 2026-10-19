"""CarRacing-v3 preprocessing and optional historical off-track reward."""

from collections import deque

import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
import numpy as np


def off_track_from_rgb(obs: np.ndarray) -> bool:
    """Reproduce the small green-pixel probe in the November 2025 script."""
    region = obs[:84, 6:90, :]
    probe = region[75, 35:48, 1]
    return bool((probe[:2] > 200).sum() + (probe[-2:] > 200).sum() == 4)


class RepeatAndReward(gym.Wrapper):
    def __init__(self, env: gym.Env, repeat: int = 5,
                 off_track_reward: float | None = None) -> None:
        super().__init__(env)
        self.repeat = repeat
        self.off_track_reward = off_track_reward

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if self.off_track_reward is not None and off_track_from_rgb(obs):
                reward = self.off_track_reward
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def make_env(render_mode: str | None = None, repeat: int = 5,
             off_track_reward: float | None = None) -> gym.Env:
    env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=True)
    env = RepeatAndReward(env, repeat=repeat, off_track_reward=off_track_reward)
    env = GrayscaleObservation(env, keep_dim=False)
    return ResizeObservation(env, shape=(84, 84))


class FrameStackState:
    """Four grayscale 84x84 frames, normalized to [0, 1]."""

    def __init__(self, count: int = 4) -> None:
        self.frames = deque(maxlen=count)
        self.count = count

    def reset(self, frame: np.ndarray) -> np.ndarray:
        normalized = np.asarray(frame, dtype=np.float32) / 255.0
        self.frames.clear()
        for _ in range(self.count):
            self.frames.append(normalized.copy())
        return np.stack(self.frames)

    def append(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(np.asarray(frame, dtype=np.float32) / 255.0)
        return np.stack(self.frames)
