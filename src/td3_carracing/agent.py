"""Twin Delayed DDPG update with explicit configuration."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .model import Actor, Critic
from .replay import ReplayBuffer


@dataclass(frozen=True)
class TD3Config:
    actor_lr: float = 2.5e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2


class TD3Agent:
    def __init__(self, device: torch.device, config: TD3Config | None = None) -> None:
        self.device = device
        self.config = config or TD3Config()
        if self.config.policy_delay < 1:
            raise ValueError("policy_delay must be at least 1")
        self.actor = Actor().to(device)
        self.actor_target = Actor().to(device)
        self.critic1 = Critic().to(device)
        self.critic2 = Critic().to(device)
        self.critic1_target = Critic().to(device)
        self.critic2_target = Critic().to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_target.eval()
        self.critic1_target.eval()
        self.critic2_target.eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=self.config.critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.config.critic_lr)
        self.update_count = 0

    def select_action(self, state: np.ndarray, sigma: float = 0.0) -> np.ndarray:
        was_training = self.actor.training
        self.actor.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action = self.actor(tensor, sigma=sigma)[0].cpu().numpy()
        self.actor.train(was_training)
        return action

    def update(self, replay: ReplayBuffer, batch_size: int) -> dict[str, float] | None:
        if len(replay) < batch_size:
            return None
        state, action, next_state, reward, done = replay.sample(batch_size, self.device)
        cfg = self.config
        with torch.no_grad():
            noise = (torch.randn_like(action) * cfg.policy_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
            next_action = self.actor_target(next_state) + noise
            next_action[:, 0].clamp_(-1.0, 1.0)
            next_action[:, 1:].clamp_(0.0, 1.0)
            target_q = torch.minimum(
                self.critic1_target(next_state, next_action),
                self.critic2_target(next_state, next_action),
            )
            target = reward + cfg.gamma * (1.0 - done) * target_q

        loss1 = F.mse_loss(self.critic1(state, action), target)
        loss2 = F.mse_loss(self.critic2(state, action), target)
        self.critic1_optimizer.zero_grad()
        loss1.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        loss2.backward()
        self.critic2_optimizer.step()

        self.update_count += 1
        metrics = {"critic1_loss": float(loss1.item()), "critic2_loss": float(loss2.item())}
        if self.update_count % cfg.policy_delay == 0:
            actor_loss = -self.critic1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
            metrics["actor_loss"] = float(actor_loss.item())
        return metrics

    def _soft_update(self, online: torch.nn.Module, target: torch.nn.Module) -> None:
        tau = self.config.tau
        with torch.no_grad():
            for source, destination in zip(online.parameters(), target.parameters()):
                destination.mul_(1.0 - tau).add_(source, alpha=tau)
            for source, destination in zip(online.buffers(), target.buffers()):
                destination.copy_(source)

    def save_actor(self, path: str) -> None:
        torch.save(self.actor.state_dict(), path)

    def load_actor(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(state)
        self.actor_target.load_state_dict(state)
