"""CNN networks compatible with the November 2025 actor checkpoint."""

import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, state_dim: int = 84, action_dim: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
        )
        self.down1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
        )
        self.down2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Mish(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
        )
        self.down3 = nn.MaxPool2d(kernel_size=2)
        self.linear = nn.Sequential(
            nn.Linear(16 * (state_dim // 8) ** 2, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, action_dim),
            nn.LayerNorm(action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor, sigma: float = 0.0) -> torch.Tensor:
        if state.ndim == 3:
            state = state.unsqueeze(0)
        h = self.down1(self.conv1(state))
        h = self.down2(self.conv2(h))
        h = self.down3(self.conv3(h))
        h = self.linear(torch.flatten(h, start_dim=1))
        if sigma:
            h = h + torch.randn_like(h) * sigma

        steer = h[:, 0].clamp(-1.0, 1.0)
        gas = ((h[:, 1] + 1.0) * 0.5 + 0.05).clamp(0.0, 1.0)
        brake = ((h[:, 2] + 1.0) * 0.02)
        brake = (brake + (steer.abs() > 0.5).float() * steer.abs() * 0.1).clamp(0.0, 1.0)
        return torch.stack((steer, gas, brake), dim=1)


class Critic(nn.Module):
    def __init__(self, state_dim: int = 84, action_dim: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
        )
        self.down1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
        )
        self.down2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
        )
        self.down3 = nn.MaxPool2d(kernel_size=2)
        self.action_linear = nn.Sequential(
            nn.Linear(action_dim, 256), nn.LayerNorm(256), nn.Mish()
        )
        self.state_linear = nn.Sequential(
            nn.Linear(16 * (state_dim // 8) ** 2, 256), nn.LayerNorm(256), nn.Mish()
        )
        self.concat_linear = nn.Sequential(
            nn.Linear(512, 64), nn.LayerNorm(64), nn.Mish(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.Mish(), nn.Linear(64, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.ndim == 3:
            state = state.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        h = self.down1(self.conv1(state))
        h = self.down2(self.conv2(h))
        h = self.down3(self.conv3(h))
        state_h = self.state_linear(torch.flatten(h, start_dim=1))
        action_h = self.action_linear(action)
        return self.concat_linear(torch.cat((state_h, action_h), dim=1))
