import warnings
import cv2
import gymnasium as gym
import random
from collections import deque
import torch
from torch import nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from typing import Union

warnings.filterwarnings("ignore")


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ 使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("❌ 使用 CPU (GPU 不可用)")
    return device


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
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
            nn.Mish()
        )
        self.down2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Mish(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish()
        )
        self.down3 = nn.MaxPool2d(kernel_size=2)

        self.linear = nn.Sequential(
            nn.Linear(16 * (state_dim // 8) ** 2, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, action_dim),
            nn.LayerNorm(action_dim),
            nn.Tanh()
        )

    def forward(self, state: Union[np.ndarray, torch.Tensor, list], sigma=0, brake_rate=1.0, turn_speed_reduction=0.0):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        elif isinstance(state, list):
            state = torch.FloatTensor(state)

        if len(state.shape) == 3:
            state = state.unsqueeze(0)

        h = self.conv1(state)
        h = self.down1(h)
        h = self.conv2(h)
        h = self.down2(h)
        h = self.conv3(h)
        h = self.down3(h)
        h = torch.flatten(h, start_dim=1)

        h = self.linear(h)
        h = h + torch.randn_like(h) * sigma

        steer = h[:, 0]
        gas = (h[:, 1] + 1) * 0.5 + 0.1
        brake = (h[:, 2] + 1) * 0.015

        turn_magnitude = torch.abs(steer)
        gas_reduction = turn_magnitude * turn_speed_reduction
        gas = gas * (1.0 - gas_reduction)
        sharp_turn_mask = (turn_magnitude > 0.5).float()
        brake = brake + sharp_turn_mask * turn_magnitude * 0.1

        steer = torch.clamp(steer, -1.0, 1.0)
        gas = torch.clamp(gas, 0.0, 1.0)
        brake = torch.clamp(brake, 0.0, 1.0)

        result = torch.stack([steer, gas, brake], dim=1)
        return result


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
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
            nn.Mish()
        )
        self.down2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish()
        )
        self.down3 = nn.MaxPool2d(kernel_size=2)
        self.action_linear = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.LayerNorm(256),
            nn.Mish()
        )
        self.state_linear = nn.Sequential(
            nn.Linear(16 * (state_dim // 8) ** 2, 256),
            nn.LayerNorm(256),
            nn.Mish(),
        )
        self.concat_linear = nn.Sequential(
            nn.Linear(512, 64),
            nn.LayerNorm(64),
            nn.Mish(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.Mish(),
            nn.Linear(64, 1)
        )

    def forward(self, state: Union[np.ndarray, torch.Tensor, list], action: Union[np.ndarray, torch.Tensor, list]):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        elif isinstance(state, list):
            state = torch.FloatTensor(state)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action)
        elif isinstance(action, list):
            action = torch.FloatTensor(action)

        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        state_h = self.conv1(state)
        state_h = self.down1(state_h)
        state_h = self.conv2(state_h)
        state_h = self.down2(state_h)
        state_h = self.conv3(state_h)
        state_h = self.down3(state_h)
        state_h = torch.flatten(state_h, start_dim=1)
        state_h = self.state_linear(state_h)
        action_h = self.action_linear(action)
        h = self.concat_linear(torch.cat((state_h, action_h), dim=1))
        return h


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, next_state, reward, done):
        experience = (state, action, next_state, reward, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward, done = map(np.stack, zip(*batch))
        return (torch.FloatTensor(state),
                torch.FloatTensor(action),
                torch.FloatTensor(next_state),
                torch.FloatTensor(reward),
                torch.FloatTensor(done))

    def __len__(self):
        return len(self.buffer)


class Agent:
    ### MODIFICATION START: Different learning rates for Actor and Critic ###
    def __init__(self, state_dim: int, action_dim: int, actor_lr: float, critic_lr: float, device="cpu") -> None:
        ### MODIFICATION END ###
        self.brake_rate = 0
        self.turn_speed_reduction = 0.0
        self.device = device

        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1 = Critic(state_dim, action_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2 = Critic(state_dim, action_dim).to(device)
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        ### MODIFICATION START: Use separate learning rates ###
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        ### MODIFICATION END ###

        self.critic_mse_loss = nn.MSELoss().to(device)

    def select_action(self, state, sigma=0):
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if len(state.shape) == 3:
                state = state.unsqueeze(0)
            action = self.actor(
                state.to(self.device),
                sigma=sigma,
                brake_rate=self.brake_rate,
                turn_speed_reduction=self.turn_speed_reduction
            )
            return action.cpu().numpy()[0]

    def train(self, replay_buffer: ReplayBuffer, batch_size: int, gamma: float, tau: float, update_time=0):
        sample_result = replay_buffer.sample(batch_size)
        if sample_result is None:
            return

        state, action, next_state, reward, done = sample_result
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)

        predicted_q1 = self.critic1(state, action)
        predicted_q2 = self.critic2(state, action)

        with torch.no_grad():
            target_action = self.target_actor(
                next_state,
                brake_rate=self.brake_rate,
                turn_speed_reduction=self.turn_speed_reduction
            )
            target_action = target_action + torch.clamp(0.02 * torch.randn_like(target_action), -0.05,
                                                        0.05)  # Clipped noise
            target_action = torch.clamp(target_action, -1.0, 1.0)
            target_q1 = self.target_critic1(next_state, target_action)
            target_q2 = self.target_critic2(next_state, target_action)
            target_q = torch.min(target_q1, target_q2)
            TD_target = reward.unsqueeze(1) + gamma * target_q * (1 - done.unsqueeze(1))

        critic1_loss = self.critic_mse_loss(predicted_q1, TD_target)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        self.critic1_optimizer.step()

        critic2_loss = self.critic_mse_loss(predicted_q2, TD_target)
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic2_optimizer.step()

        ### MODIFICATION START: Use actor_update_freq parameter ###
        actor_update_freq = 2  # Default TD3 delay
        if update_time % actor_update_freq == 0:
            ### MODIFICATION END ###
            predicted_action = self.actor(
                state,
                brake_rate=self.brake_rate,
                turn_speed_reduction=self.turn_speed_reduction
            )
            actor_loss = -self.critic1(state, predicted_action).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            with torch.no_grad():
                for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


### MODIFICATION START: Simplified reward function for early training ###
def car_in_road_reward_simplified(image):
    """Simplified reward: just checks if the car is on the road using a central region."""
    part_image = image[60:84, 40:60, :]  # Region right under the car
    road_color_lower = np.array([90, 90, 90], dtype=np.uint8)
    road_color_upper = np.array([120, 120, 120], dtype=np.uint8)
    grass_color_lower = np.array([90, 180, 90], dtype=np.uint8)
    grass_color_upper = np.array([120, 255, 120], dtype=np.uint8)

    road_mask = cv2.inRange(part_image, road_color_lower, road_color_upper)
    grass_mask = cv2.inRange(part_image, grass_color_lower, grass_color_upper)

    road_pixel_count = cv2.countNonZero(road_mask)
    grass_pixel_count = cv2.countNonZero(grass_mask)

    if road_pixel_count + grass_pixel_count == 0:
        return -0.1  # Small penalty if we can't see road or grass

    # Reward is proportional to road pixels, becomes a penalty if mostly grass
    reward = (road_pixel_count) / (road_pixel_count + grass_pixel_count) - 0.5
    return reward


### MODIFICATION END ###

def detect_curve_direction(image):
    height, width = image.shape[:2]
    roi = image[height // 2:, :, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    left_half = edges[:, :width // 2]
    right_half = edges[:, width // 2:]
    left_edge_density = np.sum(left_half > 0)
    right_edge_density = np.sum(right_half > 0)
    curve_intensity = abs(left_edge_density - right_edge_density) / (left_edge_density + right_edge_density + 1e-6)
    return curve_intensity


def curve_speed_control(curve_intensity, current_speed_factor=1.0):
    speed_reduction = min(0.7, curve_intensity * 1.5)
    speed_adjustment = speed_reduction * current_speed_factor
    return speed_adjustment


if __name__ == "__main__":
    device = setup_device()
    print(f"使用设备: {device}")
    torch.autograd.set_detect_anomaly(False)

    env = gym.make('CarRacing-v3', render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ### MODIFICATION START: Adjusted hyperparameters ###
    actor_lr = 2e-5  # Slower learning for actor
    critic_lr = 7e-5  # Faster learning for critic
    gamma = 0.99
    batch_size = 128
    max_epoch = 30000
    # Exploration parameters for exponential decay
    initial_sigma = 0.4
    final_sigma = 0.05
    sigma_decay_rate = 0.997  # Slower decay
    current_sigma = initial_sigma
    ### MODIFICATION END ###

    max_step_per_epoch = 1000
    replay_capacity = 5000  # Increased capacity
    tau = 0.005  # Using a fixed small tau is often more stable

    agent = Agent(state_dim, action_dim, actor_lr, critic_lr, device)
    replay_buffer = ReplayBuffer(replay_capacity)
    transformer = transforms.ToTensor()
    writer = SummaryWriter('./data_optimized')
    total_steps = 0

    ### MODIFICATION START: Smoothing for curve detection ###
    recent_curve_intensities = deque(maxlen=5)
    ### MODIFICATION END ###

    for epoch in range(max_epoch):
        try:
            observation, info = env.reset()
            state = transformer(observation)
            episode_reward = 0
            continue_negative_reward_time = 0
            continue_out_of_road_time = 0
            update_time = 0
            episode_steps = 0

            ### MODIFICATION START: Update exploration noise sigma ###
            current_sigma = max(final_sigma, current_sigma * sigma_decay_rate)
            writer.add_scalar('training/sigma', current_sigma, epoch)
            ### MODIFICATION END ###

            for step in range(max_step_per_epoch):
                ### MODIFICATION START: Curriculum Learning Logic ###
                # Phase 1: Basic driving (Epoch 0-499)
                if epoch < 500:
                    agent.turn_speed_reduction = 0.0  # No auto-braking in curves
                # Phase 3: Advanced curve handling (Epoch 1500+)
                elif epoch >= 1500 and step > 5:
                    raw_curve_intensity = detect_curve_direction(observation)
                    recent_curve_intensities.append(raw_curve_intensity)
                    curve_intensity = np.mean(recent_curve_intensities) if recent_curve_intensities else 0.0
                    agent.turn_speed_reduction = curve_speed_control(curve_intensity)
                else:  # Phase 2 (500-1499)
                    agent.turn_speed_reduction = 0.0
                ### MODIFICATION END ###

                action = agent.select_action(state, sigma=current_sigma)
                next_state_image, reward, terminated, truncated, info = env.step(action)
                next_state = transformer(next_state_image)
                done = terminated or truncated

                episode_reward += reward
                if reward < -0.1:
                    continue_negative_reward_time += 1


                ### MODIFICATION START: Phased Reward Shaping ###
                # Phase 1: No extra reward, just env reward
                custom_reward = 0
                if epoch >= 500:  # Phase 2 and 3
                    in_road_reward = car_in_road_reward_simplified(next_state_image)
                    custom_reward = in_road_reward * 0.5  # Apply with a fixed weight

                reward += custom_reward

                if custom_reward < -0.1:
                    continue_out_of_road_time += 1
                else:
                    continue_out_of_road_time = 0
                ### MODIFICATION END ###

                replay_buffer.add(state, action, next_state, reward, done)
                if len(replay_buffer) > batch_size:
                    agent.train(replay_buffer, batch_size, gamma, tau, update_time)
                    update_time += 1

                if step % 20 == 0:
                    writer.add_scalar('action/steer', action[0], total_steps)
                    writer.add_scalar('action/gas', action[1], total_steps)
                    writer.add_scalar('action/brake', action[2], total_steps)

                state = next_state
                observation = next_state_image
                total_steps += 1
                episode_steps += 1

                if done or continue_negative_reward_time > 40:
                    break

            writer.add_scalar('episode/reward', episode_reward, epoch)
            writer.add_scalar('episode/length', episode_steps, epoch)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: Reward = {episode_reward:.2f}, Steps = {episode_steps}, Sigma = {current_sigma:.3f}")

            if epoch % 100 == 0 and epoch > 0:
                torch.save(agent.actor.state_dict(), f"./model/agent_opt_{epoch}.pt")
                print(f"Epoch {epoch}: Model saved.")

        except Exception as e:
            print(f"Epoch {epoch} 发生错误: {e}")
            continue

    torch.save(agent.actor.state_dict(), "./model/agent_opt_final.pt")
    print("训练完成，最终优化模型已保存")
    env.close()
