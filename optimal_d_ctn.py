import warnings
import os
import csv
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

# 全局 CSV 路径（所有 run 共享一张总表）
LOG_CSV_PATH = os.path.join("runs", "experiment_log.csv")


def append_experiment_log(row: dict):
    """
    往实验总表里追加一行。
    如果文件不存在，会自动写表头。
    """
    fieldnames = [
        "run_idx",
        "run_name",
        "epoch",
        "is_final",
        "model_path",
        "episode_reward",
        "best_reward_so_far",
        "total_steps",
        "sigma",

        # 训练超参数
        "actor_lr",
        "critic_lr",
        "gamma",
        "batch_size",
        "max_step_per_epoch",
        "replay_capacity",
        "sigma_decay_rate",

        # 探索&停止条件
        "initial_sigma",
        "final_sigma",
        "target_sigma",
        "min_epoch_before_stop",

        # 小车控制相关
        "brake_rate",

        # ROI & 弯道注意力相关
        "near_start_ratio",
        "far_end_ratio",
        "near_weight",
    ]

    os.makedirs(os.path.dirname(LOG_CSV_PATH), exist_ok=True)
    file_exists = os.path.exists(LOG_CSV_PATH)

    with open(LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_finished_runs_from_csv():
    """
    从全局 experiment_log.csv 里读出已经完成的 run_name 集合。
    判定标准：is_final == 1
    """
    finished = set()
    if not os.path.exists(LOG_CSV_PATH):
        return finished

    with open(LOG_CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                is_final = int(row.get("is_final", 0))
            except ValueError:
                is_final = 0
            if is_final == 1:
                run_name = row.get("run_name")
                if run_name:
                    finished.add(run_name)
    return finished


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


# ==== 弯道检测相关超参数（可以在不同实验里改）====
ROI_NEAR_START_RATIO = 0.4  # 近处从高度的哪一部分开始（0~1），默认从 1/2H 往下
ROI_FAR_END_RATIO = 1.0 - ROI_NEAR_START_RATIO  # 远处到高度的哪一部分结束（0~1），默认到 1/2H
NEAR_WEIGHT = 0.8  # 近处权重
FAR_WEIGHT = 1.0 - NEAR_WEIGHT  # 远处权重


def detect_curve_direction(image):
    height, width = image.shape[:2]

    # 根据全局超参数算出近/远的分界
    near_start = int(height * ROI_NEAR_START_RATIO)
    far_end = int(height * ROI_FAR_END_RATIO)

    # 近处：靠近车前的下方区域
    near_roi = image[near_start:, :, :]
    # 远处：道路远端的上方区域
    far_roi = image[:far_end, :, :]

    # roi = image[height // 2:, :, :]

    def curve_intensity_from_roi(roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 注意这里用 roi 的宽度
        h, w = edges.shape[:2]
        left_half = edges[:, :w // 2]
        right_half = edges[:, w // 2:]

        left_edge_density = np.sum(left_half > 0)
        right_edge_density = np.sum(right_half > 0)

        curve_intensity = abs(left_edge_density - right_edge_density) / (
                left_edge_density + right_edge_density + 1e-6
        )
        return curve_intensity

    near_curve = curve_intensity_from_roi(near_roi)
    far_curve = curve_intensity_from_roi(far_roi)

    # 用权重融合近/远弯道信息
    curve_intensity = NEAR_WEIGHT * near_curve + FAR_WEIGHT * far_curve
    return curve_intensity


def curve_speed_control(curve_intensity, current_speed_factor=1.0):
    speed_reduction = min(0.7, curve_intensity * 1.5)
    speed_adjustment = speed_reduction * current_speed_factor
    return speed_adjustment


def train_one_run(run_idx: int, config: dict, device):
    """
    单次训练：在一个独立目录下训练并保存模型。
    config 里可以放：actor_lr / critic_lr / initial_sigma / final_sigma / brake_rate 等等。
    """
    # 当前实验的配置的全局变量
    global ROI_NEAR_START_RATIO, ROI_FAR_END_RATIO, NEAR_WEIGHT, FAR_WEIGHT

    ROI_NEAR_START_RATIO = config.get("near_start_ratio", ROI_NEAR_START_RATIO)
    ROI_FAR_END_RATIO = 1 - ROI_NEAR_START_RATIO
    NEAR_WEIGHT = config.get("near_weight", NEAR_WEIGHT)
    FAR_WEIGHT = 1.0 - NEAR_WEIGHT
    # === 1. 输出目录 ===
    run_name = config.get("name", f"run_{run_idx:03d}")
    base_dir = os.path.join("runs", run_name)
    model_dir = os.path.join(base_dir, "model")
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"[{run_name}] 输出目录: {base_dir}")

    # === 2. 环境 & 超参数 ===
    # env = gym.make('CarRacing-v3', render_mode="human")
    env = gym.make('CarRacing-v3', render_mode="None")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor_lr = config.get("actor_lr", 2e-5)
    critic_lr = config.get("critic_lr", 7e-5)
    gamma = config.get("gamma", 0.99)
    batch_size = config.get("batch_size", 128)
    max_epoch = config.get("max_epoch", 30000)

    max_step_per_epoch = config.get("max_step_per_epoch", 1000)
    replay_capacity = config.get("replay_capacity", 5000)
    tau = config.get("tau", 0.005)

    # 提前停止条件：sigma 到达多少就停；以及最少训练多少 epoch 以后才允许停
    initial_sigma = config.get("initial_sigma", 0.4)
    final_sigma = config.get("final_sigma", 0.05)
    sigma_decay_rate = config.get("sigma_decay_rate", 0.997)
    target_sigma = config.get("target_sigma", final_sigma)
    min_epoch_before_stop = config.get("min_epoch_before_stop", 1000)

    # 小车控制参数
    brake_rate = config.get("brake_rate", 0.0)

    # ROI / 弯道权重参数（如果 config 不给就用默认值）
    near_start_ratio = config.get("near_start_ratio", ROI_NEAR_START_RATIO)
    far_end_ratio = config.get("far_end_ratio", ROI_FAR_END_RATIO)
    near_weight = config.get("near_weight", NEAR_WEIGHT)

    # === 3. Agent / Buffer / 日志 ===
    agent = Agent(state_dim, action_dim, actor_lr, critic_lr, device)
    # 这里可以用 config 里的参数来修改小车行为，例如刹车增强倍率：
    agent.brake_rate = config.get("brake_rate", 0.0)

    replay_buffer = ReplayBuffer(replay_capacity)
    transformer = transforms.ToTensor()
    writer = SummaryWriter(log_dir)
    total_steps = 0

    # 曲率平滑
    recent_curve_intensities = deque(maxlen=5)
    current_sigma = initial_sigma
    best_reward = float("-inf")

    # === 4. 主训练循环（基本沿用你原来的逻辑） ===
    for epoch in range(max_epoch):
        try:
            observation, info = env.reset()
            state = transformer(observation)
            episode_reward = 0
            continue_negative_reward_time = 0
            continue_out_of_road_time = 0
            update_time = 0
            episode_steps = 0

            # 更新探索噪声 sigma
            current_sigma = max(final_sigma, current_sigma * sigma_decay_rate)
            writer.add_scalar('training/sigma', current_sigma, epoch)

            for step in range(max_step_per_epoch):
                # === Curriculum Learning ===
                if epoch < 500:
                    agent.turn_speed_reduction = 0.0  # Phase 1
                elif epoch >= 1500 and step > 5:
                    raw_curve_intensity = detect_curve_direction(observation)
                    recent_curve_intensities.append(raw_curve_intensity)
                    curve_intensity = np.mean(recent_curve_intensities) if recent_curve_intensities else 0.0
                    agent.turn_speed_reduction = curve_speed_control(curve_intensity)
                else:  # Phase 2 (500-1499)
                    agent.turn_speed_reduction = 0.0

                action = agent.select_action(state, sigma=current_sigma)
                next_state_image, reward, terminated, truncated, info = env.step(action)
                next_state = transformer(next_state_image)
                done = terminated or truncated

                episode_reward += reward
                if reward < -0.1:
                    continue_negative_reward_time += 1

                # === 自定义路面奖励 ===
                custom_reward = 0
                if epoch >= 500:
                    in_road_reward = car_in_road_reward_simplified(next_state_image)
                    custom_reward = in_road_reward * 0.5

                reward += custom_reward

                if custom_reward < -0.1:
                    continue_out_of_road_time += 1
                else:
                    continue_out_of_road_time = 0

                # 存入经验回放
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

            # 日志 & 中间模型
            writer.add_scalar('episode/reward', episode_reward, epoch)
            writer.add_scalar('episode/length', episode_steps, epoch)

            # 更新本 run 的 best reward
            if episode_reward > best_reward:
                best_reward = episode_reward

            if epoch % 10 == 0:
                print(
                    f"[{run_name}] Epoch {epoch}: Reward = {episode_reward:.2f}, "
                    f"Steps = {episode_steps}, Sigma = {current_sigma:.3f}"
                )

            if epoch % 100 == 0 and epoch > 0:
                model_path = os.path.join(model_dir, f"agent_opt_{epoch}.pt")
                torch.save(agent.actor.state_dict(), model_path)
                print(f"[{run_name}] Epoch {epoch}: Model saved to {model_path}")

                # === 记录到全局 CSV ===
                append_experiment_log({
                    "run_idx": run_idx,
                    "run_name": run_name,
                    "epoch": epoch,
                    "is_final": 0,
                    "model_path": model_path,
                    "episode_reward": episode_reward,
                    "best_reward_so_far": best_reward,
                    "total_steps": total_steps,
                    "sigma": float(current_sigma),

                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                    "gamma": gamma,
                    "batch_size": batch_size,
                    "max_step_per_epoch": max_step_per_epoch,
                    "replay_capacity": replay_capacity,
                    "sigma_decay_rate": sigma_decay_rate,

                    "initial_sigma": initial_sigma,
                    "final_sigma": final_sigma,
                    "target_sigma": target_sigma,
                    "min_epoch_before_stop": min_epoch_before_stop,

                    "brake_rate": brake_rate,

                    "near_start_ratio": near_start_ratio,
                    "far_end_ratio": far_end_ratio,
                    "near_weight": near_weight,
                })

            # === 提前停止条件：sigma 足够小就结束本轮训练 ===
            if current_sigma <= target_sigma and epoch >= min_epoch_before_stop:
                print(f"[{run_name}] sigma 已到达 {current_sigma:.3f} <= 目标 {target_sigma:.3f}，结束本次训练。")
                break

        except Exception as e:
            print(f"[{run_name}] Epoch {epoch} 发生错误: {e}")
            continue

    # 最终模型
    final_path = os.path.join(model_dir, "agent_opt_final.pt")
    torch.save(agent.actor.state_dict(), final_path)
    print(f"[{run_name}] 训练完成，最终模型已保存到 {final_path}")

    # 最终模型也写一条记录（标记 is_final=1）
    append_experiment_log({
        "run_idx": run_idx,
        "run_name": run_name,
        "epoch": epoch,  # 最后一个 epoch
        "is_final": 1,
        "model_path": final_path,
        "episode_reward": episode_reward,
        "best_reward_so_far": best_reward,
        "total_steps": total_steps,
        "sigma": float(current_sigma),

        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "gamma": gamma,
        "batch_size": batch_size,
        "max_step_per_epoch": max_step_per_epoch,
        "replay_capacity": replay_capacity,
        "sigma_decay_rate": sigma_decay_rate,

        "initial_sigma": initial_sigma,
        "final_sigma": final_sigma,
        "target_sigma": target_sigma,
        "min_epoch_before_stop": min_epoch_before_stop,

        "brake_rate": brake_rate,

        "near_start_ratio": near_start_ratio,
        "far_end_ratio": far_end_ratio,
        "near_weight": near_weight,
    })
    env.close()
    writer.close()


if __name__ == "__main__":
    device = setup_device()
    print(f"使用设备: {device}")
    torch.autograd.set_detect_anomaly(False)

    # 这里列出你想要尝试的“小车参数配置”
    # 可以在每个 dict 里改 brake_rate、initial_sigma、final_sigma 等
    experiment_list = [
        {
            "name": "run_baseline",
            "brake_rate": 0.0,
            "initial_sigma": 0.4,
            "final_sigma": 0.05,
            "target_sigma": 0.05,
            "min_epoch_before_stop": 1500,
            "near_start_ratio": 0.5,
            "near_weight": 0.9,
        },
        {
            "name": "run_more_brake",
            "brake_rate": 0.5,  # 刹车更激进
            "initial_sigma": 0.4,
            "final_sigma": 0.03,
            "target_sigma": 0.03,
            "min_epoch_before_stop": 1500,
            "near_start_ratio": 0.5,
            "near_weight": 0.9,
        },
        {
            "name": "run_less_explore",
            "brake_rate": 0.2,
            "initial_sigma": 0.3,
            "final_sigma": 0.02,
            "target_sigma": 0.02,
            "min_epoch_before_stop": 2000,
            "near_start_ratio": 0.5,
            "near_weight": 0.9,
        }, {
            "name": "run_baseline",
            "brake_rate": 0.0,
            "initial_sigma": 0.4,
            "final_sigma": 0.05,
            "target_sigma": 0.05,
            "min_epoch_before_stop": 1500,
            "near_start_ratio": 0.5,
            "near_weight": 0.8,
        },
        {
            "name": "run_more_brake",
            "brake_rate": 0.5,  # 刹车更激进
            "initial_sigma": 0.4,
            "final_sigma": 0.03,
            "target_sigma": 0.03,
            "min_epoch_before_stop": 1500,
            "near_start_ratio": 0.5,
            "near_weight": 0.8,
        },
        {
            "name": "run_less_explore",
            "brake_rate": 0.2,
            "initial_sigma": 0.3,
            "final_sigma": 0.02,
            "target_sigma": 0.02,
            "min_epoch_before_stop": 2000,
            "near_start_ratio": 0.5,
            "near_weight": 0.8,
        }, {
            "name": "run_baseline",
            "brake_rate": 0.0,
            "initial_sigma": 0.4,
            "final_sigma": 0.05,
            "target_sigma": 0.05,
            "min_epoch_before_stop": 1500,
            "near_start_ratio": 0.5,
            "near_weight": 0.7,
        },
        {
            "name": "run_more_brake",
            "brake_rate": 0.5,  # 刹车更激进
            "initial_sigma": 0.4,
            "final_sigma": 0.03,
            "target_sigma": 0.03,
            "min_epoch_before_stop": 1500,
            "near_start_ratio": 0.5,
            "near_weight": 0.7,
        },
        {
            "name": "run_less_explore",
            "brake_rate": 0.2,
            "initial_sigma": 0.3,
            "final_sigma": 0.02,
            "target_sigma": 0.02,
            "min_epoch_before_stop": 2000,
            "near_start_ratio": 0.5,
            "near_weight": 0.7,
        },
    ]

    # 从 CSV 里读出已经完成的 run_name
    finished_runs = load_finished_runs_from_csv()
    print(f"已完成的 run: {finished_runs}")

    run_idx = 0
    while True:
        config = experiment_list[run_idx % len(experiment_list)]
        run_name = config.get("name", f"run_{run_idx:03d}")

        # 如果这个超参数组合已经有 is_final=1 的记录，就跳过
        if run_name in finished_runs:
            print(f"[{run_name}] 已在 CSV 中标记为完成，跳过。")
            run_idx += 1
            continue

        print(f"\n================ 开始第 {run_idx} 次训练: {run_name} ================\n")
        train_one_run(run_idx, config, device)

        # train_one_run 结束说明已经训练到 final，CSV 里也写了 is_final=1
        finished_runs.add(run_name)
        run_idx += 1
