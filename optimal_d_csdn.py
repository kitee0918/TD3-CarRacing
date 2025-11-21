import warnings
import cv2
import os
import csv
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
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, RecordVideo

FRAME_STACK = 4   # 堆叠 4 帧
IMG_SIZE = 84     # 将图像缩放到 84x84

warnings.filterwarnings("ignore")
# ===== 多实验 & CSV 日志配置 =====
LOG_CSV_PATH = os.path.join("runs", "experiment_log.csv")  # 全局实验总表

def append_experiment_log(row: dict):
    """
    往实验总表里追加一行。
    如果文件不存在，会自动写表头。
    """
    fieldnames = [
        "run_idx",
        "run_name",
        "seed",
        "episode",
        "is_final",            # 0=中间模型, 1=最终模型
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
        "num_episodes",
        "max_episode_steps",
        "replay_capacity",
        "tau",

        # 探索相关
        "initial_sigma",
        "min_sigma",
        "sigma_decay_factor",

        # 环境 / 预处理相关
        "env_name",
        "frame_stack",
        "img_size",
        "skip_frames",
    ]

    os.makedirs(os.path.dirname(LOG_CSV_PATH), exist_ok=True)
    file_exists = os.path.exists(LOG_CSV_PATH)

    with open(LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

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
            nn.Conv2d(FRAME_STACK, 16, kernel_size=3, padding=1),
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
            nn.Conv2d(FRAME_STACK, 16, kernel_size=3, padding=1),
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

class SkipFrame(gym.Wrapper):
    """简单的跳帧封装：同一动作重复 skip 次，奖励累加。"""
    def __init__(self, env, skip: int = 5):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        obs = None
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def make_car_env(render_mode=None):
    """
    构造带预处理的 CarRacing 环境：
    - 跳帧（SkipFrame）
    - 灰度图 GrayscaleObservation
    - 缩放为 84x84 ResizeObservation
    """
    env = gym.make("CarRacing-v3", render_mode=render_mode)
    env = SkipFrame(env, skip=5)                        # 对应参考代码的 skip=5
    env = GrayscaleObservation(env, keep_dim=False)     # 变成 (H, W) 灰度
    env = ResizeObservation(env, shape=(IMG_SIZE, IMG_SIZE))  # 变成 (84, 84)
    return env


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

            # 对齐参考工程：policy_noise = 0.2, policy_noise_clip = 0.5
            target_action = target_action + torch.clamp(0.2 * torch.randn_like(target_action),
                                                        -0.5, 0.5)

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
        # 对齐参考工程：delay_freq = 1（每次都更新 actor）
        actor_update_freq = 1
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

    near_roi = image[height  // 2:, :, :]# 近处（当前车前路面）：下半部分
    far_roi = image[:height  // 2 :, :] # 远处弯道：上半部分
    near_curve = curve_intensity_from_roi(near_roi)
    far_curve = curve_intensity_from_roi(far_roi)

    curve_intensity = 0.9 * near_curve + 0.1 * far_curve
    return curve_intensity


def curve_speed_control(curve_intensity, current_speed_factor=1.0):
    speed_reduction = min(0.7, curve_intensity * 1.5)
    speed_adjustment = speed_reduction * current_speed_factor
    return speed_adjustment


def render_one_episode(agent, transformer, max_episode_steps=1200):
    """
    用当前 agent 跑一局可视化（render_mode='human'），不训练，只看效果。
    """
    import gymnasium as gym

    env_vis = gym.make('CarRacing-v3', render_mode="human")
    obs, info = env_vis.reset()
    state = transformer(obs)
    done = False
    steps = 0
    episode_reward = 0.0

    while not done and steps < max_episode_steps:
        # 可视化的时候一般不加探索噪声，sigma=0.0
        action = agent.select_action(state, sigma=0.0)
        obs, reward, terminated, truncated, info = env_vis.step(action)
        state = transformer(obs)

        episode_reward += reward
        done = terminated or truncated
        steps += 1

    print(f"[Render] 可视化 episode_reward = {episode_reward:.2f}, steps = {steps}")
    env_vis.close()

def record_video_episode(agent, transformer, episode_idx: int,
                         max_episode_steps: int = 1200,
                         video_folder: str = "videos"):
    """
    用当前 agent 跑一局，录成 mp4 保存到 video_folder。
    不训练，只前向推理，sigma=0。
    """
    os.makedirs(video_folder, exist_ok=True)

    # 用 rgb_array 渲染 + RecordVideo 包装
    env_vis = gym.make("CarRacing-v3", render_mode="rgb_array")
    env_vis = RecordVideo(
        env_vis,
        video_folder=video_folder,
        episode_trigger=lambda ep: True,           # 我们只跑一局，录这一局即可
        name_prefix=f"car_td3_ep_{episode_idx:06d}"
    )

    obs, info = env_vis.reset()
    state = transformer(obs)
    done = False
    steps = 0
    episode_reward = 0.0

    while not done and steps < max_episode_steps:
        # 录视频时一般不加探索噪声，sigma=0.0
        action = agent.select_action(state, sigma=0.0)
        obs, reward, terminated, truncated, info = env_vis.step(action)
        state = transformer(obs)

        episode_reward += reward
        done = terminated or truncated
        steps += 1

    env_vis.close()
    print(f"[RecordVideo] episode {episode_idx}: reward={episode_reward:.2f}, "
          f"steps={steps}, saved to '{video_folder}'")

def record_eval_video(run_name: str, episode: int, agent, device: str,
                      max_episode_steps: int = 1000):
    """
    用当前 agent 在一个带视频录制的环境里跑一局，
    保存 mp4 到 runs/{run_name}/videos 下。
    """
    video_dir = os.path.join("runs", run_name, "videos")
    os.makedirs(video_dir, exist_ok=True)

    # 评估环境：和训练一样的预处理，但 render_mode='rgb_array' 以便 RecordVideo 录屏
    eval_env = make_car_env(render_mode="rgb_array")
    eval_env = RecordVideo(
        eval_env,
        video_folder=video_dir,
        name_prefix=f"ep_{episode}",
        episode_trigger=lambda e: True  # 这个 env 只跑一局，所以每次都录
    )

    transformer = transforms.ToTensor()
    frame_stack = deque(maxlen=FRAME_STACK)

    obs, info = eval_env.reset()
    first_frame = transformer(obs)  # 灰度 (84,84) -> (1,84,84)

    frame_stack.clear()
    for _ in range(FRAME_STACK):
        frame_stack.append(first_frame)

    state = torch.cat(list(frame_stack), dim=0)  # (4,84,84)
    total_reward = 0.0

    # 评估时一般不加噪声：sigma=0
    for step in range(max_episode_steps):
        action = agent.select_action(state, sigma=0.0)
        next_obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward

        next_frame = transformer(next_obs)
        frame_stack.append(next_frame)
        state = torch.cat(list(frame_stack), dim=0)

        done = terminated or truncated
        if done:
            break

    eval_env.close()
    print(
        f"[{run_name}] 评估视频 Episode {episode} 总奖励 = {total_reward:.2f}，"
        f"已保存到 {video_dir}"
    )



def train_one_run(run_idx: int, device: str, sigma_decay_factor: float):
    """
    单次完整训练：
    - 使用固定的一套超参数（当前参考工程的配置）
    - 在 runs/run_xxx 下保存模型和 TensorBoard 日志
    - 在全局 CSV 中记录中间模型和最终模型的信息
    """
    env_name = "CarRacing-v3"

    # === 每个 run 独立的输出目录 ===
    # run_name = f"run_{run_idx:03d}"
    run_name = f"run_{run_idx:03d}_decay_{sigma_decay_factor:.6f}"
    base_dir = os.path.join("runs", run_name)
    model_dir = os.path.join(base_dir, "model")
    tb_dir = os.path.join(base_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    print(f"[{run_name}] 输出目录: {base_dir}")

    # === 为可复现性设置 seed（也写进 CSV） ===
    seed = 42 + run_idx
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 使用预处理后的 CarRacing 环境：跳帧 + 灰度 + 84x84
    env = make_car_env(render_mode=None)
    try:
        env.reset(seed=seed)
    except TypeError:
        # 旧版 gymnasium 可能 reset 不支持 seed 参数，忽略即可
        pass

    # 预处理后的观测空间是 (84, 84)，我们用高度作为 state_dim
    obs_shape = env.observation_space.shape  # (84, 84)
    state_dim = obs_shape[0]
    action_dim = env.action_space.shape[0]

    # ===== TD3 超参数（统一定死为“当前最佳”） =====
    actor_lr = 2.5e-4
    critic_lr = 1e-3
    gamma = 0.99
    batch_size = 128
    num_episodes = 15000
    max_episode_steps = 1200

    replay_capacity = 102400   # 经验回放容量 ~1e5
    tau = 0.05                 # target 网络软更新系数

    # 探索噪声：expl_noise = 0.5, expl_noise_exp_reduce_factor = 1 - 1e-4
    initial_sigma = 0.5
    min_sigma = 0.05
    # sigma_decay_factor = 1 - 1e-4
    current_sigma = initial_sigma

    # ===== 初始化 Agent / Buffer / 状态堆叠 =====
    agent = Agent(state_dim, action_dim, actor_lr, critic_lr, device)
    replay_buffer = ReplayBuffer(replay_capacity)
    transformer = transforms.ToTensor()

    # 手动 FrameStack：存储最近 FRAME_STACK 帧 (每帧 shape = (1,84,84))
    frame_stack = deque(maxlen=FRAME_STACK)

    writer = SummaryWriter(tb_dir)
    total_steps = 0
    best_reward = float("-inf")

    for episode in range(num_episodes):
        try:
            obs, info = env.reset()

            # 第一帧 -> Tensor (1,84,84)
            first_frame = transformer(obs)
            frame_stack.clear()
            for _ in range(FRAME_STACK):
                frame_stack.append(first_frame)

            # 堆叠 4 帧得到状态 (4,84,84)
            state = torch.cat(list(frame_stack), dim=0)

            episode_reward = 0.0
            episode_steps = 0
            update_time = 0  # 用于 TD3 的 delay_freq（你在 Agent.train 里已经用到）

            # 更新探索强度（指数衰减）
            current_sigma = max(min_sigma, current_sigma * sigma_decay_factor)
            writer.add_scalar('training/sigma', current_sigma, episode)

            for step in range(max_episode_steps):
                # 不额外做手工控制逻辑，交给 TD3 自己学
                agent.turn_speed_reduction = 0.0

                action = agent.select_action(state, sigma=current_sigma)
                next_obs, reward, terminated, truncated, info = env.step(action)

                next_frame = transformer(next_obs)
                frame_stack.append(next_frame)
                next_state = torch.cat(list(frame_stack), dim=0)

                done = terminated or truncated

                # 使用环境原始 reward
                episode_reward += reward

                # 存入经验回放
                replay_buffer.add(state, action, next_state, reward, done)
                if len(replay_buffer) > batch_size:
                    agent.train(replay_buffer, batch_size, gamma, tau, update_time)
                    update_time += 1

                # 记录动作
                if step % 20 == 0:
                    writer.add_scalar('action/steer', action[0], total_steps)
                    writer.add_scalar('action/gas', action[1], total_steps)
                    writer.add_scalar('action/brake', action[2], total_steps)

                state = next_state
                total_steps += 1
                episode_steps += 1

                if done:
                    break

            # 记录每个 episode 的统计
            writer.add_scalar('episode/reward', episode_reward, episode)
            writer.add_scalar('episode/length', episode_steps, episode)

            # 更新 best_reward
            if episode_reward > best_reward:
                best_reward = episode_reward

            if episode % 10 == 0:
                print(
                    f"[{run_name}] Episode {episode}: Reward = {episode_reward:.2f}, "
                    f"Steps = {episode_steps}, Sigma = {current_sigma:.3f}"
                )

            # === 每 50 个 episode 录制一个可视化视频 ===
            if episode % 50 == 0 and episode > 0:
                print(f"[{run_name}] Episode {episode}: 开始录制评估视频...")
                record_eval_video(
                    run_name=run_name,
                    episode=episode,
                    agent=agent,
                    device=device,
                    max_episode_steps=max_episode_steps  # 可以和训练一样，也可以单独设
                )

            # 每 100 个 episode 保存一次模型，并记一条 CSV 记录
            if episode % 100 == 0 and episode > 0:
                model_path = os.path.join(model_dir, f"agent_opt_{episode}.pt")
                torch.save(agent.actor.state_dict(), model_path)
                print(f"[{run_name}] Episode {episode}: Model saved to {model_path}")

                append_experiment_log({
                    "run_idx": run_idx,
                    "run_name": run_name,
                    "seed": seed,
                    "episode": episode,
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
                    "num_episodes": num_episodes,
                    "max_episode_steps": max_episode_steps,
                    "replay_capacity": replay_capacity,
                    "tau": tau,

                    "initial_sigma": initial_sigma,
                    "min_sigma": min_sigma,
                    "sigma_decay_factor": sigma_decay_factor,

                    "env_name": env_name,
                    "frame_stack": FRAME_STACK,
                    "img_size": IMG_SIZE,
                    "skip_frames": 5,
                })

        except Exception as e:
            print(f"[{run_name}] Episode {episode} 发生错误: {e}")
            continue

    # ===== 最终模型保存 & CSV 记录 =====
    final_path = os.path.join(model_dir, "agent_opt_final.pt")
    torch.save(agent.actor.state_dict(), final_path)
    print(f"[{run_name}] 训练完成，最终模型已保存到 {final_path}")

    append_experiment_log({
        "run_idx": run_idx,
        "run_name": run_name,
        "seed": seed,
        "episode": episode,
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
        "num_episodes": num_episodes,
        "max_episode_steps": max_episode_steps,
        "replay_capacity": replay_capacity,
        "tau": tau,

        "initial_sigma": initial_sigma,
        "min_sigma": min_sigma,
        "sigma_decay_factor": sigma_decay_factor,

        "env_name": env_name,
        "frame_stack": FRAME_STACK,
        "img_size": IMG_SIZE,
        "skip_frames": 5,
    })

    env.close()
    writer.close()

if __name__ == "__main__":
    device = setup_device()
    print(f"使用设备: {device}")
    torch.autograd.set_detect_anomaly(False)

    # 想要搜索的一组 sigma_decay_factor
    sigma_decay_list = [
        1.0 - 1e-2,  # 0.990  衰减很快
        1.0 - 5e-3,  # 0.995
        1.0 - 1e-3,   # 0.9990  衰减快
        1.0 - 5e-4,   # 0.9995
        1.0 - 1e-4,   # 0.9999  参考工程的默认值
        # 1.0 - 5e-5,   # 0.99995
        # 1.0 - 1e-5,   # 0.99999 衰减非常慢
    ]

    run_idx = 0
    while True:
        sigma_decay_factor = sigma_decay_list[run_idx % len(sigma_decay_list)]
        print(
            f"\n================ 开始第 {run_idx} 次训练 "
            f"(sigma_decay_factor = {sigma_decay_factor:.6f}) ================\n"
        )
        train_one_run(run_idx, device, sigma_decay_factor)
        run_idx += 1

