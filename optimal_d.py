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


device = setup_device()


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

    def forward(self, state: Union[np.ndarray, torch.Tensor, list], sigma=0, brake_rate=1.0):
        # 兼容多种输入
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        elif isinstance(state, list):
            state = torch.FloatTensor(state)

        if len(state.shape) == 3:
            state = state.unsqueeze(0)

        # 提取特征
        h = self.conv1(state)
        h = self.down1(h)
        h = self.conv2(h)
        h = self.down2(h)
        h = self.conv3(h)
        h = self.down3(h)
        h = torch.flatten(h, start_dim=1)

        # 基础动作向量（带噪声）
        h = self.linear(h)
        h = h + torch.randn_like(h) * sigma

        # 修复：避免inplace操作，创建新的张量
        steer = h[:, 0]  # 方向
        gas = (h[:, 1] + 1) * 0.5 + 0.1  # 油门：[-1,1] -> [0.1,1.1]
        brake = (h[:, 2] + 1) * 0.015  # 刹车：很小的值

        # 裁剪到合理范围
        steer = torch.clamp(steer, -1.0, 1.0)
        gas = torch.clamp(gas, 0.0, 1.0)
        brake = torch.clamp(brake, 0.0, 1.0)

        # 使用torch.stack创建新张量，避免inplace操作
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
        # 确保输入是张量
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

        # extract the state features
        state_h = self.conv1(state)
        state_h = self.down1(state_h)
        state_h = self.conv2(state_h)
        state_h = self.down2(state_h)
        state_h = self.conv3(state_h)
        state_h = self.down3(state_h)
        state_h = torch.flatten(state_h, start_dim=1)

        state_h = self.state_linear(state_h)
        # action features
        action_h = self.action_linear(action)

        # concat
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
    def __init__(self, state_dim: int, action_dim: int, lr: float, device="cpu") -> None:
        self.brake_rate = 0
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

        # 设置不同的学习率
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        self.critic_mse_loss = nn.MSELoss().to(device)

    def select_action(self, state, sigma=0):
        with torch.no_grad():  # 确保不计算梯度
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)

            if len(state.shape) == 3:
                state = state.unsqueeze(0)

            action = self.actor(state.to(self.device), sigma=sigma, brake_rate=self.brake_rate)
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

        # 计算当前Q值
        predicted_q1 = self.critic1(state, action)
        predicted_q2 = self.critic2(state, action)

        # 计算目标Q值（修复：确保使用detach）
        with torch.no_grad():
            target_action = self.target_actor(next_state, brake_rate=self.brake_rate)
            # 添加少量噪声进行平滑
            target_action = target_action + 0.005 * torch.randn_like(target_action)
            target_action = torch.clamp(target_action, -1.0, 1.0)

            target_q1 = self.target_critic1(next_state, target_action)
            target_q2 = self.target_critic2(next_state, target_action)
            target_q = torch.min(target_q1, target_q2)
            TD_target = reward.unsqueeze(1) + gamma * target_q * (1 - done.unsqueeze(1))

        # 更新critic网络
        critic1_loss = self.critic_mse_loss(predicted_q1, TD_target)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)  # 梯度裁剪
        self.critic1_optimizer.step()

        critic2_loss = self.critic_mse_loss(predicted_q2, TD_target)
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)  # 梯度裁剪
        self.critic2_optimizer.step()

        # 每3步更新一次actor
        if update_time % 3 == 0:
            predicted_action = self.actor(state, brake_rate=self.brake_rate)
            actor_loss = -self.critic1(state, predicted_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)  # 梯度裁剪
            self.actor_optimizer.step()

            # 软更新目标网络
            with torch.no_grad():
                for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def car_in_road_reward(image):
    # 计算小车得分，按照小车附近绿色和灰色像素比来确定reward
    part_image = image[60:84, 40:60, :]
    # 定义道路、小车和草地的颜色范围
    road_color_lower = np.array([90, 90, 90], dtype=np.uint8)
    road_color_upper = np.array([120, 120, 120], dtype=np.uint8)
    grass_color_lower = np.array([90, 180, 90], dtype=np.uint8)
    grass_color_upper = np.array([120, 255, 120], dtype=np.uint8)

    # 根据颜色范围创建掩膜
    road_mask = cv2.inRange(part_image, road_color_lower, road_color_upper)
    grass_mask = cv2.inRange(part_image, grass_color_lower, grass_color_upper)

    # 计算像素数量
    road_pixel_count = cv2.countNonZero(road_mask)
    grass_pixel_count = cv2.countNonZero(grass_mask)

    if road_pixel_count + grass_pixel_count == 0:
        return 0

    reward = (road_pixel_count) / (road_pixel_count + grass_pixel_count) - 0.5
    return reward * 6


if __name__ == "__main__":
    device = setup_device()
    print(f"使用设备: {device}")

    # 启用异常检测来帮助调试
    torch.autograd.set_detect_anomaly(False)  # 设置为True可以检测问题，但会变慢

    env = gym.make('CarRacing-v3', render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 超参数
    lr = 4.5e-5
    gamma = 0.985
    batch_size = 96
    max_epoch = 2000
    max_step_per_epoch = 10000
    replay_capacity = 2000
    tau_initial = 0.5
    tau_final = 0.001
    tau_decay = 0.0005

    agent = Agent(state_dim, action_dim, lr, device)
    replay_buffer = ReplayBuffer(replay_capacity)
    transformer = transforms.ToTensor()
    writer = SummaryWriter('./data')

    total_steps = 0

    for epoch in range(max_epoch):
        try:
            observation, info = env.reset()
            state = transformer(observation)
            episode_reward = 0
            continue_negative_reward_time = 0
            continue_out_of_road_time = 0
            tau = max(tau_final, tau_initial - epoch * tau_decay)
            update_time = 0

            for step in range(max_step_per_epoch):
                action = agent.select_action(state, sigma=0.1 * (1 - epoch / max_epoch))

                next_state_image, reward, terminated, truncated, info = env.step(action)
                next_state = transformer(next_state_image)

                done = terminated or truncated

                if reward < 0:
                    continue_negative_reward_time += 1
                else:
                    continue_negative_reward_time = 0

                episode_reward += reward

                if step > 30:
                    raw_reward = car_in_road_reward(next_state_image)
                    in_road_reward = min(0.01, raw_reward)

                    if raw_reward < 0:
                        continue_out_of_road_time += 1
                    else:
                        continue_out_of_road_time = 0

                    reward += in_road_reward

                if step % 3 == 0:
                    replay_buffer.add(state, action, next_state, reward, done)
                    if len(replay_buffer) > batch_size:
                        agent.train(replay_buffer, batch_size, gamma, tau, update_time)
                        update_time += 1

                    # 记录动作价值
                    with torch.no_grad():
                        action_value1 = agent.critic1(
                            state.unsqueeze(0).to(device),
                            torch.FloatTensor(action).unsqueeze(0).to(device)
                        )
                        action_value2 = agent.critic2(
                            state.unsqueeze(0).to(device),
                            torch.FloatTensor(action).unsqueeze(0).to(device)
                        )
                        writer.add_scalar('action value 1', action_value1, total_steps)
                        writer.add_scalar('action value 2', action_value2, total_steps)

                state = next_state
                total_steps += 1

                # 终止条件
                if done or episode_reward < -5 or continue_negative_reward_time > 60 or continue_out_of_road_time > 10:
                    writer.add_scalar('episode reward', episode_reward, epoch)
                    agent.brake_rate = min(1, epoch / max_epoch * 0.5)
                    break

            # 每100轮保存模型
            if epoch % 100 == 0 and epoch > 0:
                torch.save(agent.actor.state_dict(), f"./model/agent{epoch}.pt")
                print(f"Epoch {epoch}: 奖励 = {episode_reward:.2f}, 模型已保存")

            # 每10轮打印进度
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: 奖励 = {episode_reward:.2f}, 步数 = {step}")

        except Exception as e:
            print(f"Epoch {epoch} 发生错误: {e}")
            # 继续训练而不是停止
            continue

    # 保存最终模型
    torch.save(agent.actor.state_dict(), "./model/agent_final.pt")
    print("训练完成，最终模型已保存")
    env.close()