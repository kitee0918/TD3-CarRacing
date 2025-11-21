import os
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms as transforms

# 从你的训练脚本里导入现成的组件
from optimal_d_csdn import Agent, make_car_env, setup_device, FRAME_STACK


# ======= 配置：修改这里选择要演示的模型 =======
# 例子：请按你自己的目录修改
# 比如 runs/run_000_decay_0.999900/model/agent_opt_final.pt
MODEL_PATH = r"runs/run_000_decay_0.999900/model/agent_opt_500.pt"

# 评估时每局最多跑多少步（可以比训练时的小一点）
MAX_EPISODE_STEPS = 1200


def load_agent(model_path: str, device: torch.device) -> Agent:
    """
    构造一个 Agent，只加载 Actor 权重用于推理（不训练）。
    """
    # 用预处理后的环境推一遍，拿到 state_dim / action_dim
    # 这里不用 render，纯拿空间信息
    tmp_env = make_car_env(render_mode=None)
    obs_shape = tmp_env.observation_space.shape  # (84,84)
    state_dim = obs_shape[0]
    action_dim = tmp_env.action_space.shape[0]
    tmp_env.close()

    # 这里 lr 随便给个值，反正不训练
    agent = Agent(state_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3, device=device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    agent.actor.load_state_dict(state_dict)
    agent.actor.eval()
    print(f"✅ 已加载模型: {model_path}")
    return agent


def run_demo(model_path: str):
    device = setup_device()

    # 加载训练好的 Agent
    agent = load_agent(model_path, device)

    # 用于可视化演示的环境：render_mode='human' 会弹出窗口
    env = make_car_env(render_mode="human")

    transformer = transforms.ToTensor()
    frame_stack = deque(maxlen=FRAME_STACK)

    episode_idx = 0

    try:
        while True:
            obs, info = env.reset()
            first_frame = transformer(obs)  # (1,84,84)
            frame_stack.clear()
            for _ in range(FRAME_STACK):
                frame_stack.append(first_frame)

            state = torch.cat(list(frame_stack), dim=0)  # (4,84,84)
            episode_reward = 0.0

            for step in range(MAX_EPISODE_STEPS):
                # 推理时不加噪声：sigma=0.0
                action = agent.select_action(state, sigma=0.0)

                next_obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                next_frame = transformer(next_obs)
                frame_stack.append(next_frame)
                state = torch.cat(list(frame_stack), dim=0)

                done = terminated or truncated
                if done:
                    break

            print(f"[DEMO] Episode {episode_idx} 结束，reward = {episode_reward:.2f}")
            episode_idx += 1

    except KeyboardInterrupt:
        print("\n收到中断信号，结束演示。")
    finally:
        env.close()


if __name__ == "__main__":
    run_demo(MODEL_PATH)
