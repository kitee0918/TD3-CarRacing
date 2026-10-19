# file: td3_car_racing_train_parallel_drivefix.py
# coding: utf-8
"""
TD3 for CarRacing-v3 — drive-fix edition
- NUM_ENVS=12, UPS=1, SyncVectorEnv (Windows 稳定)
- 内置 GrayScale(84x84)+FrameStack(4)，无依赖也可跑
- 交互探索：高斯噪声 + 退火
- 动作约束：ONE-PEDAL（油门/刹车互斥）+ 动作平滑（EMA）
- Actor 轻正则：惩罚大幅转向与刹车，鼓励直行
- 真实吞吐日志：fps~...
"""

import os, time, random
from typing import Callable, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv  # Windows 更稳

# ============ 环境/工具 ============
def env_int(name, d):
    try: return int(os.environ.get(name, d))
    except: return d
def env_f(name, d):
    try: return float(os.environ.get(name, d))
    except: return d
def set_seed_all(seed):
    random.seed(seed); np.random.seed(seed)

# FrameStack 兜底
try:
    from gymnasium.wrappers import FrameStack as _FrameStack
except Exception:
    try:
        from gymnasium.wrappers.frame_stack import FrameStack as _FrameStack  # type: ignore
    except Exception:
        _FrameStack = None
if _FrameStack is None:
    from collections import deque as _deque
    class FrameStack(gym.Wrapper):
        def __init__(self, env, k: int):
            super().__init__(env); self.k=int(k); self.frames=_deque(maxlen=self.k)
            assert isinstance(env.observation_space, spaces.Box)
            low, high = env.observation_space.low, env.observation_space.high
            self.observation_space = spaces.Box(
                low=np.repeat(low, self.k, axis=-1),
                high=np.repeat(high, self.k, axis=-1),
                dtype=env.observation_space.dtype,
            )
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs); self.frames.clear()
            for _ in range(self.k): self.frames.append(obs)
            return np.concatenate(list(self.frames), axis=-1), info
        def step(self, action):
            obs, r, term, trunc, info = self.env.step(action); self.frames.append(obs)
            return np.concatenate(list(self.frames), axis=-1), r, term, trunc, info
else:
    FrameStack = _FrameStack

def wrap_obs(env: gym.Env) -> gym.Env:
    """灰度 + 84x84（优先官方，缺失则本地实现）"""
    try:
        from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, (84, 84))
        return env
    except Exception:
        from gymnasium import ObservationWrapper
        class GrayResize84(ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                self.observation_space = spaces.Box(low=0, high=255, shape=(84,84,1), dtype=np.uint8)
            def observation(self, obs):
                g = (0.299*obs[...,0] + 0.587*obs[...,1] + 0.114*obs[...,2]).astype(np.float32)
                try:
                    import cv2; g = cv2.resize(g, (84,84), interpolation=cv2.INTER_AREA)
                except Exception:
                    ys=(np.linspace(0, g.shape[0]-1, 84)).astype(np.int32)
                    xs=(np.linspace(0, g.shape[1]-1, 84)).astype(np.int32)
                    g = g[np.ix_(ys,xs)]
                return np.clip(g,0,255).astype(np.uint8)[...,None]
        return GrayResize84(env)

def make_env_fn(seed: int) -> Callable[[], gym.Env]:
    env_id = os.getenv("ENV_ID", "CarRacing-v3")
    def _thunk():
        env = gym.make(env_id, render_mode=None, continuous=True)
        env = wrap_obs(env)
        env = FrameStack(env, 4)
        env.reset(seed=seed)
        return env
    return _thunk

# ============ 回放缓存 ============
class ReplayBuffer:
    def __init__(self, cap: int, obs_shape: Tuple[int,int,int], act_dim: int):
        self.cap=int(cap); self.ptr=0; self.full=False
        h,w,c = obs_shape
        self.S  = np.empty((cap,h,w,c), np.float32)
        self.A  = np.empty((cap,act_dim), np.float32)
        self.NS = np.empty((cap,h,w,c), np.float32)
        self.R  = np.empty((cap,1), np.float32)
        self.D  = np.empty((cap,1), np.float32)
    def add(self, s,a,ns,r,d):
        i=self.ptr; self.S[i]=s; self.A[i]=a; self.NS[i]=ns; self.R[i]=r; self.D[i]=d
        self.ptr+=1;
        if self.ptr>=self.cap: self.ptr=0; self.full=True
    def sample(self, n: int):
        size=self.cap if self.full else self.ptr
        idx=np.random.randint(0,size,size=n)
        return (self.S[idx], self.A[idx], self.NS[idx], self.R[idx], self.D[idx])
    def __len__(self): return self.cap if self.full else self.ptr

# ============ 主程序 ============
def main():
    # ---- 默认参数（可用环境变量覆写）----
    NUM_ENVS      = env_int("NUM_ENVS", 12)
    TRAIN_BATCH   = env_int("TRAIN_BATCH_SIZE", 704)
    TOTAL_STEPS   = env_int("TOTAL_STEPS", 100_000)
    START_RANDOM  = env_int("START_RANDOM_STEPS", 1_000)
    TRAIN_AFTER   = env_int("TRAIN_AFTER", 3_000)
    TRAIN_FREQ    = env_int("TRAIN_FREQ", 1)
    UPS           = env_int("UPS", 1)
    POLICY_DELAY  = env_int("POLICY_DELAY", 2)
    GAMMA         = env_f("GAMMA", 0.99)
    TAU           = env_f("TAU", 0.005)
    LR_ACTOR      = env_f("LR_ACTOR", 3e-4)
    LR_CRITIC     = env_f("LR_CRITIC", 3e-4)
    REPLAY_CAP    = env_int("REPLAY_CAP", 120_000)
    LOG_EVERY     = env_int("LOG_EVERY", 256)
    SAVE_EVERY    = env_int("SAVE_EVERY", 50_000)
    ACTION_REPEAT = env_int("ACTION_REPEAT", 2)
    # 驾驶相关增强
    STEER_MAX     = env_f("STEER_MAX", 0.8)         # 稍收一点转向上限，避免早期打大角
    ONE_PEDAL     = env_int("ONE_PEDAL", 1) == 1    # 油门/刹车互斥
    SMOOTH_ALPHA  = env_f("ACTION_SMOOTH", 0.2)     # 动作平滑 EMA 系数
    # 探索噪声（线性退火到 min）
    N_STEER, N_GAS, N_BRAKE = env_f("NOISE_STEER", 0.35), env_f("NOISE_GAS", 0.15), env_f("NOISE_BRAKE", 0.05)
    N_MIN                 = env_f("NOISE_MIN", 0.05)
    NOISE_ANNEAL_STEPS    = env_int("NOISE_ANNEAL_STEPS", 250_000)
    # Actor 轻正则（惩罚转向和刹车）
    L2_STEER, L2_BRAKE = env_f("L2_STEER", 1e-3), env_f("L2_BRAKE", 5e-3)
    SEED          = env_int("SEED", 0)

    set_seed_all(SEED)

    # Torch（延迟导入）
    import torch, torch.nn as nn, torch.nn.functional as F
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try: torch.set_float32_matmul_precision("high")
    except: pass

    # Envs
    envs = SyncVectorEnv([make_env_fn(SEED+i) for i in range(NUM_ENVS)])
    obs, _ = envs.reset(seed=SEED)
    assert obs.shape[1:] == (84,84,4), f"unexpected obs shape {obs.shape}"

    # Models
    class ConvBody(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(4,32,8,4), nn.ReLU(),
                nn.Conv2d(32,64,4,2), nn.ReLU(),
                nn.Conv2d(64,64,3,1), nn.ReLU(),
            ); self.out_dim=64*7*7
        def forward(self,x): x=self.net(x); return x.view(x.size(0),-1)

    class Actor(nn.Module):
        def __init__(self):
            super().__init__(); self.body=ConvBody()
            self.fc = nn.Sequential(nn.Linear(self.body.out_dim,512), nn.ReLU(),
                                    nn.Linear(512,256), nn.ReLU(), nn.Linear(256,3))
        def forward(self,s):
            z=self.body(s); out=self.fc(z)
            steer=torch.tanh(out[:, :1]) * STEER_MAX
            gas  =torch.sigmoid(out[:, 1:2])
            brake=torch.sigmoid(out[:, 2:3])
            return torch.cat([steer,gas,brake], dim=1)

    class Critic(nn.Module):
        def __init__(self):
            super().__init__(); self.body=ConvBody()
            self.q1=nn.Sequential(nn.Linear(self.body.out_dim+3,512), nn.ReLU(),
                                  nn.Linear(512,256), nn.ReLU(), nn.Linear(256,1))
            self.q2=nn.Sequential(nn.Linear(self.body.out_dim+3,512), nn.ReLU(),
                                  nn.Linear(512,256), nn.ReLU(), nn.Linear(256,1))
        def forward(self,s,a):
            z=self.body(s); x=torch.cat([z,a], dim=1); return self.q1(x), self.q2(x)
        def q1_only(self,s,a):
            z=self.body(s); x=torch.cat([z,a], dim=1); return self.q1(x)

    actor = Actor().to(device); critic = Critic().to(device)
    actor_t = Actor().to(device); critic_t = Critic().to(device)
    actor_t.load_state_dict(actor.state_dict()); critic_t.load_state_dict(critic.state_dict())
    opt_actor = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
    opt_critic= torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)
    scaler = torch.amp.GradScaler(device, enabled=(device=="cuda"))
    mse = nn.MSELoss()
    rb = ReplayBuffer(REPLAY_CAP, obs_shape=(84,84,4), act_dim=3)

    def to_t(s_np: np.ndarray) -> torch.Tensor:
        s=torch.from_numpy(s_np).to(device=device, dtype=torch.float32)/255.0
        return s.permute(0,3,1,2).contiguous()

    # 日志
    last_log_step=0; t0=time.time()
    def log_if_needed(step:int):
        nonlocal last_log_step,t0
        if step-last_log_step>=LOG_EVERY:
            dt=max(1e-6, time.time()-t0); fps=(step-last_log_step)/dt
            print(f"step={step} | fps~{fps:.1f} | rb={len(rb)}")
            last_log_step=step; t0=time.time()

    # 动作后处理：互斥+平滑
    prev_actions = np.zeros((NUM_ENVS, 3), dtype=np.float32)
    def post_process_actions(a: np.ndarray) -> np.ndarray:
        # 裁剪
        a[:,0] = np.clip(a[:,0], -STEER_MAX, STEER_MAX)
        a[:,1:] = np.clip(a[:,1:], 0.0, 1.0)
        # ONE-PEDAL
        if ONE_PEDAL:
            m_b = a[:,2] > 0.10; a[m_b,1] = 0.0
            m_g = a[:,1] > 0.20; a[m_g,2] = 0.0
        # 平滑（EMA）
        nonlocal prev_actions
        a = SMOOTH_ALPHA * a + (1.0 - SMOOTH_ALPHA) * prev_actions
        prev_actions = a
        return a

    # 训练
    def train_td3(step:int):
        if len(rb) < max(TRAIN_AFTER, TRAIN_BATCH): return
        for _ in range(UPS):
            S,A,NS,R,D = rb.sample(TRAIN_BATCH)
            S_t,NS_t = to_t(S), to_t(NS)
            A_t = torch.from_numpy(A).to(device, dtype=torch.float32)
            R_t = torch.from_numpy(R).to(device, dtype=torch.float32)
            D_t = torch.from_numpy(D).to(device, dtype=torch.float32)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device=="cuda")):
                with torch.no_grad():
                    a_targ = actor_t(NS_t)
                    noise  = (torch.randn_like(a_targ) * 0.1).clamp_(-0.3,0.3)
                    a_targ = a_targ + noise
                    a_targ[:,0].clamp_(-STEER_MAX, STEER_MAX)
                    a_targ[:,1:].clamp_(0.0,1.0)
                    if ONE_PEDAL:
                        mb = a_targ[:,2] > 0.10; a_targ[mb,1] = 0.0
                        mg = a_targ[:,1] > 0.20; a_targ[mg,2] = 0.0
                    q1n,q2n = critic_t(NS_t, a_targ)
                    y = R_t + (1.0 - D_t) * GAMMA * torch.minimum(q1n,q2n)
                q1,q2 = critic(S_t, A_t)
                critic_loss = mse(q1,y) + mse(q2,y)
            opt_critic.zero_grad(set_to_none=True)
            scaler.scale(critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            scaler.step(opt_critic)

            if step % POLICY_DELAY == 0:
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device=="cuda")):
                    a_pred = actor(S_t)
                    # 轻正则：抑制大角与长时间刹车
                    reg = L2_STEER*(a_pred[:,0]**2).mean() + L2_BRAKE*(a_pred[:,2]**2).mean()
                    actor_loss = -critic.q1_only(S_t, a_pred).mean() + reg
                opt_actor.zero_grad(set_to_none=True)
                scaler.scale(actor_loss).backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                scaler.step(opt_actor)

            scaler.update()
            with torch.no_grad():
                for p,tp in zip(critic.parameters(), critic_t.parameters()): tp.data.mul_(1-TAU).add_(TAU*p.data)
                for p,tp in zip(actor.parameters(),  actor_t.parameters()): tp.data.mul_(1-TAU).add_(TAU*p.data)

    # 主循环
    step=0
    while step < TOTAL_STEPS:
        if step < START_RANDOM:
            a = np.stack([
                np.random.uniform(-STEER_MAX, STEER_MAX, size=NUM_ENVS),
                np.random.uniform(0.0, 1.0, size=NUM_ENVS),
                np.random.uniform(0.0, 1.0, size=NUM_ENVS),
            ], axis=1).astype(np.float32)
        else:
            import torch
            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16, enabled=(device=="cuda")):
                s_t = to_t(obs); a = actor(s_t).float().cpu().numpy()
            # —— 探索噪声（线性退火）——
            frac = min(1.0, max(0.0, step/float(NOISE_ANNEAL_STEPS)))
            ns = np.array([N_STEER, N_GAS, N_BRAKE])*(1.0-frac) + N_MIN*frac
            a = a + np.random.normal(0.0, ns[None,:], size=a.shape).astype(np.float32)
        actions = post_process_actions(a)

        # 执行 action repeat
        total_r = np.zeros(NUM_ENVS, dtype=np.float32)
        nobs = obs; done_flag = np.zeros(NUM_ENVS, dtype=bool)
        for _ in range(ACTION_REPEAT):
            nobs, r, term, trunc, info = envs.step(actions)
            total_r += r; done_flag = np.logical_or(done_flag, np.logical_or(term,trunc))

        for i in range(NUM_ENVS):
            rb.add(obs[i], actions[i], nobs[i], np.array([total_r[i]],np.float32), np.array([done_flag[i]],np.float32))
        obs = nobs

        step += NUM_ENVS
        log_if_needed(step)
        if step >= TRAIN_AFTER and step % TRAIN_FREQ == 0: train_td3(step)

        if done_flag.any():
            idxs = np.where(done_flag)[0].tolist()
            for j in idxs:
                ob,_ = envs.envs[j].reset(seed=SEED + random.randint(0,10_000)); obs[j]=ob

        # 命中窗口保存（避免跨整点漏存）
        if SAVE_EVERY>0 and step % SAVE_EVERY < NUM_ENVS:
            os.makedirs("./model", exist_ok=True)
            ckpt=f"./model/agent_step{step}.pt"
            import torch as _t
            _t.save({"actor": (actor.module.state_dict() if hasattr(actor,"module") else actor.state_dict()),
                     "critic": (critic.module.state_dict() if hasattr(critic,"module") else critic.state_dict()),
                     "step": step}, ckpt)
            print(f"SAVED {ckpt}")

    envs.close(); print("Training finished.")

if __name__ == "__main__":
    if os.name == "nt":
        os.environ.setdefault("PYTHONUTF8","1")
        os.environ.setdefault("PYTHONIOENCODING","utf-8")
        os.environ.setdefault("CUDA_MODULE_LOADING","LAZY")
    main()
