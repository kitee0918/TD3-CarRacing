import os
import time
import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
from gymnasium import ObservationWrapper, spaces

# -------- Simple console progress bar (optional) --------
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ------------------------------
# Robust wrappers (version-proof)
# ------------------------------
# FrameStack 位置可能不同：
# ------- FrameStack 通杀版（优先库，缺失则用本地实现） -------
try:
    from gymnasium.wrappers import FrameStack as _FrameStack
except Exception:
    try:
        from gymnasium.wrappers.frame_stack import FrameStack as _FrameStack
    except Exception:
        _FrameStack = None

if _FrameStack is None:
    from collections import deque as _deque
    class FrameStack(gym.Wrapper):
        def __init__(self, env, k: int):
            super().__init__(env)
            self.k = k
            self.frames = _deque(maxlen=k)
            assert isinstance(env.observation_space, gym.spaces.Box)
            low  = env.observation_space.low
            high = env.observation_space.high
            self.observation_space = gym.spaces.Box(
                low=np.repeat(low,  k, axis=-1),
                high=np.repeat(high, k, axis=-1),
                dtype=env.observation_space.dtype,
            )
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            self.frames.clear()
            for _ in range(self.k):
                self.frames.append(obs)
            return self._get_obs(), info
        def step(self, action):
            obs, reward, term, trunc, info = self.env.step(action)
            self.frames.append(obs)
            return self._get_obs(), reward, term, trunc, info
        def _get_obs(self):
            return np.concatenate(list(self.frames), axis=-1)
else:
    FrameStack = _FrameStack


def wrap_obs(env):
    try:
        from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
        env = GrayScaleObservation(env, keep_dim=True)     # (H,W,1)
        env = ResizeObservation(env, (84, 84))             # -> (84,84,1)
        return env
    except Exception:
        class GrayResize84(ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
            def observation(self, obs):
                gray = (0.299*obs[:,:,0] + 0.587*obs[:,:,1] + 0.114*obs[:,:,2]).astype(np.float32)
                try:
                    import cv2
                    gray84 = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
                except Exception:
                    ys = (np.linspace(0, gray.shape[0]-1, 84)).astype(np.int32)
                    xs = (np.linspace(0, gray.shape[1]-1, 84)).astype(np.int32)
                    gray84 = gray[np.ix_(ys, xs)]
                return np.clip(gray84, 0, 255).astype(np.uint8)[..., None]
        return GrayResize84(env)


# ==========================
# Utils
# ==========================
# NumPy 2.x compatibility helpers (avoid np.array(copy=False) pitfalls)
# Use np.asarray with dtype to allow copying when required by NumPy>=2.0
from typing import Any as _Any

def as_u8(x: _Any) -> np.ndarray:
    return np.asarray(x, dtype=np.uint8)

def as_f32(x: _Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dynamic_env_settings(base_envs=4):
    device = get_device()
    if device == "cuda":
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_gb = free_mem / (1024 ** 3)
            total_gb = total_mem / (1024 ** 3)
            estimated_env_mem = 0.9  # 更激进，便于提高并行度
            num_envs = max(2, int(free_gb / estimated_env_mem))
            batch_size = min(768, num_envs * 64)
            print(f"🔍 Detected GPU memory: {free_gb:.1f} GB free / {total_gb:.1f} GB total → NUM_ENVS={num_envs}, BATCH_SIZE={batch_size}")
            return num_envs, batch_size
        except Exception as e:
            print(f"⚠️ GPU memory check failed ({e}), fallback to default.")
            return base_envs, base_envs * 64
    else:
        print("⚠️ Running on CPU → NUM_ENVS=4, BATCH_SIZE=256")
        return base_envs, base_envs * 64


def make_env_fn(seed: int = 0):
    def _thunk():
        env = gym.make("CarRacing-v3", render_mode=None)
        # === 观测预处理：灰度 + 缩放 + 帧堆叠(4) ===
        env = wrap_obs(env)  # 版本兜底
        env = FrameStack(env, 4)  # (84,84,4)
        env.reset(seed=seed)
        return env
    return _thunk


def preprocess(obs: np.ndarray) -> torch.Tensor:
    """Convert (N, H, W, C) -> (N, C, H, W) float32 in [0,1]."""
    if obs.ndim == 3:
        obs = obs[0:1, ...]
    obs_t = torch.from_numpy(obs).float()
    if obs_t.max() > 1.5:
        obs_t = obs_t / 255.0
    obs_t = obs_t.permute(0, 3, 1, 2).contiguous()
    return obs_t


# ==========================
# Networks (支持可变输入通道)
# ==========================
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, in_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.Mish(),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.Mish(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.Mish(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.Mish(),
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.Mish(),
            nn.MaxPool2d(2),
        )
        flat_dim = 16 * (state_dim // 8) ** 2
        self.head = nn.Sequential(
            nn.Linear(flat_dim, 256), nn.LayerNorm(256), nn.Mish(),
            nn.Linear(256, action_dim), nn.LayerNorm(action_dim), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, sigma: float = 0.0):
        h = self.conv(x)
        h = torch.flatten(h, start_dim=1)
        a = self.head(h)
        if sigma > 0:
            a = a + sigma * torch.randn_like(a)
        a_out = a.clone()
        a_out[:, 0] = a[:, 0]                 # steer [-1,1]
        a_out[:, 1] = (a[:, 1] + 1) * 0.5     # gas [0,1]
        a_out[:, 2] = (a[:, 2] + 1) * 0.015   # brake ~[0,0.03]
        return a_out


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, in_ch: int):
        super().__init__()
        self.sconv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.Mish(),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.Mish(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.Mish(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.Mish(),
            nn.MaxPool2d(2),
        )
        flat_dim = 16 * (state_dim // 8) ** 2
        self.shead = nn.Sequential(nn.Linear(flat_dim, 256), nn.LayerNorm(256), nn.Mish())
        self.ahead = nn.Sequential(nn.Linear(action_dim, 256), nn.LayerNorm(256), nn.Mish())
        self.q = nn.Sequential(
            nn.Linear(512, 64), nn.LayerNorm(64), nn.Mish(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.Mish(),
            nn.Linear(64, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        sh = self.sconv(s)
        sh = torch.flatten(sh, start_dim=1)
        sh = self.shead(sh)
        ah = self.ahead(a)
        h = torch.cat([sh, ah], dim=1)
        return self.q(h)


# ==========================
# Replay Buffer
# ==========================
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def add(self, s_u8, a, ns_u8, r, d):
        # store obs as uint8 NHWC to reduce RAM; convert on sample
        self.buf.append((
            s_u8.astype(np.uint8, copy=False),
            a.astype(np.float32, copy=False),
            ns_u8.astype(np.uint8, copy=False),
            np.asarray(r, dtype=np.float32),
            np.asarray(d, dtype=np.float32),
        ))

    def extend_batch(self, S, A, NS, R, D):
        for i in range(S.shape[0]):
            self.add(S[i], A[i], NS[i], R[i], D[i])

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s_u8, a, ns_u8, r, d = map(np.stack, zip(*batch))
        # to torch tensors (CPU); preprocess: NHWC uint8 -> NCHW float32 in [0,1]
        s = torch.from_numpy(s_u8).to(torch.float32).permute(0, 3, 1, 2).contiguous().div_(255.0)
        ns = torch.from_numpy(ns_u8).to(torch.float32).permute(0, 3, 1, 2).contiguous().div_(255.0)
        a = torch.from_numpy(a).to(torch.float32)
        r = torch.from_numpy(r).to(torch.float32)
        d = torch.from_numpy(d).to(torch.float32)
        return s, a, ns, r, d

    def __len__(self):
        return len(self.buf)
            # torch.tensor(a, dtype=torch.float32, pin_memory=True),
            # torch.tensor(ns, dtype=torch.float32, pin_memory=True),
            # torch.tensor(r, dtype=torch.float32, pin_memory=True),
            # torch.tensor(d, dtype=torch.float32, pin_memory=True),
        # )

    def __len__(self):
        return len(self.buf)


# ==========================
# TD3 Agent
# ==========================
class TD3:
    def __init__(self, state_dim: int, action_dim: int, in_ch: int, lr=1e-4, device="cpu"):
        self.device = device
        self.actor = Actor(state_dim, action_dim, in_ch).to(device)
        self.actor_t = Actor(state_dim, action_dim, in_ch).to(device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.c1 = Critic(state_dim, action_dim, in_ch).to(device)
        self.c2 = Critic(state_dim, action_dim, in_ch).to(device)
        self.c1_t = Critic(state_dim, action_dim, in_ch).to(device)
        self.c2_t = Critic(state_dim, action_dim, in_ch).to(device)
        self.c1_t.load_state_dict(self.c1.state_dict())
        self.c2_t.load_state_dict(self.c2.state_dict())

        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c1 = torch.optim.Adam(self.c1.parameters(), lr=lr)
        self.opt_c2 = torch.optim.Adam(self.c2.parameters(), lr=lr)
        self.mse = nn.MSELoss()
        # AMP scaler (CUDA only)
        try:
            self.scaler = torch.amp.GradScaler('cuda') if str(device).startswith('cuda') else None
        except Exception:
            self.scaler = None  # older torch fallback

    @torch.no_grad()
    def act(self, s: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        a = self.actor(s.to(self.device), sigma=sigma)
        return a.clamp(torch.tensor([-1.0, 0.0, 0.0], device=self.device),
                       torch.tensor([1.0, 1.0, 1.0], device=self.device))

    def train_step(self, rb: ReplayBuffer, batch_size: int, gamma: float, tau: float, policy_delay: int, step_i: int):
        if len(rb) < batch_size:
            return None
        s, a, ns, r, d = rb.sample(batch_size)
        s = s.to(self.device, non_blocking=True)
        a = a.to(self.device, non_blocking=True)
        ns = ns.to(self.device, non_blocking=True)
        r = r.to(self.device, non_blocking=True).unsqueeze(1)
        d = d.to(self.device, non_blocking=True).unsqueeze(1)

        with torch.no_grad():
            noise = (0.2 * torch.randn_like(a)).clamp(-0.5, 0.5)  # TD3 target policy smoothing
            na = self.actor_t(ns) + noise
            na = na.clamp(
                torch.tensor([-1.0, 0.0, 0.0], device=self.device),
                torch.tensor([ 1.0, 1.0, 1.0], device=self.device)
            )
            tq1 = self.c1_t(ns, na)
            tq2 = self.c2_t(ns, na)
            y = r + gamma * torch.min(tq1, tq2) * (1 - d)

        use_amp = (self.scaler is not None)
        if use_amp:
            self.opt_c1.zero_grad(set_to_none=True)
            self.opt_c2.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                q1 = self.c1(s, a)
                q2 = self.c2(s, a)
                l1 = self.mse(q1, y)
                l2 = self.mse(q2, y)
            self.scaler.scale(l1).backward()
            self.scaler.step(self.opt_c1)
            self.scaler.scale(l2).backward()
            self.scaler.step(self.opt_c2)
            self.scaler.update()
        else:
            q1 = self.c1(s, a)
            q2 = self.c2(s, a)
            l1 = self.mse(q1, y)
            l2 = self.mse(q2, y)
            self.opt_c1.zero_grad(); l1.backward(); self.opt_c1.step()
            self.opt_c2.zero_grad(); l2.backward(); self.opt_c2.step()

        info = {"critic_loss1": l1.item(), "critic_loss2": l2.item(), "target": y.mean().item()}

        if step_i % policy_delay == 0:
            use_amp = (self.scaler is not None)
            if use_amp:
                self.opt_a.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    pa = self.actor(s)
                    aloss = -self.c1(s, pa).mean()
                self.scaler.scale(aloss).backward()
                self.scaler.step(self.opt_a)
                self.scaler.update()
            else:
                pa = self.actor(s)
                aloss = -self.c1(s, pa).mean()
                self.opt_a.zero_grad(); aloss.backward(); self.opt_a.step()
            with torch.no_grad():
                for p, tp in zip(self.actor.parameters(), self.actor_t.parameters()):
                    tp.data.mul_(1 - tau).add_(tau * p.data)
                for p, tp in zip(self.c1.parameters(), self.c1_t.parameters()):
                    tp.data.mul_(1 - tau).add_(tau * p.data)
                for p, tp in zip(self.c2.parameters(), self.c2_t.parameters()):
                    tp.data.mul_(1 - tau).add_(tau * p.data)
            info.update({"actor_loss": aloss.item()})
        return info


# ==========================
# Checkpoint helpers（保存 Actor 模型对象）
# ==========================
def save_ckpt(agent: TD3, step: int, out_dir: str = "./model"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"agent_step{step}.pt")
    torch.save(agent.actor, path)
    print(f"💾 Saved: {path}")


def load_latest_ckpt(agent: TD3, out_dir: str = "./model") -> int:
    if not os.path.exists(out_dir):
        return 0
    files = [f for f in os.listdir(out_dir) if f.startswith("agent_step") and f.endswith(".pt")]
    if not files:
        return 0
    latest = max(files, key=lambda x: int(x.replace("agent_step", "").replace(".pt", "")))
    path = os.path.join(out_dir, latest)

    # --- 读取 checkpoint（兼容 torch 2.6 的 weights_only 以及 allowlist） ---
    global torch
    try:
        obj = torch.load(path, weights_only=False, map_location=agent.device)
    except TypeError:
        import __main__
        import torch.serialization as torch_serialization
        torch_serialization.add_safe_globals([__main__.Actor])
        obj = torch.load(path, map_location=agent.device)

    # 取出 state_dict
    src_sd = obj.state_dict() if isinstance(obj, nn.Module) else obj
    dst_sd = agent.actor.state_dict()

    new_sd = {}
    matched, adapted, skipped = 0, 0, 0

    # 1) 先拷贝所有 shape 完全一致的参数
    for k, v_dst in dst_sd.items():
        v_src = src_sd.get(k, None)
        if v_src is not None and tuple(v_src.shape) == tuple(v_dst.shape):
            new_sd[k] = v_src
            matched += 1

    # 2) 特判：把 3 通道 conv 权重适配到 4 通道
    k0 = "conv.0.weight"  # [16, C, 3, 3]
    if k0 in src_sd and k0 in dst_sd:
        w_old = src_sd[k0]
        w_dst = dst_sd[k0]
        if w_old.ndim == 4 and w_dst.ndim == 4 and w_old.shape[1] == 3 and w_dst.shape[1] == 4:
            w_new = torch.zeros_like(w_dst)
            w_new[:, :3, :, :] = w_old
            w_new[:, 3:4, :, :] = w_old.mean(dim=1, keepdim=True)  # 第4通道=RGB均值
            new_sd[k0] = w_new
            adapted += 1

    # 3) 丢弃线性层 head.0.weight（in_features 不同：2304 vs 1600）
    bad_keys = ["head.0.weight"]
    for bk in bad_keys:
        if bk in new_sd and tuple(new_sd[bk].shape) != tuple(dst_sd[bk].shape):
            del new_sd[bk]
            skipped += 1

    # 4) 加载（strict=False 允许缺失键）
    incompatible = agent.actor.load_state_dict(new_sd, strict=False)
    print(
        f"🔁 partial load: matched={matched}, adapted_conv={adapted}, "
        f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}, skipped_linear={skipped}"
    )

    # 同步到 target
    agent.actor_t.load_state_dict(agent.actor.state_dict())
    step = int(latest.replace("agent_step", "").replace(".pt", ""))
    print(f"✅ Loaded (partial) from {path} (step={step})")
    return step

# ==========================
# Auto-eval video (short clip)
# ==========================
AUTO_EVAL_EVERY = int(os.environ.get("AUTO_EVAL_EVERY", 50_000))  # 0=off
EVAL_SECONDS = int(os.environ.get("EVAL_SECONDS", 30))
EVAL_MAX_FRAMES = int(os.environ.get("EVAL_MAX_FRAMES", 50 * EVAL_SECONDS))
EVAL_EPISODES = int(os.environ.get("EVAL_EPISODES", 1))
EVAL_DIR = os.environ.get("EVAL_DIR", "./eval_videos")


def auto_eval_video(agent: TD3, step: int, device: str, out_dir: str = EVAL_DIR,
                    episodes: int = EVAL_EPISODES, max_frames: int = EVAL_MAX_FRAMES):
    """Record a short evaluation video compatible with all gym versions."""
    render_mode = "rgb_array"
    env = gym.make("CarRacing-v3", render_mode=render_mode)
    # RecordVideo 兼容导入
    try:
        from gymnasium.wrappers import RecordVideo
    except Exception:
        from gymnasium.wrappers.record_video import RecordVideo
    # 灰度/缩放 + 帧堆叠（与训练一致）
    env = wrap_obs(env)
    env = FrameStack(env, 4)

    os.makedirs(out_dir, exist_ok=True)
    clip_dir = os.path.join(out_dir, f"step{step}")
    os.makedirs(clip_dir, exist_ok=True)
    env = RecordVideo(env, video_folder=clip_dir, episode_trigger=lambda e: True, name_prefix=f"eval_step{step}")

    actor = agent.actor
    actor.eval()
    rets = []
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        frames = 0
        ep_ret = 0.0
        while not done and frames < max_frames:
            s = preprocess(obs)
            with torch.no_grad():
                a = actor(s.to(device), sigma=0.0).cpu().numpy()[0].astype(np.float64)
            obs, r, term, trunc, info = env.step(a)
            done = bool(term) or bool(trunc)
            ep_ret += float(r)
            frames += 1
        rets.append(ep_ret)
    env.close()
    mean_ret = float(np.mean(rets))
    print(f"🎥 Auto-eval saved to {clip_dir} | episodes={episodes} | mean_return={mean_ret:.2f} | frames≤{max_frames}")
    return clip_dir, mean_ret


# ==========================
# Training (Parallel)
# ==========================
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = get_device()
    num_envs, batch_size = get_dynamic_env_settings(base_envs=4)

    # Optional overrides via env vars
    FORCE_NUM_ENVS = int(os.environ.get("NUM_ENVS", "0") or "0")
    if FORCE_NUM_ENVS > 0:
        num_envs = FORCE_NUM_ENVS
        batch_size = min(768, num_envs * 64)

    # Optional explicit batch-size override (decouple from NUM_ENVS)
    FORCE_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "0") or os.environ.get("BATCH_SIZE", "0") or "0")
    if FORCE_BATCH_SIZE > 0:
        batch_size = FORCE_BATCH_SIZE

    TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", 1_000_000))
    START_RANDOM_STEPS = int(os.environ.get("START_RANDOM_STEPS", 10_000))
    TRAIN_AFTER = int(os.environ.get("TRAIN_AFTER", 20_000))
    TRAIN_FREQ = int(os.environ.get("TRAIN_FREQ", 2))
    POLICY_DELAY = int(os.environ.get("POLICY_DELAY", 3))

    REPLAY_CAP = int(os.environ.get("REPLAY_CAP", 100_000))
    GAMMA = 0.985
    TAU = 0.005
    SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 50_000))

    print("[boot] creating envs...", flush=True)
    try:
        envs = AsyncVectorEnv([make_env_fn(seed=42 + i) for i in range(num_envs)],
                              autoreset_mode=AutoresetMode.NEXT_STEP)
    except TypeError:
        envs = AsyncVectorEnv([make_env_fn(seed=42 + i) for i in range(num_envs)])

    obs, infos = envs.reset()
    print("[boot] reset ok", flush=True)
    H, W, C = obs.shape[1:]
    action_dim = envs.single_action_space.shape[0]

    agent = TD3(state_dim=H, action_dim=action_dim, in_ch=C, lr=1e-4, device=device)
    rb = ReplayBuffer(REPLAY_CAP)
    writer = SummaryWriter("./data")
    writer.add_text("boot", f"envs={num_envs}, batch_size={batch_size}, device={device}", 0)
    writer.flush()

    step = load_latest_ckpt(agent)
    # --- threshold-based save/eval steps to handle vectorized stepping (step+=num_envs) ---
    next_save_step = ((step // SAVE_EVERY) + 1) * SAVE_EVERY if SAVE_EVERY > 0 else float('inf')
    next_eval_step = ((step // AUTO_EVAL_EVERY) + 1) * AUTO_EVAL_EVERY if AUTO_EVAL_EVERY > 0 else float('inf')
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    episode_count = 0

    s = preprocess(obs).to(device)
    start_time = time.time()

    # 动作平滑与约束
    prev_actions = np.zeros((num_envs, 3), dtype=np.float64)
    SMOOTH = 0.2

    # Progress bar
    pbar = None
    if 'tqdm' in globals() and tqdm is not None:
        pbar = tqdm(total=TOTAL_STEPS, initial=step, desc="train", ncols=100)

    while step < TOTAL_STEPS:
        if step < START_RANDOM_STEPS:
            a = np.stack([envs.single_action_space.sample() for _ in range(num_envs)], axis=0)
        else:
            with torch.no_grad():
                act = agent.act(s, sigma=max(0.1 * (1 - step / TOTAL_STEPS), 0.01))
            a = act.detach().cpu().numpy()
        a = a.astype(np.float64)

        # 动作约束与平滑
        a[:, 1] *= (1.0 - 0.6 * np.clip(np.abs(a[:, 0]), 0, 1))  # 转弯减油
        a[:, 2] = np.clip(a[:, 2] * 6.0, 0.0, 1.0)               # 刹车放大
        a[:, 0] = (1 - SMOOTH) * prev_actions[:, 0] + SMOOTH * a[:, 0]
        dsteer = np.abs(a[:, 0] - prev_actions[:, 0])
        prev_actions = a.copy()

        next_obs, rewards, terms, truncs, infos = envs.step(a)
        dones = np.logical_or(terms, truncs)
        rewards = rewards + (-0.002 * dsteer)  # 轻量稳定器

        # store raw uint8 NHWC to save RAM; convert to float on sampling
        rb.extend_batch(
            obs.astype(np.uint8, copy=False),
            a.astype(np.float32, copy=False),
            next_obs.astype(np.uint8, copy=False),
            rewards.astype(np.float32, copy=False),
            dones.astype(np.float32, copy=False),
        )

        episode_rewards += rewards
        episode_lengths += 1

        if np.any(dones):
            for i, d in enumerate(dones):
                if d:
                    writer.add_scalar("episode/return", episode_rewards[i], episode_count)
                    writer.add_scalar("episode/length", episode_lengths[i], episode_count)
                    episode_count += 1
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0

            done_ids = np.where(dones)[0]
            if len(done_ids) > 0:
                # ✅ Autoreset handled by AsyncVectorEnv (NEXT_STEP). Do NOT call envs.reset or envs.call('reset').
                # Reset per-env trackers only to avoid action carry-over into new episodes.
                prev_actions[done_ids] = 0.0

        # Learn
        if step >= TRAIN_AFTER and step % TRAIN_FREQ == 0:
            info = agent.train_step(rb, batch_size, GAMMA, TAU, POLICY_DELAY, step)
            if info is not None and step % 500 == 0:
                writer.add_scalar("loss/critic1", info["critic_loss1"], step)
                writer.add_scalar("loss/critic2", info["critic_loss2"], step)
                if "actor_loss" in info:
                    writer.add_scalar("loss/actor", info["actor_loss"], step)
                writer.add_scalar("target/mean_q", info["target"], step)

        # Logs
        LOG_EVERY = int(os.environ.get("LOG_EVERY", 1000))
        if step % LOG_EVERY == 0 and step > 0:
            elapsed = time.time() - start_time
            fps = (LOG_EVERY * num_envs) / max(elapsed, 1e-6)
            print(f"step={step} | fps~{fps:.1f} | rb={len(rb)} | sigma={max(0.1 * (1 - step / TOTAL_STEPS), 0.01):.3f}")
            writer.add_scalar("sys/fps", fps, step)
            writer.flush()
            start_time = time.time()

        # Save & Auto-Eval (threshold-based, not exact modulo)
        if SAVE_EVERY > 0 and step >= next_save_step:
            save_ckpt(agent, step)
            next_save_step += SAVE_EVERY

        if AUTO_EVAL_EVERY > 0 and step >= next_eval_step:
            try:
                clip_dir, mean_ret = auto_eval_video(agent, step, device)
                writer.add_scalar("eval/return", mean_ret, step)
            except Exception as e:
                print(f"⚠️ Auto-eval failed: {e}")
            finally:
                next_eval_step += AUTO_EVAL_EVERY

        s = preprocess(next_obs).to(device)
        obs = next_obs
        step += num_envs

        # update progress bar
        if pbar is not None:
            pbar.update(num_envs)
            if (step // max(num_envs, 1)) % 10 == 0:
                pbar.set_postfix(rb=len(rb), sigma=float(max(0.1 * (1 - step / TOTAL_STEPS), 0.01)))

    if pbar is not None:
        pbar.close()

    save_ckpt(agent, step)
    envs.close()
    writer.close()
