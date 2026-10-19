# /mnt/data/td3_car_racing_train_fast.py
# 目标：在CPU受限时提升样本吞吐与GPU利用。建议起步：
# ACTION_REPEAT=3 NUM_ENVS=16 UPS=2 TRAIN_BATCH_SIZE=768
import os, time, random
from collections import deque
from typing import Any, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
from gymnasium import ObservationWrapper, spaces

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ============== Wrappers ==============
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
            super().__init__(env); self.k = k; self.frames = _deque(maxlen=k)
            assert isinstance(env.observation_space, gym.spaces.Box)
            low, high = env.observation_space.low, env.observation_space.high
            self.observation_space = gym.spaces.Box(
                low=np.repeat(low,k,axis=-1), high=np.repeat(high,k,axis=-1),
                dtype=env.observation_space.dtype)
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs); self.frames.clear()
            for _ in range(self.k): self.frames.append(obs)
            return self._get_obs(), info
        def step(self, action):
            obs, r, term, trunc, info = self.env.step(action); self.frames.append(obs)
            return self._get_obs(), r, term, trunc, info
        def _get_obs(self): return np.concatenate(list(self.frames), axis=-1)
else:
    FrameStack = _FrameStack

class ActionRepeat(gym.Wrapper):
    """Repeat same action k times; accumulate reward; early break on done.
       为什么：在CPU受限时等效减少环境步数，显著提速。"""
    def __init__(self, env, k: int = 3):
        super().__init__(env)
        assert k >= 1
        self.k = int(k)
    def step(self, action):
        total_r = 0.0
        term = False; trunc = False; info = {}
        obs = None
        for _ in range(self.k):
            obs, r, t, u, info = self.env.step(action)
            total_r += float(r); term = term or bool(t); trunc = trunc or bool(u)
            if term or trunc:
                break
        return obs, total_r, term, trunc, info

def wrap_obs(env):
    try:
        from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, (84, 84))
        return env
    except Exception:
        class GrayResize84(ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                self.observation_space = spaces.Box(low=0, high=255, shape=(84,84,1), dtype=np.uint8)
            def observation(self, obs):
                g = (0.299*obs[:,:,0] + 0.587*obs[:,:,1] + 0.114*obs[:,:,2]).astype(np.float32)
                try:
                    import cv2; g84 = cv2.resize(g, (84,84), interpolation=cv2.INTER_AREA)
                except Exception:
                    ys = (np.linspace(0, g.shape[0]-1, 84)).astype(np.int32)
                    xs = (np.linspace(0, g.shape[1]-1, 84)).astype(np.int32)
                    g84 = g[np.ix_(ys, xs)]
                return np.clip(g84, 0, 255).astype(np.uint8)[...,None]
        return GrayResize84(env)

# ============== Utils ==============
def as_u8(x: Any) -> np.ndarray: return np.asarray(x, dtype=np.uint8)
def as_f32(x: Any) -> np.ndarray: return np.asarray(x, dtype=np.float32)

def get_device(): return "cuda" if torch.cuda.is_available() else "cpu"

def preprocess(obs: np.ndarray) -> torch.Tensor:
    if obs.ndim == 3: obs = obs[None, ...]
    t = torch.from_numpy(obs).float()
    if t.max() > 1.5: t = t / 255.0
    return t.permute(0,3,1,2).contiguous()

def get_dynamic_env_settings(base_envs=4):
    device = get_device()
    if device == "cuda":
        try:
            free, total = torch.cuda.mem_get_info()
            free_gb, total_gb = free/(1024**3), total/(1024**3)
            num_envs = max(2, int(free_gb / 0.9))
            batch = min(768, num_envs * 64)
            print(f"[GPU] Detected GPU memory: {free_gb:.1f} / {total_gb:.1f} GB -> NUM_ENVS={num_envs}, BATCH_SIZE={batch}")
            return num_envs, batch
        except Exception as e:
            print(f"[WARN] GPU mem check failed: {e}")
    return base_envs, base_envs*64

def make_env_fn(seed: int, action_repeat: int):
    def _thunk():
        env = gym.make("CarRacing-v3", render_mode=None)
        env = wrap_obs(env)
        env = FrameStack(env, 4)
        if action_repeat > 1:
            env = ActionRepeat(env, k=action_repeat)
        env.reset(seed=seed)
        return env
    return _thunk

# ============== Nets ==============
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
        flat = 16 * (state_dim//8)**2
        self.head = nn.Sequential(
            nn.Linear(flat, 256), nn.LayerNorm(256), nn.Mish(),
            nn.Linear(256, 3), nn.LayerNorm(3), nn.Tanh(),
        )
    def forward(self, x: torch.Tensor, sigma: float = 0.0):
        h = self.conv(x); h = torch.flatten(h,1); a = self.head(h)
        if sigma > 0: a = a + sigma*torch.randn_like(a)
        out = a.clone()
        out[:,0] = a[:,0]
        out[:,1] = (a[:,1]+1)*0.5
        out[:,2] = (a[:,2]+1)*0.015
        return out

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
        flat = 16 * (state_dim//8)**2
        self.shead = nn.Sequential(nn.Linear(flat,256), nn.LayerNorm(256), nn.Mish())
        self.ahead = nn.Sequential(nn.Linear(action_dim,256), nn.LayerNorm(256), nn.Mish())
        self.q = nn.Sequential(
            nn.Linear(512,64), nn.LayerNorm(64), nn.Mish(),
            nn.Linear(64,64), nn.LayerNorm(64), nn.Mish(),
            nn.Linear(64,1),
        )
    def forward(self, s, a):
        sh = self.sconv(s); sh = torch.flatten(sh,1); sh = self.shead(sh)
        ah = self.ahead(a);  h = torch.cat([sh,ah],1)
        return self.q(h)

# ============== Replay (RAM, uint8 NHWC) ==============
class ReplayBuffer:
    def __init__(self, cap: int): self.buf = deque(maxlen=cap)
    def add(self, s_u8, a, ns_u8, r, d):
        self.buf.append((as_u8(s_u8), as_f32(a), as_u8(ns_u8), as_f32(r), as_f32(d)))
    def extend_batch(self, S,A,NS,R,D):
        for i in range(S.shape[0]): self.add(S[i],A[i],NS[i],R[i],D[i])
    def sample(self, bs: int):
        b = random.sample(self.buf, bs)
        s_u8, a, ns_u8, r, d = map(np.stack, zip(*b))
        s  = torch.from_numpy(s_u8).float().permute(0,3,1,2).contiguous().div_(255.0)
        ns = torch.from_numpy(ns_u8).float().permute(0,3,1,2).contiguous().div_(255.0)
        a  = torch.from_numpy(a).float()
        r  = torch.from_numpy(r).float()
        d  = torch.from_numpy(d).float()
        return s,a,ns,r,d
    def __len__(self): return len(self.buf)

# ============== TD3 (AMP + optional compile) ==============
class TD3:
    def __init__(self, state_dim:int, action_dim:int, in_ch:int, lr=1e-4, device="cpu", compile_flag=False):
        self.device = device
        self.actor = Actor(state_dim, action_dim, in_ch).to(device)
        self.actor_t = Actor(state_dim, action_dim, in_ch).to(device); self.actor_t.load_state_dict(self.actor.state_dict())
        self.c1 = Critic(state_dim, action_dim, in_ch).to(device)
        self.c2 = Critic(state_dim, action_dim, in_ch).to(device)
        self.c1_t = Critic(state_dim, action_dim, in_ch).to(device); self.c1_t.load_state_dict(self.c1.state_dict())
        self.c2_t = Critic(state_dim, action_dim, in_ch).to(device); self.c2_t.load_state_dict(self.c2.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c1 = torch.optim.Adam(self.c1.parameters(), lr=lr)
        self.opt_c2 = torch.optim.Adam(self.c2.parameters(), lr=lr)
        self.mse = nn.MSELoss()
        try:
            self.scaler = torch.amp.GradScaler('cuda') if str(device).startswith('cuda') else None
        except Exception:
            self.scaler = None
        if compile_flag:
            try:
                self.actor   = torch.compile(self.actor, mode="max-autotune")
                self.c1      = torch.compile(self.c1,    mode="max-autotune")
                self.c2      = torch.compile(self.c2,    mode="max-autotune")
                self.actor_t = torch.compile(self.actor_t,mode="max-autotune")
                self.c1_t    = torch.compile(self.c1_t,   mode="max-autotune")
                self.c2_t    = torch.compile(self.c2_t,   mode="max-autotune")
                print("⚡ torch.compile enabled (max-autotune)")
            except Exception as e:
                print(f"[WARN] torch.compile not available: {e}")

    @torch.no_grad()
    def act(self, s: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        a = self.actor(s.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last), sigma=sigma)
        return a.clamp(torch.tensor([-1.0,0.0,0.0], device=self.device),
                       torch.tensor([ 1.0,1.0,1.0], device=self.device))

    def train_step(self, rb: ReplayBuffer, batch_size: int, gamma: float, tau: float, policy_delay: int, step_i: int):
        if len(rb) < batch_size: return None
        s,a,ns,r,d = rb.sample(batch_size)
        s  = s.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        a  = a.to(self.device, non_blocking=True)
        ns = ns.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        r  = r.to(self.device, non_blocking=True).unsqueeze(1)
        d  = d.to(self.device, non_blocking=True).unsqueeze(1)

        with torch.no_grad():
            noise = (0.2 * torch.randn_like(a)).clamp(-0.5, 0.5)
            na = self.actor_t(ns) + noise
            na = na.clamp(torch.tensor([-1.0,0.0,0.0], device=self.device),
                          torch.tensor([ 1.0,1.0,1.0], device=self.device))
            tq1 = self.c1_t(ns, na); tq2 = self.c2_t(ns, na)
            y = r + gamma * torch.min(tq1, tq2) * (1 - d)

        use_amp = (self.scaler is not None)
        if use_amp:
            self.opt_c1.zero_grad(set_to_none=True); self.opt_c2.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                q1 = self.c1(s, a); q2 = self.c2(s, a)
                l1 = self.mse(q1, y); l2 = self.mse(q2, y)
            self.scaler.scale(l1).backward(); self.scaler.step(self.opt_c1)
            self.scaler.scale(l2).backward(); self.scaler.step(self.opt_c2)
            self.scaler.update()
        else:
            q1 = self.c1(s, a); q2 = self.c2(s, a)
            l1 = self.mse(q1, y); l2 = self.mse(q2, y)
            self.opt_c1.zero_grad(); l1.backward(); self.opt_c1.step()
            self.opt_c2.zero_grad(); l2.backward(); self.opt_c2.step()

        info = {"critic_loss1": float(l1.detach()), "critic_loss2": float(l2.detach()), "target": float(y.mean())}

        if step_i % policy_delay == 0:
            if use_amp:
                self.opt_a.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    pa = self.actor(s)
                    aloss = -self.c1(s, pa).mean()
                self.scaler.scale(aloss).backward(); self.scaler.step(self.opt_a); self.scaler.update()
            else:
                pa = self.actor(s); aloss = -self.c1(s, pa).mean()
                self.opt_a.zero_grad(); aloss.backward(); self.opt_a.step()
            with torch.no_grad():
                for p,tp in zip(self.actor.parameters(), self.actor_t.parameters()): tp.data.mul_(1 - tau).add_(tau * p.data)
                for p,tp in zip(self.c1.parameters(), self.c1_t.parameters()):    tp.data.mul_(1 - tau).add_(tau * p.data)
                for p,tp in zip(self.c2.parameters(), self.c2_t.parameters()):    tp.data.mul_(1 - tau).add_(tau * p.data)
            info["actor_loss"] = float(aloss.detach())
        return info

# ============== Checkpoint ==============
def save_ckpt(agent: TD3, step: int, out_dir="./model"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"agent_step{step}.pt"); torch.save(agent.actor, path)
    print(f"[SAVE] Saved: {path}")

def load_latest_ckpt(agent: TD3, out_dir="./model") -> int:
    if not os.path.exists(out_dir): return 0
    files = [f for f in os.listdir(out_dir) if f.startswith("agent_step") and f.endswith(".pt")]
    if not files: return 0
    latest = max(files, key=lambda x: int(x.replace("agent_step","").replace(".pt","")))
    path = os.path.join(out_dir, latest)
    try:
        obj = torch.load(path, weights_only=False, map_location=agent.device)
    except TypeError:
        import __main__, torch.serialization as ts
        try: ts.add_safe_globals([__main__.Actor])
        except Exception: pass
        obj = torch.load(path, map_location=agent.device)
    src_sd = obj.state_dict() if isinstance(obj, nn.Module) else obj
    agent.actor.load_state_dict(src_sd, strict=False)
    agent.actor_t.load_state_dict(agent.actor.state_dict())
    step = int(latest.replace("agent_step","").replace(".pt",""))
    print(f"[OK] Loaded {path} (step={step})"); return step

# ============== Auto-eval ==============
AUTO_EVAL_EVERY = int(os.environ.get("AUTO_EVAL_EVERY", 50_000))
EVAL_SECONDS = int(os.environ.get("EVAL_SECONDS", 30))
EVAL_MAX_FRAMES = int(os.environ.get("EVAL_MAX_FRAMES", 50 * EVAL_SECONDS))
EVAL_EPISODES = int(os.environ.get("EVAL_EPISODES", 1))
EVAL_DIR = os.environ.get("EVAL_DIR", "./eval_videos")
def auto_eval_video(agent: TD3, step: int, device: str, out_dir: str = EVAL_DIR,
                    episodes: int = EVAL_EPISODES, max_frames: int = EVAL_MAX_FRAMES):
    rm = "rgb_array"; env = gym.make("CarRacing-v3", render_mode=rm)
    try: from gymnasium.wrappers import RecordVideo
    except Exception: from gymnasium.wrappers.record_video import RecordVideo
    env = wrap_obs(env); env = FrameStack(env, 4)
    os.makedirs(out_dir, exist_ok=True); clip_dir = os.path.join(out_dir, f"step{step}"); os.makedirs(clip_dir, exist_ok=True)
    env = RecordVideo(env, video_folder=clip_dir, episode_trigger=lambda e: True, name_prefix=f"eval_step{step}")
    actor = agent.actor.eval(); rets=[]
    for ep in range(episodes):
        obs,_ = env.reset(); done=False; frames=0; ep_ret=0.0
        while not done and frames < max_frames:
            s = preprocess(obs)
            with torch.no_grad():
                a = actor(s.to(device), sigma=0.0).cpu().numpy()[0].astype(np.float64)
            obs, r, term, trunc, _ = env.step(a)
            done = bool(term) or bool(trunc); ep_ret += float(r); frames += 1
        rets.append(ep_ret)
    env.close(); print(f"[VIDEO] saved -> {clip_dir} | mean_return={np.mean(rets):.2f}")
    return clip_dir, float(np.mean(rets))

# ============== Training ==============
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

    device = get_device()
    num_envs, batch_size = get_dynamic_env_settings(4)

    FORCE_NUM_ENVS = int(os.environ.get("NUM_ENVS","0") or "0")
    if FORCE_NUM_ENVS>0:
        num_envs = FORCE_NUM_ENVS
        batch_size = min(768, num_envs*64)
    FORCE_BATCH = int(os.environ.get("TRAIN_BATCH_SIZE","0") or os.environ.get("BATCH_SIZE","0") or "0")
    if FORCE_BATCH>0: batch_size = FORCE_BATCH

    # 激进参数
    ACTION_REPEAT = int(os.environ.get("ACTION_REPEAT", 3))        # 建议 2~4
    UPS = int(os.environ.get("UPS", 2))                             # 每个学习时刻的反向次数
    TORCH_COMPILE = int(os.environ.get("TORCH_COMPILE", 0)) == 1

    TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", 1_000_000))
    START_RANDOM_STEPS = int(os.environ.get("START_RANDOM_STEPS", 1_000))
    TRAIN_AFTER = int(os.environ.get("TRAIN_AFTER", 3_000))
    TRAIN_FREQ = int(os.environ.get("TRAIN_FREQ", 1))
    POLICY_DELAY = int(os.environ.get("POLICY_DELAY", 2))
    REPLAY_CAP = int(os.environ.get("REPLAY_CAP", 60_000))
    SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 50_000))
    LOG_EVERY = int(os.environ.get("LOG_EVERY", 1000))
    GAMMA, TAU = 0.985, 0.005

    print(f"[boot] creating envs... (num_envs={num_envs}, action_repeat={ACTION_REPEAT})")
    try:
        envs = AsyncVectorEnv([make_env_fn(42+i, ACTION_REPEAT) for i in range(num_envs)],
                              autoreset_mode=AutoresetMode.NEXT_STEP)
    except TypeError:
        envs = AsyncVectorEnv([make_env_fn(42+i, ACTION_REPEAT) for i in range(num_envs)])
    obs, _ = envs.reset(); print("[boot] reset ok")
    H,W,C = obs.shape[1:]; action_dim = envs.single_action_space.shape[0]

    agent = TD3(H, action_dim, C, lr=1e-4, device=device, compile_flag=TORCH_COMPILE)
    rb = ReplayBuffer(REPLAY_CAP)
    writer = SummaryWriter("./data")
    writer.add_text("boot", f"envs={num_envs} batch={batch_size} action_repeat={ACTION_REPEAT} UPS={UPS} device={device}", 0); writer.flush()

    step = load_latest_ckpt(agent)
    next_save_step = ((step // SAVE_EVERY)+1)*SAVE_EVERY if SAVE_EVERY>0 else float('inf')
    next_eval_step = ((step // AUTO_EVAL_EVERY)+1)*AUTO_EVAL_EVERY if AUTO_EVAL_EVERY>0 else float('inf')

    s = preprocess(obs).to(device).contiguous(memory_format=torch.channels_last)
    start_t = time.time()
    prev_actions = np.zeros((num_envs,3), dtype=np.float64)
    SMOOTH = float(os.environ.get("SMOOTH", 0.2))

    pbar = tqdm(total=TOTAL_STEPS, initial=step, desc="train", ncols=100) if tqdm else None
    while step < TOTAL_STEPS:
        if step < START_RANDOM_STEPS:
            a = np.stack([envs.single_action_space.sample() for _ in range(num_envs)],0).astype(np.float64)
        else:
            with torch.no_grad():
                pi = agent.act(s, sigma=max(0.1*(1 - step/TOTAL_STEPS), 0.01))
            a = pi.cpu().numpy().astype(np.float64)

        a[:,1] *= (1.0 - 0.6*np.clip(np.abs(a[:,0]),0,1))
        a[:,2] = np.clip(a[:,2]*6.0, 0.0, 1.0)
        a[:,0] = (1-SMOOTH)*prev_actions[:,0] + SMOOTH*a[:,0]
        dsteer = np.abs(a[:,0]-prev_actions[:,0]); prev_actions = a.copy()

        next_obs, rewards, terms, truncs, _ = envs.step(a)
        dones = np.logical_or(terms, truncs)
        rewards = rewards + (-0.002*dsteer)

        rb.extend_batch(
            obs.astype(np.uint8, copy=False),
            a.astype(np.float32, copy=False),
            next_obs.astype(np.uint8, copy=False),
            rewards.astype(np.float32, copy=False),
            dones.astype(np.float32, copy=False),
        )

        # 学习：每个学习时刻执行 UPS 次反向
        if step >= TRAIN_AFTER and step % TRAIN_FREQ == 0:
            for _ in range(UPS):
                info = agent.train_step(rb, batch_size, GAMMA, TAU, POLICY_DELAY, step)
            if info is not None and step % 500 == 0:
                writer.add_scalar("loss/critic1", info["critic_loss1"], step)
                writer.add_scalar("loss/critic2", info["critic_loss2"], step)
                if "actor_loss" in info: writer.add_scalar("loss/actor", info["actor_loss"], step)
                writer.add_scalar("target/mean_q", info["target"], step)

        if step % LOG_EVERY == 0 and step > 0:
            el = time.time() - start_t; fps = (LOG_EVERY * num_envs) / max(el, 1e-6)
            print(f"step={step} | fps~{fps:.1f} | rb={len(rb)} | sigma={max(0.1*(1 - step/TOTAL_STEPS), 0.01):.3f}")
            writer.add_scalar("sys/fps", fps, step); writer.flush(); start_t = time.time()

        if SAVE_EVERY>0 and step >= next_save_step:
            save_ckpt(agent, step); next_save_step += SAVE_EVERY
        if AUTO_EVAL_EVERY>0 and step >= next_eval_step:
            try:
                _, mean_ret = auto_eval_video(agent, step, device); writer.add_scalar("eval/return", mean_ret, step)
            except Exception as e:
                print(f"[WARN] Auto-eval failed: {e}")
            finally:
                next_eval_step += AUTO_EVAL_EVERY

        s = preprocess(next_obs).to(device).contiguous(memory_format=torch.channels_last)
        obs = next_obs; step += num_envs
        if pbar: pbar.update(num_envs)
    if pbar: pbar.close()
    save_ckpt(agent, step); envs.close(); writer.close()
