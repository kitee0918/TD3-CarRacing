
#!/usr/bin/env python3
# file: td3_carracing_eval.py
"""
Evaluate & record a demo video for a saved TD3 CarRacing actor.
- Works with checkpoints like: ./model/agent_step500006.pt (torch.save(actor, ...))
- Reuses the same wrappers (grayscale->resize->FrameStack(4)).
- Records MP4 if MoviePy is available; otherwise runs without video.

Usage:
  python td3_carracing_eval.py --ckpt "./model/agent_step500006.pt" --episodes 1 --out "./demo_videos"
  python td3_carracing_eval.py --ckpt "./model/agent_step*.pt" --episodes 2 --render human --save_mp4 0
"""
import argparse, glob, os, re, sys, time
from typing import Tuple
import numpy as np, torch
from torch import nn
import gymnasium as gym

# ---------- Robust FrameStack ----------
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
            super().__init__(env); self.k = k
            self.frames = _deque(maxlen=k)
            assert isinstance(env.observation_space, gym.spaces.Box)
            low, high = env.observation_space.low, env.observation_space.high
            self.observation_space = gym.spaces.Box(low=np.repeat(low,k,axis=-1), high=np.repeat(high,k,axis=-1), dtype=env.observation_space.dtype)
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

from gymnasium import ObservationWrapper, spaces
def wrap_obs(env):
    try:
        from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
        env = GrayScaleObservation(env, keep_dim=True); env = ResizeObservation(env, (84, 84)); return env
    except Exception:
        class GrayResize84(ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                self.observation_space = spaces.Box(low=0, high=255, shape=(84,84,1), dtype=np.uint8)
            def observation(self, obs):
                gray = (0.299*obs[:,:,0]+0.587*obs[:,:,1]+0.114*obs[:,:,2]).astype(np.float32)
                try:
                    import cv2; gray84 = cv2.resize(gray, (84,84), interpolation=cv2.INTER_AREA)
                except Exception:
                    ys = (np.linspace(0, gray.shape[0]-1, 84)).astype(np.int32)
                    xs = (np.linspace(0, gray.shape[1]-1, 84)).astype(np.int32)
                    gray84 = gray[np.ix_(ys, xs)]
                return np.clip(gray84, 0, 255).astype(np.uint8)[...,None]
        return GrayResize84(env)

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
            nn.Linear(256, 3), nn.LayerNorm(3), nn.Tanh(),
        )
    def forward(self, x: torch.Tensor, sigma: float = 0.0):
        h = self.conv(x); h = torch.flatten(h, start_dim=1); a = self.head(h)
        if sigma > 0: a = a + sigma * torch.randn_like(a)
        a_out = a.clone(); a_out[:,0] = a[:,0]; a_out[:,1] = (a[:,1]+1)*0.5; a_out[:,2] = (a[:,2]+1)*0.015; return a_out

def preprocess(obs: np.ndarray) -> torch.Tensor:
    if obs.ndim == 3: obs = obs[None, ...]
    t = torch.from_numpy(obs).float()
    if t.max() > 1.5: t = t / 255.0
    return t.permute(0,3,1,2).contiguous()

def get_device(): return "cuda" if torch.cuda.is_available() else "cpu"

import glob, re, os
def parse_ckpt(path_glob: str) -> str:
    paths = sorted(glob.glob(path_glob)); 
    if not paths: raise FileNotFoundError(f"No checkpoint matches: {path_glob}")
    def step_num(p: str):
        m = re.search(r"step(\d+)", os.path.basename(p)); return int(m.group(1)) if m else -1
    return max(paths, key=step_num)

def load_actor(ckpt_path: str, device: str, in_ch: int, state_dim: int) -> nn.Module:
    try:
        obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        import __main__
        import torch.serialization as torch_serialization
        try: torch_serialization.add_safe_globals([__main__.Actor])
        except Exception: pass
        obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, nn.Module):
        actor = obj.to(device)
    else:
        actor = Actor(state_dim, 3, in_ch).to(device)
        actor.load_state_dict(obj, strict=False)
    actor.eval(); return actor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=3000)
    ap.add_argument("--render", type=str, default="rgb_array", choices=["rgb_array","human"])
    ap.add_argument("--save_mp4", type=int, default=1)
    ap.add_argument("--out", type=str, default="./demo_videos")
    args = ap.parse_args()

    device = get_device()
    ckpt_path = parse_ckpt(args.ckpt)
    os.makedirs(args.out, exist_ok=True)

    env = gym.make("CarRacing-v3", render_mode=args.render)
    # Attach RecordVideo if possible
    have_record = False
    if args.render == "rgb_array" and args.save_mp4:
        try:
            from gymnasium.wrappers import RecordVideo
            have_record = True
        except Exception:
            try:
                from gymnasium.wrappers.record_video import RecordVideo
                have_record = True
            except Exception:
                have_record = False
    env = wrap_obs(env); env = FrameStack(env, 4)
    H, W, C = env.observation_space.shape

    prefix = os.path.splitext(os.path.basename(ckpt_path))[0]
    if have_record:
        clip_dir = os.path.join(args.out, f"{prefix}")
        os.makedirs(clip_dir, exist_ok=True)
        env = RecordVideo(env, video_folder=clip_dir, episode_trigger=lambda e: True, name_prefix=prefix)
        print(f"[rec] Recording → {clip_dir}")
    elif args.save_mp4 and args.render == "rgb_array":
        print("⚠️ moviepy/RecordVideo not available; run: pip install \"gymnasium[other]\" (continuing without MP4).")

    actor = load_actor(ckpt_path, device, in_ch=C, state_dim=H)
    print(f"[info] Loaded: {ckpt_path} | device={device}")

    rets = []
    for ep in range(args.episodes):
        obs, info = env.reset(); done = False; steps = 0; ep_ret = 0.0
        while not done and steps < args.max_frames:
            with torch.no_grad():
                s = preprocess(obs)
                a = actor(s.to(device), sigma=0.0).cpu().numpy()[0].astype(np.float64)
            obs, r, term, trunc, info = env.step(a)
            done = bool(term) or bool(trunc)
            ep_ret += float(r); steps += 1
        rets.append(ep_ret)
        print(f"[ep {ep+1}/{args.episodes}] return={ep_ret:.2f} steps={steps}")
    env.close()
    print(f"[done] episodes={args.episodes} mean_return={float(np.mean(rets)):.2f}")
    if have_record: print(f"[video] saved under: {os.path.join(args.out, prefix)}")

if __name__ == "__main__":
    main()
