
#!/usr/bin/env python3
# file: env_fps_benchmark.py
"""
Environment-only throughput benchmark for CarRacing-v3 with your wrappers.
- Measures vectorized env stepping FPS across multiple NUM_ENVS settings.
- No neural nets involved (CPU-bound). Helps choose NUM_ENVS that your CPU can sustain.
- Uses the same wrap_obs + FrameStack(4) pipeline as your training code.
- Compatible with different Gymnasium versions (AutoresetMode fallback).

Usage examples:
  python env_fps_benchmark.py --num_envs 4,8,12,16,24 --steps_per_env 1000
  python env_fps_benchmark.py --num_envs 8-32 --steps_per_env 1500 --log_every 200

Output: prints table + saves CSV (env_fps_results.csv).
"""
import argparse
import time
import math
import csv
from typing import List, Tuple
import numpy as np

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

# ----- robust FrameStack import (matches training script) -----
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

# ----- observation wrapper (matches training script) -----
from gymnasium import ObservationWrapper, spaces
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

def make_env_fn(seed: int = 0):
    def _thunk():
        env = gym.make("CarRacing-v3", render_mode=None)
        env = wrap_obs(env)
        env = FrameStack(env, 4)
        env.reset(seed=seed)
        return env
    return _thunk

def parse_list(s: str) -> List[int]:
    out = set()
    for tok in s.split(','):
        tok = tok.strip()
        if not tok: 
            continue
        if '-' in tok:
            a, b = tok.split('-')
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            for v in range(a, b + step, step):
                out.add(int(v))
        else:
            out.add(int(tok))
    return sorted(out)

def bench_envs(num_envs: int, steps_per_env: int, log_every: int) -> Tuple[float, float]:
    try:
        from gymnasium.vector import AutoresetMode
        envs = AsyncVectorEnv([make_env_fn(42 + i) for i in range(num_envs)], autoreset_mode=AutoresetMode.NEXT_STEP)
    except Exception:
        envs = AsyncVectorEnv([make_env_fn(42 + i) for i in range(num_envs)])
    obs, infos = envs.reset()

    # warmup few steps
    a = np.stack([envs.single_action_space.sample() for _ in range(num_envs)], axis=0)
    for _ in range(10):
        obs, r, term, trunc, infos = envs.step(a)

    total_steps = 0
    total_target = steps_per_env * num_envs
    start = time.perf_counter()
    last = start
    while total_steps < total_target:
        a = np.stack([envs.single_action_space.sample() for _ in range(num_envs)], axis=0)
        obs, r, term, trunc, infos = envs.step(a)
        total_steps += num_envs
        # periodic progress without heavy prints
        if ((total_steps // num_envs) % log_every) == 0:
            now = time.perf_counter()
            span = now - last
            fps = (log_every * num_envs) / max(span, 1e-6)
            print(f"[{total_steps}/{total_target}] ~{fps:.1f} it/s")
            last = now
    elapsed = time.perf_counter() - start
    envs.close()
    overall_fps = total_steps / elapsed
    time_per_iter = elapsed / (total_steps / num_envs)
    return overall_fps, time_per_iter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", type=str, default="4,8,12,16,24")
    ap.add_argument("--steps_per_env", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--csv", type=str, default="env_fps_results.csv")
    args = ap.parse_args()

    candidates = parse_list(args.num_envs)
    rows = [("num_envs", "steps_per_env", "fps_total(it/s)", "time_per_iter(s)")]
    best = 0.0
    best_envs = None
    print(f"[bench] candidates={candidates} steps_per_env={args.steps_per_env}")
    for ne in candidates:
        try:
            fps, tpi = bench_envs(ne, args.steps_per_env, args.log_every)
            rows.append((ne, args.steps_per_env, f"{fps:.1f}", f"{tpi:.6f}"))
            print(f"[ok] num_envs={ne:<3d} fps={fps:>8.1f} it/s | time/iter={tpi:.6f}s")
            if fps > best:
                best, best_envs = fps, ne
        except Exception as e:
            rows.append((ne, args.steps_per_env, "ERR", "ERR"))
            print(f"[err] num_envs={ne} failed: {e}")

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f); w.writerows(rows)

    print("\n===== Summary =====")
    for r in rows[1:]:
        print(r)

    # Heuristic recommendation: pick smallest num_envs whose fps >= 95% of best
    if best_envs is not None:
        thr95 = 0.95 * best
        vals = [(int(r[0]), float(r[2])) for r in rows[1:] if r[2] not in ("ERR")]
        vals.sort()
        rec = next((ne for ne, fps in vals if fps >= thr95), best_envs)
        print(f"\n[recommendation] environment-limited best≈{best_envs} (fps={best:.1f}). "
              f"Use NUM_ENVS≈{rec} (≥95% of best) unless memory/CPU is constrained.")
        print(f"[saved] CSV → {args.csv}")

if __name__ == "__main__":
    main()
