
#!/usr/bin/env python3
# file: td3_gpu_benchmark.py
"""
TD3-like GPU benchmark (CarRacing model shape).

Goal: measure pure training throughput (samples/sec) independent of env step bottlenecks.
- Re-implements Actor/Critic identical to your training script (84x84x4 default).
- Runs critic updates every step, actor update every POLICY_DELAY steps.
- Supports AMP and torch.compile.
- Reports: throughput, time/step, max VRAM, best batch & recommended NUM_ENVS.

Usage:
  python td3_gpu_benchmark.py --batches 256,384,512,768,1024 --steps 400 --amp 1 --compile 0
  python td3_gpu_benchmark.py --help

Why pinned/host feed is optional:
- Default generates tensors on GPU to stress compute. Use --host_feed=1 to include H2D copy cost.
"""
import argparse
import time
import math
import csv
from typing import List, Tuple

import torch
from torch import nn

# ----------------------
# Model (matches training)
# ----------------------
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
        a_out[:, 0] = a[:, 0]
        a_out[:, 1] = (a[:, 1] + 1) * 0.5
        a_out[:, 2] = (a[:, 2] + 1) * 0.015
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


def make_models(h: int, c: int, action_dim: int, device: str, use_compile: bool):
    actor = Actor(h, action_dim, c).to(device)
    c1 = Critic(h, action_dim, c).to(device)
    c2 = Critic(h, action_dim, c).to(device)
    c1_t = Critic(h, action_dim, c).to(device)
    c2_t = Critic(h, action_dim, c).to(device)
    actor_t = Actor(h, action_dim, c).to(device)
    c1_t.load_state_dict(c1.state_dict())
    c2_t.load_state_dict(c2.state_dict())
    actor_t.load_state_dict(actor.state_dict())
    if use_compile and hasattr(torch, "compile"):
        actor = torch.compile(actor, mode="max-autotune")
        c1 = torch.compile(c1, mode="max-autotune")
        c2 = torch.compile(c2, mode="max-autotune")
        actor_t = torch.compile(actor_t, mode="max-autotune", fullgraph=False)
        c1_t = torch.compile(c1_t, mode="max-autotune", fullgraph=False)
        c2_t = torch.compile(c2_t, mode="max-autotune", fullgraph=False)
    return actor, c1, c2, actor_t, c1_t, c2_t


def bench_once(
    batch_size: int,
    steps: int,
    h: int,
    w: int,
    c: int,
    action_dim: int,
    policy_delay: int,
    gamma: float,
    tau: float,
    amp: bool,
    use_compile: bool,
    host_feed: bool,
    device: str,
) -> Tuple[float, float, float]:
    torch.backends.cudnn.benchmark = True
    actor, c1, c2, actor_t, c1_t, c2_t = make_models(h, c, action_dim, device, use_compile)
    opt_a = torch.optim.Adam(actor.parameters(), lr=1e-4)
    opt_c1 = torch.optim.Adam(c1.parameters(), lr=1e-4)
    opt_c2 = torch.optim.Adam(c2.parameters(), lr=1e-4)
    mse = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.startswith("cuda"))

    # allocate tensors
    shape = (batch_size, c, h, w)
    if host_feed:
        s_cpu = torch.randn(shape, pin_memory=True)
        ns_cpu = torch.randn(shape, pin_memory=True)
        a_cpu = torch.randn(batch_size, action_dim, pin_memory=True)
        r_cpu = torch.randn(batch_size, 1, pin_memory=True)
        d_cpu = torch.rand(batch_size, 1, pin_memory=True).bernoulli(0.05)
    else:
        s = torch.randn(shape, device=device)
        ns = torch.randn(shape, device=device)
        a = torch.randn(batch_size, action_dim, device=device)
        r = torch.randn(batch_size, 1, device=device)
        d = torch.rand(batch_size, 1, device=device).bernoulli(0.05)

    # warmup
    torch.cuda.synchronize() if device.startswith("cuda") else None
    for _ in range(10):
        pass
    torch.cuda.synchronize() if device.startswith("cuda") else None

    # measure
    start = time.perf_counter()
    it_actor = 0
    for t in range(steps):
        if host_feed:
            s = s_cpu.to(device, non_blocking=True)
            ns = ns_cpu.to(device, non_blocking=True)
            a = a_cpu.to(device, non_blocking=True)
            r = r_cpu.to(device, non_blocking=True)
            d = d_cpu.to(device, non_blocking=True)

        # target action with per-dim clamp
        with torch.no_grad():
            noise = (0.2 * torch.randn_like(a)).clamp(-0.5, 0.5)
            na = actor_t(ns) + noise
            na = torch.max(na, torch.tensor([-1.0, 0.0, 0.0], device=device))
            na = torch.min(na, torch.tensor([ 1.0, 1.0, 1.0], device=device))
            tq1 = c1_t(ns, na)
            tq2 = c2_t(ns, na)
            y = r + gamma * torch.minimum(tq1, tq2) * (1.0 - d)

        # critic step
        opt_c1.zero_grad(set_to_none=True)
        opt_c2.zero_grad(set_to_none=True)
        if amp and device.startswith("cuda"):
            with torch.cuda.amp.autocast():
                q1 = c1(s, a)
                q2 = c2(s, a)
                l1 = mse(q1, y)
                l2 = mse(q2, y)
            scaler.scale(l1).backward()
            scaler.step(opt_c1)
            scaler.scale(l2).backward()
            scaler.step(opt_c2)
            scaler.update()
        else:
            q1 = c1(s, a)
            q2 = c2(s, a)
            l1 = mse(q1, y)
            l2 = mse(q2, y)
            l1.backward()
            opt_c1.step()
            l2.backward()
            opt_c2.step()

        # actor step (delay)
        if (t % policy_delay) == 0:
            opt_a.zero_grad(set_to_none=True)
            if amp and device.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    pa = actor(s)
                    aloss = -c1(s, pa).mean()
                scaler.scale(aloss).backward()
                scaler.step(opt_a)
                scaler.update()
            else:
                pa = actor(s)
                aloss = -c1(s, pa).mean()
                aloss.backward()
                opt_a.step()
            with torch.no_grad():
                for p, tp in zip(actor.parameters(), actor_t.parameters()):
                    tp.data.mul_(1 - tau).add_(tau * p.data)
                for p, tp in zip(c1.parameters(), c1_t.parameters()):
                    tp.data.mul_(1 - tau).add_(tau * p.data)
                for p, tp in zip(c2.parameters(), c2_t.parameters()):
                    tp.data.mul_(1 - tau).add_(tau * p.data)
            it_actor += 1

    torch.cuda.synchronize() if device.startswith("cuda") else None
    elapsed = time.perf_counter() - start
    samples = batch_size * steps
    thr = samples / elapsed
    mem_gb = 0.0
    if device.startswith("cuda"):
        mem_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats()
    return thr, elapsed / steps, mem_gb


def parse_batches(s: str) -> List[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            a, b = int(a), int(b)
            out += list(range(a, b + 1, 64))
        else:
            out.append(int(tok))
    # unique sorted
    return sorted(set(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h", type=int, default=84)
    p.add_argument("--w", type=int, default=84)
    p.add_argument("--c", type=int, default=4)
    p.add_argument("--action_dim", type=int, default=3)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batches", type=str, default="256,384,512,768")
    p.add_argument("--policy_delay", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.985)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--compile", type=int, default=0)
    p.add_argument("--host_feed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--csv", type=str, default="td3_bench_results.csv")
    args = p.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"

    print(f"[bench] device={device} amp={bool(args.amp)} compile={bool(args.compile)} host_feed={bool(args.host_feed)}")
    if device.startswith("cuda"):
        i = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(i)
        print(f"[gpu] name={prop.name} total_mem_gb={prop.total_memory/1024**3:.2f} sm={prop.multi_processor_count} capability={prop.major}.{prop.minor}")

    batches = parse_batches(args.batches)
    rows = [("batch", "steps", "throughput(samples/s)", "time/iter(s)", "max_reserved_mem(GB)")]
    best_thr = 0.0
    best_batch = None
    for bs in batches:
        try:
            thr, tpi, mem_gb = bench_once(
                batch_size=bs,
                steps=args.steps,
                h=args.h,
                w=args.w,
                c=args.c,
                action_dim=args.action_dim,
                policy_delay=args.policy_delay,
                gamma=args.gamma,
                tau=args.tau,
                amp=bool(args.amp),
                use_compile=bool(args.compile),
                host_feed=bool(args.host_feed),
                device=device,
            )
            rows.append((bs, args.steps, f"{thr:.1f}", f"{tpi:.6f}", f"{mem_gb:.2f}"))
            print(f"[ok] bs={bs:<4d} thr={thr:>8.1f} samp/s | t/iter={tpi:.6f}s | mem={mem_gb:.2f}GB")
            if thr > best_thr:
                best_thr, best_batch = thr, bs
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                rows.append((bs, args.steps, "OOM", "NA", "NA"))
                print(f"[oom] bs={bs} → out of memory")
                torch.cuda.empty_cache() if device.startswith("cuda") else None
            else:
                raise

    # save csv
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)

    print("\n===== Summary =====")
    for r in rows[1:]:
        print(r)
    if best_batch is not None:
        # map best batch to recommended NUM_ENVS per training rule: batch_size = min(768, NUM_ENVS*64)
        target_batch = min(768, best_batch)
        rec_num_envs = max(2, math.ceil(target_batch / 64))
        print(f"\n[recommendation] best_batch={best_batch} → set NUM_ENVS≈{rec_num_envs} (to hit batch≈{target_batch}).")
        print("If 'sys/fps' from your training logs is low, prefer increasing NUM_ENVS first; else increase TRAIN_FREQ.")
    print(f"[saved] CSV → {args.csv}")
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
