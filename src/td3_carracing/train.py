"""Train the publication edition, then evaluate on fixed independent seeds."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
import random
import statistics

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .agent import TD3Agent, TD3Config
from .env import FrameStackState, make_env
from .evaluate import evaluate_actor, write_results
from .replay import ReplayBuffer


REWARDS = {"raw": None, "offtrack-small": -0.1, "offtrack-strong": -1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward", choices=REWARDS, default="offtrack-small")
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--sigma-decay", type=float, default=0.99)
    parser.add_argument("--replay-capacity", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--actor-lr", type=float, default=2.5e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[1001, 1002, 1003, 1004, 1005])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.batch_size < 1 or args.replay_capacity < args.batch_size:
        raise ValueError("episodes and batch-size must be positive; replay-capacity must cover a batch")
    if args.eval_every < 1 or not args.eval_seeds:
        raise ValueError("eval-every and eval-seeds must be nonempty and positive")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TD3Config(actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                    tau=args.tau, policy_delay=args.policy_delay)
    manifest = {
        "environment": "CarRacing-v3", "algorithm": "TD3", "seed": args.seed,
        "training": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
        "td3": asdict(cfg),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "gymnasium": gym.__version__, "numpy": np.__version__},
        "evaluation_note": "Raw environment return on fixed seeds, with exploration off",
    }
    (args.run_dir / "config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    agent = TD3Agent(device, cfg)
    replay = ReplayBuffer(args.replay_capacity)
    env = make_env(off_track_reward=REWARDS[args.reward])
    writer = SummaryWriter(str(args.run_dir / "tensorboard"))
    csv_path = args.run_dir / "episodes.csv"
    eval_rows: list[dict] = []
    sigma = args.sigma
    environment_steps = 0
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            log = csv.DictWriter(handle, fieldnames=[
                "episode", "train_return", "agent_steps", "sigma", "eval_mean_score"
            ])
            log.writeheader()
            for episode in range(1, args.episodes + 1):
                obs, _ = env.reset(seed=args.seed + episode - 1)
                stack = FrameStackState()
                state = stack.reset(obs)
                train_return = 0.0
                eval_mean = ""
                for step in range(1, args.max_steps + 1):
                    action = agent.select_action(state, sigma=sigma)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    next_state = stack.append(obs)
                    done = terminated or truncated
                    replay.add(state, action, next_state, reward, done)
                    environment_steps += 1
                    if environment_steps >= args.warmup_steps:
                        metrics = agent.update(replay, args.batch_size)
                        if metrics and environment_steps % 100 == 0:
                            for name, value in metrics.items():
                                writer.add_scalar(f"loss/{name}", value, environment_steps)
                    state = next_state
                    train_return += float(reward)
                    if done:
                        break

                writer.add_scalar("train/return", train_return, episode)
                writer.add_scalar("train/sigma", sigma, episode)
                if episode % args.eval_every == 0 or episode == args.episodes:
                    checkpoint = args.run_dir / f"actor_ep{episode}.pt"
                    agent.save_actor(str(checkpoint))
                    results = evaluate_actor(agent.actor, args.eval_seeds, args.max_steps)
                    eval_mean = statistics.mean(float(row["score"]) for row in results)
                    writer.add_scalar("eval/mean_score", eval_mean, episode)
                    eval_rows.extend({"episode": episode, "checkpoint": str(checkpoint), **row}
                                     for row in results)
                    write_results(args.run_dir / f"eval_ep{episode}.csv", results, str(checkpoint))
                log.writerow({"episode": episode, "train_return": train_return,
                              "agent_steps": step, "sigma": sigma,
                              "eval_mean_score": eval_mean})
                handle.flush()
                print(f"episode={episode} train_return={train_return:.2f} eval_mean={eval_mean}")
                sigma = max(args.sigma_min, sigma * args.sigma_decay)
    finally:
        env.close()
        writer.close()
    if eval_rows:
        with (args.run_dir / "evaluation_all.csv").open("w", newline="", encoding="utf-8") as handle:
            log = csv.DictWriter(handle, fieldnames=[
                "episode", "checkpoint", "seed", "score", "agent_steps", "terminated", "truncated"
            ])
            log.writeheader()
            log.writerows(eval_rows)


if __name__ == "__main__":
    main()
