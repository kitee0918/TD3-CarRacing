"""Independent deterministic evaluation on fixed CarRacing-v3 seeds."""

import argparse
import csv
from pathlib import Path
import statistics

import torch

from .env import FrameStackState, make_env
from .model import Actor


def evaluate_actor(actor: Actor, seeds: list[int], max_steps: int = 1200,
                   render: bool = False) -> list[dict[str, float | int | bool]]:
    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    was_training = actor.training
    actor.eval()
    device = next(actor.parameters()).device
    env = make_env(render_mode="human" if render else None, off_track_reward=None)
    results = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            stack = FrameStackState()
            state = stack.reset(obs)
            total = 0.0
            terminated = truncated = False
            for step in range(1, max_steps + 1):
                with torch.inference_mode():
                    tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
                    action = actor(tensor)[0].cpu().numpy()
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                state = stack.append(obs)
                if terminated or truncated:
                    break
            results.append({
                "seed": seed,
                "score": total,
                "agent_steps": step,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
    finally:
        env.close()
        actor.train(was_training)
    return results


def write_results(path: Path, results: list[dict], checkpoint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "checkpoint", "seed", "score", "agent_steps", "terminated", "truncated"
        ])
        writer.writeheader()
        for result in results:
            writer.writerow({"checkpoint": checkpoint, **result})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002, 1003, 1004, 1005])
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.csv"))
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor().to(device)
    actor.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    scores = evaluate_actor(actor, args.seeds, args.max_steps, args.render)
    write_results(args.output, scores, str(args.checkpoint))
    print(f"mean_score={statistics.mean(float(row['score']) for row in scores):.2f}")
    print(f"min_score={min(float(row['score']) for row in scores):.2f}")
    print(f"results={args.output}")


if __name__ == "__main__":
    main()
