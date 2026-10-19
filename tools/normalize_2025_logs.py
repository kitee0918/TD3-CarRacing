"""Normalize the two incompatible November 2025 experiment CSV formats.

This intentionally keeps training returns distinct from independent evaluation.
Session numbers are inferred whenever a run name's episode counter restarts.
"""

import argparse
import csv
from pathlib import Path


NEW_FIELDS = [
    "run_idx", "run_name", "seed", "episode", "is_final", "model_path",
    "episode_reward", "best_reward_so_far", "total_steps", "sigma", "actor_lr",
    "critic_lr", "gamma", "batch_size", "num_episodes", "max_episode_steps",
    "replay_capacity", "tau", "initial_sigma", "min_sigma",
    "sigma_decay_factor", "env_name", "frame_stack", "img_size", "skip_frames",
]
OUTPUT_FIELDS = [
    "source", "source_line", "run_name", "session", "seed", "episode",
    "train_episode_reward", "best_train_reward_so_far", "sigma",
    "replay_capacity", "env_name", "model_path",
]


def normalize_files(paths: list[Path]) -> list[dict[str, str | int]]:
    result = []
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if len(header) not in (24, 25):
                raise ValueError(f"Unexpected header width in {path}: {len(header)}")
            session: dict[str, int] = {}
            last_episode: dict[str, int] = {}
            for line, raw in enumerate(reader, start=2):
                if not raw:
                    continue
                if len(raw) == len(header):
                    row = dict(zip(header, raw))
                elif len(header) == 24 and len(raw) == 25:
                    row = dict(zip(NEW_FIELDS, raw))
                else:
                    raise ValueError(f"Unexpected row width in {path}:{line}: {len(raw)}")
                run_name = row["run_name"]
                episode = int(row.get("episode", row.get("epoch", "0")))
                restarted = (
                    run_name in last_episode
                    and (episode < last_episode[run_name]
                         or (episode == last_episode[run_name]
                             and row.get("is_final") != "1"))
                )
                if restarted:
                    session[run_name] += 1
                else:
                    session.setdefault(run_name, 1)
                last_episode[run_name] = episode
                result.append({
                    "source": path.name,
                    "source_line": line,
                    "run_name": run_name,
                    "session": session[run_name],
                    "seed": row.get("seed", ""),
                    "episode": episode,
                    "train_episode_reward": row["episode_reward"],
                    "best_train_reward_so_far": row["best_reward_so_far"],
                    "sigma": row["sigma"],
                    "replay_capacity": row["replay_capacity"],
                    "env_name": row.get("env_name", ""),
                    "model_path": row["model_path"],
                })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = normalize_files(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} historical training rows to {args.output}")


if __name__ == "__main__":
    main()
