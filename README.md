# TD3 for CarRacing-v3 · 2025 course project

This repository presents the **2025 course work** built on [Yanran Zhang's 2023 TD3-CarRacing implementation](https://github.com/Zhangyr2022/TD3-CarRacing). The original license and attribution are retained in [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). The genuine November 2025 course commits are preserved on `main`; October 2025 local scripts are archived with their actual provenance. No historical commit dates were invented.

## What is here

| Directory | Purpose |
| --- | --- |
| `src/td3_carracing/` | Maintained PyTorch Actor, twin Critic, compact replay, CarRacing-v3 wrappers, TD3 updates, training and evaluation |
| `experiments/raw/` | Original November 2025 CSV exports, kept separate because their schemas differ |
| `experiments/results_2025_training.csv` | Normalized **training** log with source line and inferred session number |
| `checkpoints/` | One historical actor snapshot for demonstration; its score has **not** been independently verified |
| `legacy/oct_2025/` | Selected local parallel-training and benchmark work from October 29–November 2, 2025 |
| `legacy/nov_2025/` | Course scripts preserved from the November Git history |
| `docs/` | Development chronology and the source audit |

The maintained TD3 implementation includes clipped double Q targets, target policy smoothing, delayed actor updates, and soft target updates. The 2025 high-score script in `legacy/nov_2025/optimal_d_csdn.py` set `actor_update_freq=1`; its historical score must **not** be attributed to delayed updates. The publication edition defaults to `policy_delay=2` and needs its own training and evaluation results.

## Install

Use a Python environment with a compatible PyTorch build for your operating system and accelerator. Then, from this repository root:

```bash
python -m pip install -e .
```

The `legacy` scripts additionally need `python -m pip install -e ".[legacy]"`. The exact 2025 package versions were not recorded, so the historical runs are evidence of the course work rather than a verified bit-for-bit reproducible environment. A fresh package lock and environment smoke test should be recorded before citing results from the maintained edition.

## Evaluate the included checkpoint

```bash
td3-carracing-eval \
  --checkpoint checkpoints/course_2025_actor_ep200.pt \
  --seeds 1001 1002 1003 1004 1005 \
  --output runs/checkpoint_ep200_eval.csv
```

This command uses raw CarRacing-v3 rewards and no exploration noise. The original CSV reused the same run name and checkpoint path across multiple sessions; the exact training row corresponding to the retained checkpoint is uncertain. The command above is the way to establish a new, attributable test score. No test score is claimed in this README.

## Train a new run

```bash
td3-carracing-train \
  --run-dir runs/offtrack-small-seed42 \
  --episodes 200 --seed 42 \
  --reward offtrack-small --sigma-decay 0.99 \
  --policy-delay 2 --eval-every 50 \
  --eval-seeds 1001 1002 1003 1004 1005
```

The three reward modes are `raw`, `offtrack-small` (replace off-track frame reward with -0.1), and `offtrack-strong` (-1.0). Runs write a configuration manifest, per-episode training CSV, TensorBoard events, checkpoints, and per-seed evaluation CSV in the supplied `--run-dir`. The directory must not exist yet, preventing accidental result overwrite. Keep all other parameters and evaluation seeds fixed when comparing reward or exploration settings.

## Historical result boundary

At episode 200, one committed 2025 training log reports `best_reward_so_far=803.70`, with that episode's reward `591.53`. A separate local log reports `best_reward_so_far=865.65`, with that episode's reward `659.32`. These are **historical best training returns**, not independent test scores. The logs use different replay capacities, and the run name was reused, so they cannot be treated as repeated identical trials. See [experimental evidence](experiments/README.md).

## History and credit

[Development history](docs/DEVELOPMENT_HISTORY.md) maps the original 2025 commits and the uncommitted October snapshot. [The audit](docs/PROJECT_AUDIT_2025.md) records the pre-publication file analysis. The 2023 implementation and course team contributions are credited in [NOTICE.md](NOTICE.md).
