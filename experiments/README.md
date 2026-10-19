# Experiment evidence

`raw/2025-11-gitee-experiment_log.csv` was committed to the 2025 course branch. `raw/2025-11-local-experiment_log.csv` came from a separate local checkout. The latter contains 24-column old-format rows followed by 25-column new-format rows under one old header. The normalization script keeps both sources identifiable:

```bash
python -m tools.normalize_2025_logs \
  experiments/raw/2025-11-gitee-experiment_log.csv \
  experiments/raw/2025-11-local-experiment_log.csv \
  --output experiments/results_2025_training.csv
```

The normalized file contains **training-only** columns. `session` increments when the same `run_name` starts its episode counter over; this is an inference from the CSV order, not a recovered run identifier.

| Source and run | Episode 200 reward | Best training reward so far | Replay capacity |
| --- | ---: | ---: | ---: |
| Committed Gitee CSV, `run_000_decay_0.990000`, inferred session 1 | 591.53 | 803.70 | 65,536 |
| Separate local CSV, same run name | 659.32 | 865.65 | 102,400 |

The 2025 scripts recorded TensorBoard training reward and generated evaluation videos, but no complete per-seed independent evaluation table was found. Scores in these CSVs must not be labeled test scores. The retained `checkpoints/course_2025_actor_ep200.pt` came from the committed Gitee branch; because that branch reused the `agent_opt_200.pt` path across sessions, its relation to either row above is ambiguous until evaluated again.

The older `run_baseline`, `run_more_brake`, and `run_less_explore` logs changed brake and exploration settings together. The two decay logs also changed replay capacity. They document the search process, but they are not controlled single-variable comparisons. For new experiments, run several training seeds, hold other hyperparameters fixed, and evaluate each checkpoint on the same separate seed list.
