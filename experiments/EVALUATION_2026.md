# Independent checkpoint evaluation · 2026-10-19

This is a new evaluation of the archived 2025 actor, not a recovered 2025 test result. The checkpoint is `checkpoints/course_2025_actor_ep200.pt` (SHA-256 `6408de89116bc52bc27110387b7b712594633e1b70fa2e912102b834907f12a3`). Its exact relation to the historical episode-200 CSV row remains uncertain because the original run path was reused.

Run from the repository root after `python -m pip install -e .`:

```bash
td3-carracing-eval --checkpoint checkpoints/course_2025_actor_ep200.pt --seeds 1001 1002 1003 1004 1005 --output runs/checkpoint_ep200_eval.csv
```

The actor uses no exploration noise. Evaluation uses unmodified CarRacing-v3 reward, five fixed seeds, five simulator frames per actor step, no rendering, and a 1,200 actor-step limit. Each episode reached CarRacing's own 1,000-frame truncation at 200 actor steps. The per-seed output is in [checkpoint_ep200_eval_2026-10-19.csv](checkpoint_ep200_eval_2026-10-19.csv).

| Seed | Raw return |
| ---: | ---: |
| 1001 | 355.13 |
| 1002 | 507.27 |
| 1003 | 526.67 |
| 1004 | 504.03 |
| 1005 | 504.29 |

Mean: **479.48**; minimum: **355.13**; maximum: **526.67**. This evaluation does not support a claim that the archived checkpoint achieved an independent test score above 800. The 800+ numbers in the 2025 CSVs are training best-so-far returns with a different reward setup.

The checked environment was Windows, CPU, Python 3.11.17, PyTorch 2.14.1, Gymnasium 1.4.0, Box2D 2.3.10, Pygame CE 2.5.8, OpenCV 5.0.0.93, and NumPy 2.4.6. The full package-version snapshot is in [constraints_eval_2026-10-19.txt](constraints_eval_2026-10-19.txt). All three tests in `python -m unittest discover -s tests -v` passed in this environment before evaluation. A one-episode, one-step training CLI smoke run also completed and wrote its CSV, checkpoint, and TensorBoard artifacts; it was deliberately too short to test learning. These versions identify this evaluation run; they are not claimed to reproduce the unknown 2025 training environment exactly.
