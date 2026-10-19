# Development chronology and Git provenance

All dates below are Asia/Shanghai unless explicitly noted. The original `main` ancestry was imported from the local Gitee checkout's `origin/kite_long_obsv` ref. The November author/committer dates and messages were retained. Publication commits have their actual later dates.

| Date | Evidence | Development milestone |
| --- | --- | --- |
| 2023-06-01 | Separate upstream Git history | Yanran Zhang's original TD3-CarRacing source and MIT license. This is the base project, not a 2025 course contribution. |
| 2025-10-29 to 11-02 | Local file modification dates in `legacy/oct_2025/` | Gymnasium/CarRacing-v3 adaptation, parallel training, throughput benchmarks, auto-evaluation, and grid-search attempts. The local checkout has **no Git commits** for these edits. |
| 2025-11-18 | `0deeb8e`, `08b76a0` | Course repository starts from the imported baseline and adds `optimal_d.py`; early score was described as about 200. |
| 2025-11-19 | `f74347d`, related branch commits | Better driving with sharp-turn weakness; hyperparameter and turn handling explorations. |
| 2025-11-20 | `27ae200`, `75a8079` | Road ROI work and parameter-search variant. Other same-day branch commits are visible in the original course repository. |
| 2025-11-21 | `0818481`, `d09138b`, `e865588`, `4ebd7b3` | CNN/parameter revision, run directories and CSV/TensorBoard/video workflow, compact replay, checkpoint-driven demo. |
| 2025-11-22 | `5283156` commit date | Off-track reward change. This commit's author date is Nov 20; its **commit date** is Nov 22. |
| 2025-11-23 | `e09a2a6`, `0822636`, `64da20e`, `3fff80f` | Action mapping experiment and revert, lighter off-track penalty, committed experiment material, final merge. |

The original November main line has 18 reachable commits before the publication cleanup. It spans November 18–23, not a month. The October files support an earlier work phase, but no evidence places development in mid-October. We did not backdate, synthesize, or reattribute commits to lengthen the timeline.

The initial `car_racing_TD3.py` in the November repository is about 96% text-similar to the 2023 upstream file. The new package and historical scripts should therefore be presented as modifications and experiments built on an open-source implementation. The historical course commits have several authors; individual contributions should follow those authorship records and any course documentation.
