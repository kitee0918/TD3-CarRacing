#!/usr/bin/env python3
# file: td3_gridsearch_strict.py
"""
Accuracy-first grid search for TD3 CarRacing.

Key features
- Measure ONLY training-phase fps (step >= TRAIN_AFTER + LOG_EVERY).
- Collect multiple fps points per run, early-stop when collected.
- Repeat each combo multiple times; aggregate with trimmed-mean (robust).
- Robust UTF-8 decoding on Windows; tqdm disabled; CUDA lazy loading.
- Writes per-run logs + CSV; prints copy-paste commands.

Usage (recommended):
  python td3_gridsearch_strict.py --script td3_car_racing_train_fast.py \
    --num_envs 8,10 --batch 768 --action_repeat 2,3 --ups 1,2 \
    --train_after 3000 --until_step 6500 --collect_n 6 --warmup 2 \
    --repeat 2 --log_every 128 --max_wall 240
"""
import argparse, csv, os, re, shlex, subprocess, sys, time, math
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import List, Tuple

STEP_RE      = re.compile(r"step\s*=\s*(\d+)")
FPS_RE       = re.compile(r"fps~\s*([0-9.]+)")
FPS_ENV_RE   = re.compile(r"fps_env~\s*([0-9.]+)")  # optional

def parse_list(v: str) -> List[int]:
    v = (v or "").strip()
    if not v: return []
    if "," in v: return [int(x) for x in v.split(",") if x.strip()]
    if "-" in v:
        rng, *step = v.split("x")
        lo, hi = [int(x) for x in rng.split("-")]
        st = int(step[0]) if step else 1
        return list(range(lo, hi + 1, st))
    return [int(v)]

@dataclass
class RunCfg:
    ne: int; bs: int; ar: int; ups: int

@dataclass
class SingleRun:
    combo: RunCfg
    seed: int
    wall_sec: float
    n_points: int
    fps_points: List[float]
    fps_env_points: List[float]
    return_code: int
    workdir: str
    notes: str = ""

@dataclass
class ComboAgg:
    ne: int; bs: int; ar: int; ups: int
    repeats: int
    n_points: int
    fps_trim_mean: float
    fps_median: float
    fps_p95: float
    fps_max: float
    fps_std: float
    fps_env_mean: float
    wall_mean: float

def trimmed_mean(vals: List[float], trim_ratio: float = 0.15) -> float:
    if not vals: return 0.0
    vals_sorted = sorted(vals)
    k = int(math.floor(trim_ratio * len(vals_sorted)))
    core = vals_sorted[k: len(vals_sorted)-k] if (len(vals_sorted) - 2*k) > 0 else vals_sorted
    return sum(core) / len(core)

def p95(vals: List[float]) -> float:
    if not vals: return 0.0
    s = sorted(vals)
    idx = int(0.95 * (len(s)-1))
    return s[idx]

def mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals: return 0.0, 0.0
    m = sum(vals)/len(vals)
    var = sum((x-m)*(x-m) for x in vals) / max(1, len(vals)-1)
    return m, math.sqrt(var)

def run_once(script: Path, python_exe: str, cfg: RunCfg, seed: int,
             train_after: int, until_step: int, log_every: int,
             collect_n: int, warmup: int, max_wall: int,
             base_env: dict, workdir: Path) -> SingleRun:
    env = os.environ.copy(); env.update(base_env or {})
    env["NUM_ENVS"] = str(cfg.ne)
    env["TRAIN_BATCH_SIZE"] = str(cfg.bs)
    env["ACTION_REPEAT"] = str(cfg.ar)
    env["UPS"] = str(cfg.ups)
    env["TRAIN_AFTER"] = str(train_after)
    env["LOG_EVERY"] = str(log_every)
    env.setdefault("START_RANDOM_STEPS","1000")
    env.setdefault("TRAIN_FREQ","1")
    env.setdefault("POLICY_DELAY","2")
    env.setdefault("REPLAY_CAP","60000")
    env["AUTO_EVAL_EVERY"]="0"; env["SAVE_EVERY"]="0"

    # robust console + Windows friendliness
    env.setdefault("PYTHONUTF8","1")
    env.setdefault("PYTHONIOENCODING","utf-8")
    env.setdefault("DISABLE_TQDM","1")
    env.setdefault("CUDA_MODULE_LOADING","LAZY")
    env["SEED"] = str(seed)
    env["PYTHONHASHSEED"] = str(seed)

    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / f"run_seed{seed}.log"

    cmd = [python_exe, str(script)]
    print(f"\n=== RUN: {cfg} (seed={seed}) ===")
    print(f"CWD={workdir}")
    print("CMD=", " ".join(shlex.quote(x) for x in cmd))
    print("ENV=", {k: env[k] for k in ["NUM_ENVS","TRAIN_BATCH_SIZE","ACTION_REPEAT","UPS","TRAIN_AFTER","LOG_EVERY","SEED"]})

    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=str(workdir), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1, universal_newlines=True
    )

    fps_pts: List[float] = []
    fps_env_pts: List[float] = []
    last_step = 0
    min_train_step = max(train_after, 0) + log_every  # first stable training point
    notes = ""
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
            while True:
                if max_wall and (time.time()-t0) > max_wall:
                    notes = (notes + " wall_timeout").strip(); break
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None: break
                    time.sleep(0.02); continue
                lf.write(line); sys.stdout.write(line)

                m_step = STEP_RE.search(line)
                if m_step:
                    last_step = int(m_step.group(1))
                    if until_step and last_step >= until_step:
                        notes = (notes + f" until_step({until_step})").strip(); break

                m_fps = FPS_RE.search(line)
                if m_fps and last_step >= min_train_step:
                    try:
                        fps_val = float(m_fps.group(1))
                        fps_pts.append(fps_val)
                        if collect_n and len(fps_pts) >= collect_n:
                            notes = (notes + f" early_stop({collect_n} fps)").strip(); break
                    except ValueError:
                        pass

                m_fe = FPS_ENV_RE.search(line)
                if m_fe:
                    try: fps_env_pts.append(float(m_fe.group(1)))
                    except ValueError: pass
    except Exception as e:
        notes = f"read_err={e!r}"

    # terminate process
    try:
        proc.terminate(); rc = proc.wait(timeout=5)
    except Exception:
        proc.kill(); rc = proc.returncode if proc.returncode is not None else -9

    wall = time.time() - t0
    return SingleRun(cfg, seed, wall, max(0, len(fps_pts) - warmup), fps_pts[warmup:], fps_env_pts[warmup:], rc, str(workdir), notes)

def aggregate_combo(runs: List[SingleRun]) -> ComboAgg:
    assert runs
    fps_all = [x for r in runs for x in r.fps_points]
    fps_env_all = [x for r in runs for x in r.fps_env_points]
    fps_med = (sorted(fps_all)[len(fps_all)//2] if fps_all else 0.0)
    fps_trim = trimmed_mean(fps_all, 0.15) if fps_all else 0.0
    fps_p95v = p95(fps_all) if fps_all else 0.0
    fps_maxv = max(fps_all) if fps_all else 0.0
    _, fps_stdv = mean_std(fps_all)
    fps_env_mean = (sum(fps_env_all)/len(fps_env_all)) if fps_env_all else 0.0
    wall_mean = sum(r.wall_sec for r in runs) / len(runs)
    c = runs[0].combo
    return ComboAgg(c.ne, c.bs, c.ar, c.ups, len(runs), len(fps_all), fps_trim, fps_med, fps_p95v, fps_maxv, fps_stdv, fps_env_mean, wall_mean)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", type=str, default="td3_car_racing_train_fast.py")
    ap.add_argument("--num_envs", type=str, default="8,10")
    ap.add_argument("--batch", type=str, default="768")
    ap.add_argument("--action_repeat", type=str, default="2,3")
    ap.add_argument("--ups", type=str, default="1,2")
    ap.add_argument("--train_after", type=int, default=3000)
    ap.add_argument("--until_step", type=int, default=6500)
    ap.add_argument("--log_every", type=int, default=128)
    ap.add_argument("--collect_n", type=int, default=6, help="number of training fps points per run")
    ap.add_argument("--warmup", type=int, default=2, help="skip first N training fps points per run")
    ap.add_argument("--repeat", type=int, default=2, help="repeat runs per combo")
    ap.add_argument("--max_wall", type=int, default=240, help="wall-clock cap per single run (s)")
    ap.add_argument("--root", type=str, default="./grid_runs_strict")
    args = ap.parse_args()

    script = Path(args.script).resolve()
    if not script.exists():
        print(f"ERROR: not found: {script}", file=sys.stderr); sys.exit(2)

    ne_list  = parse_list(args.num_envs)
    bs_list  = parse_list(args.batch)
    ar_list  = parse_list(args.action_repeat)
    ups_list = parse_list(args.ups)
    combos   = [RunCfg(ne, bs, ar, up) for (ne, bs, ar, up) in product(ne_list, bs_list, ar_list, ups_list)]

    root = Path(args.root).resolve(); root.mkdir(parents=True, exist_ok=True)

    # Per-run results
    per_runs: List[SingleRun] = []
    # Aggregated per-combo
    per_combo: List[ComboAgg] = []

    base_env = {}  # extra envs injected across runs if needed
    seed0 = 1234

    for cfg in combos:
        runs = []
        for i in range(args.repeat):
            run_dir = root / f"ne{cfg.ne}_bs{cfg.bs}_ar{cfg.ar}_ups{cfg.ups}" / f"rep{i+1}"
            one = run_once(script, sys.executable, cfg, seed0 + i,
                           args.train_after, args.until_step, args.log_every,
                           args.collect_n, args.warmup, args.max_wall,
                           base_env, run_dir)
            per_runs.append(one)
            runs.append(one)
        agg = aggregate_combo(runs)
        per_combo.append(agg)

    # Save CSVs
    csv_runs = root / "grid_single_runs.csv"
    with open(csv_runs, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(per_runs[0]).keys()) if per_runs else
                                ["combo","seed","wall_sec","n_points","fps_points","fps_env_points","return_code","workdir","notes"])
        writer.writeheader()
        for r in per_runs:
            row = asdict(r).copy()
            row["combo"] = f"ne{r.combo.ne}-bs{r.combo.bs}-ar{r.combo.ar}-ups{r.combo.ups}"
            writer.writerow(row)

    csv_combo = root / "grid_results.csv"
    with open(csv_combo, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(per_combo[0]).keys()) if per_combo else
                                ["ne","bs","ar","ups","repeats","n_points","fps_trim_mean","fps_median","fps_p95","fps_max","fps_std","fps_env_mean","wall_mean"])
        writer.writeheader()
        for a in per_combo:
            writer.writerow(asdict(a))

    # Ranking by trimmed mean
    ranked = sorted(per_combo, key=lambda a: (a.fps_trim_mean, a.fps_p95, a.fps_max), reverse=True)
    print("\n===== Ranked by training fps (trimmed-mean) =====")
    for a in ranked:
        print(f"NE={a.ne:>2} BS={a.bs:>3} AR={a.ar} UPS={a.ups} | "
              f"trim={a.fps_trim_mean:7.1f} med={a.fps_median:7.1f} p95={a.fps_p95:7.1f} max={a.fps_max:7.1f} "
              f"| n={a.n_points:2d} rep={a.repeats} std={a.fps_std:6.1f} | wall~{a.wall_mean:5.1f}s")

    if ranked:
        best = ranked[0]
        # Prefer AR<=2 if within 95% of best trimmed-mean
        balanced = [a for a in ranked if a.ar <= 2 and a.fps_trim_mean >= 0.95*best.fps_trim_mean]
        rec = balanced[0] if balanced else best
        rec_type = "balanced(AR<=2, >=95% best)" if balanced else "max_fps"

        ps = (f"$env:NUM_ENVS={rec.ne}; $env:TRAIN_BATCH_SIZE={rec.bs}; $env:ACTION_REPEAT={rec.ar}; $env:UPS={rec.ups}; "
              f"$env:PYTHONUTF8=1; $env:PYTHONIOENCODING='utf-8'; $env:CUDA_MODULE_LOADING='LAZY'; "
              f"$env:LOG_EVERY=256; python {script.name}")
        sh = (f"NUM_ENVS={rec.ne} TRAIN_BATCH_SIZE={rec.bs} ACTION_REPEAT={rec.ar} UPS={rec.ups} "
              f"PYTHONUTF8=1 PYTHONIOENCODING=utf-8 CUDA_MODULE_LOADING=LAZY LOG_EVERY=256 "
              f"python {script.name}")

        print("\n[recommendation]", rec_type)
        print("PowerShell:", ps)
        print("bash:", sh)

    print(f"\n[saved] per-run → {csv_runs}")
    print(f"[saved] per-combo → {csv_combo}")

if __name__ == "__main__":
    main()
