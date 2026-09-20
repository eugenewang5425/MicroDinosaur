"""Training supervisor: spawn training, watch the log, stop on convergence.

Why this exists: this session cannot create scheduled tasks, so convergence
watching used to be manual.  This script does it in-process instead:

  * spawns `uv run train <task>` (or resumes from a checkpoint)
  * polls the log: parses iteration + mean reward
  * AUTO-STOP when either
      - `--target-iter` is reached, or
      - the reward PLATEAUS: |mean(last window) - mean(prev window)| is within
        `--plateau-pct` % for `--confirm` consecutive windows, after
        `--min-iter` iterations (defaults: window 1000, 0.6 %, confirm 2)
  * AUTO-RESUME if the training process dies (crash / BSOD aftermath),
    picking the newest checkpoint of that experiment
  * then export ONNX from the final checkpoint and validate [1,81] -> [1,19]

Usage:
  uv run scripts/train_supervisor.py --task Mjlab-Velocity-Flat-MicroDinosaur \
      --envs 4096 --log duckrex_p2_train.log --onnx duckrex_p2.onnx \
      [--resume-run <run dir> --resume-ckpt model_8750.pt] \
      [--target-iter 12000] [--min-iter 2000] [--plateau-pct 0.6] [--window 1000]

Everything is printed ASCII-only (this machine's console is a GBK code page).
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time

ITER_RE = re.compile(r"Learning iteration (\d+)/")
REW_RE = re.compile(r"Mean reward:\s*([-\d.]+)")
SWEEP_RE = re.compile(r"Episode_Reward/air_time:\s*([-\d.]+)")


def parse_log(path):
    """Return (last_iter, [(iter, reward)...], last_air_time)."""
    last_iter = 0
    rewards = []
    air = None
    if not os.path.exists(path):
        return last_iter, rewards, air
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            cur = 0
            for line in f:
                m = ITER_RE.search(line)
                if m:
                    cur = int(m.group(1))
                    last_iter = cur
                    continue
                m = REW_RE.search(line)
                if m and cur:
                    rewards.append((cur, float(m.group(1))))
                    continue
                m = SWEEP_RE.search(line)
                if m:
                    air = float(m.group(1))
    except OSError:
        pass
    return last_iter, rewards, air


def mean_between(rewards, lo, hi):
    vals = [v for i, v in rewards if lo <= i <= hi]
    return sum(vals) / len(vals) if vals else None


def latest_ckpt(exp_dir, run_glob="*"):
    """Newest checkpoint by MTIME -- not by iteration number.

    Iteration numbers only increase within one run, so after a chained resume an
    older run directory can hold a HIGHER number than the run just trained:
    resuming 13500 -> 14300 while an earlier run in the same experiment dir still
    had model_15500 made the supervisor export the OLD policy (2026-09-12,
    train6).  The freshly written file is the one to ship.
    """
    runs = sorted(glob.glob(os.path.join(exp_dir, run_glob)))
    best = None
    for r in runs:
        for f in glob.glob(os.path.join(r, "model_*.pt")):
            m = os.path.getmtime(f)
            if best is None or m > best[0]:
                best = (m, f)
    return best


def kill_tree(pid):
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                   capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--log", required=True)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--experiment", default="velocity_microdinosaur",
                    help="logs/rsl_rl/<experiment> holds the run dirs")
    ap.add_argument("--resume-run", default=None)
    ap.add_argument("--resume-ckpt", default=None)
    ap.add_argument("--target-iter", type=int, default=None)
    ap.add_argument("--min-iter", type=int, default=2000)
    ap.add_argument("--plateau-pct", type=float, default=0.6)
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--confirm", type=int, default=2)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--max-restarts", type=int, default=6)
    ap.add_argument("--extra", default="", help="extra args appended to train")
    ap.add_argument("--video-every-iters", type=int, default=None,
                    help="record a training clip every N iterations (mjlab renders "
                         "1 env offscreen; clips land in <run>/videos/train/)")
    ap.add_argument("--video-frames", type=int, default=300,
                    help="frames per clip (50 Hz -> 300 frames = 6 s)")
    args = ap.parse_args()

    exp_dir = os.path.join("logs", "rsl_rl", args.experiment)
    cmd = ["uv", "run", "train", args.task, "--env.scene.num-envs", str(args.envs)]
    if args.resume_run and args.resume_ckpt:
        cmd += ["--agent.resume", "True", "--agent.load_run", args.resume_run,
                "--agent.load_checkpoint", args.resume_ckpt]
    if args.video_every_iters:
        # mjlab's --video-interval is in ENV STEPS (24 steps per iteration).
        cmd += ["--video", "True",
                "--video-interval", str(args.video_every_iters * 24),
                "--video-length", str(args.video_frames)]
    if args.extra:
        cmd += args.extra.split()

    def spawn(resume_from=None):
        """Build the command, always with exactly ONE pair of resume flags."""
        base = []
        skip = 0
        for i, tok in enumerate(cmd):
            if skip:
                skip -= 1
                continue
            if tok in ("--agent.resume", "--agent.load_run", "--agent.load_checkpoint"):
                skip = 1
                continue
            base.append(tok)
        c = base + ["--agent.resume", "True"]
        if resume_from:
            c += ["--agent.load_run", os.path.basename(os.path.dirname(resume_from[1])),
                  "--agent.load_checkpoint", os.path.basename(resume_from[1])]
        elif args.resume_run and args.resume_ckpt:
            c += ["--agent.load_run", args.resume_run,
                  "--agent.load_checkpoint", args.resume_ckpt]
        print("[supervisor] spawning:", " ".join(c), flush=True)
        logf = open(args.log, "a", encoding="utf-8", errors="replace")
        return subprocess.Popen(c, stdout=logf, stderr=subprocess.STDOUT), logf

    proc, logf = spawn()
    restarts = 0
    low_windows = []
    last_report = 0.0

    try:
        while True:
            time.sleep(args.poll)
            it, rewards, air = parse_log(args.log)

            if it and time.time() - last_report > 60:
                last_report = time.time()
                r = rewards[-1][1] if rewards else float("nan")
                aw = mean_between(rewards, it - args.window, it)
                print(f"[supervisor] iter {it}  reward {r:.1f}  "
                      f"window-mean {aw:.1f}" if aw else f"[supervisor] iter {it}",
                      flush=True)

            # --- process died: resume from the newest checkpoint of this run
            if proc.poll() is not None:
                restarts += 1
                print(f"[supervisor] training died (rc={proc.returncode}); "
                      f"restart {restarts}/{args.max_restarts}", flush=True)
                if restarts > args.max_restarts:
                    print("[supervisor] too many restarts -- giving up", flush=True)
                    break
                logf.close()
                ck = latest_ckpt(exp_dir)
                print("[supervisor] resuming from", ck, flush=True)
                proc, logf = spawn(ck)
                continue

            # --- convergence: reward plateau
            if it >= args.min_iter and rewards:
                prev_w = mean_between(rewards, it - 2 * args.window, it - args.window)
                last_w = mean_between(rewards, it - args.window, it)
                if prev_w and last_w:
                    delta = abs(last_w - prev_w) / max(abs(prev_w), 1e-6) * 100.0
                    if delta <= args.plateau_pct:
                        low_windows.append((it, last_w, delta))
                        print(f"[supervisor] plateau check {len(low_windows)}/"
                              f"{args.confirm}: delta {delta:.2f}% "
                              f"(<= {args.plateau_pct}%)", flush=True)
                        if len(low_windows) >= args.confirm:
                            print(f"[supervisor] CONVERGED at iter {it} "
                                  f"(reward ~{last_w:.1f}, air_time {air})", flush=True)
                            break
                    else:
                        low_windows.clear()

            # --- target iteration
            if args.target_iter and it >= args.target_iter:
                print(f"[supervisor] reached target iter {args.target_iter}", flush=True)
                break

            # --- if the process is gone and we still want to run, respawn
            if proc.poll() is not None:
                restarts += 1
                print(f"[supervisor] training died (rc={proc.returncode}); "
                      f"restart {restarts}/{args.max_restarts}", flush=True)
                if restarts > args.max_restarts:
                    print("[supervisor] too many restarts -- giving up", flush=True)
                    break
                logf.close()
                ck = latest_ckpt(exp_dir)
                print("[supervisor] resuming from", ck, flush=True)
                proc, logf = spawn(ck)
    finally:
        if proc.poll() is None:
            print("[supervisor] stopping training...", flush=True)
            kill_tree(proc.pid)
            time.sleep(5)

    it, rewards, air = parse_log(args.log)
    print(f"[supervisor] final: iter {it}  air_time {air}", flush=True)

    if args.onnx:
        ck = latest_ckpt(exp_dir)
        if ck:
            print(f"[supervisor] exporting ONNX from {ck[1]}", flush=True)
            subprocess.run(["uv", "run", "scripts/export.py", args.task,
                            "--checkpoint-file", ck[1], "--onnx-file", args.onnx],
                           check=False)
            try:
                import numpy as np
                import onnxruntime as ort
                s = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
                i, o = s.get_inputs()[0], s.get_outputs()[0]
                y = s.run(None, {i.name: np.random.randn(1, i.shape[1]).astype(np.float32)})[0]
                print(f"[supervisor] ONNX ok: {i.shape} -> {o.shape} "
                      f"nan={np.isnan(y).any()} size={os.path.getsize(args.onnx)}B",
                      flush=True)
            except Exception as exc:
                print("[supervisor] ONNX check failed:", exc, flush=True)
    print("[supervisor] DONE", flush=True)


if __name__ == "__main__":
    main()
