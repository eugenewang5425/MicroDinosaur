#!/usr/bin/env python3
"""Queue the three new roller policies, one after another, with ONNX export.

Run this when the GPU is free (DuckRex finished / paused). Each task trains,
then its latest checkpoint is exported to ONNX and validated. A failing task
does not stop the queue — its error is logged and the next one starts.

    uv run python scripts/train_roller_batch.py                # all three
    uv run python scripts/train_roller_batch.py --only standup  # one of them
    uv run python scripts/train_roller_batch.py --iters 15000 --envs 2048

Order matters: standup first (it closes the "fell down, can't get up" gap),
then sitstand, then figure-8 (the two new skills on top of a working glide).
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (key, task_id, experiment_name, onnx_name, note)
# stride first: it is the corrected skating push (the base Velocity-Rollers
# checkpoint came out walk-like and lopsided — see the stride env docstring).
QUEUE = [
    ("stride", "Mjlab-Stride-Flat-MicroDuck-Rollers", "stride_rollers",
     "microduck_stride.onnx", "轮滑蹬地（侧向推蹬 + 左右交替）"),
    ("standup", "Mjlab-RollerStandUp-Flat-MicroDuck", "velocity_rollers",
     "microduck_roller_standup.onnx", "四方向起身（前/后/左/右）"),
    ("sitstand", "Mjlab-SitStand-Flat-MicroDuck-Rollers", "roller_sitstand",
     "microduck_roller_sitstand.onnx", "轮上坐站"),
    ("figure8", "Mjlab-Figure8-Flat-MicroDuck-Rollers", "figure8_rollers",
     "microduck_roller_figure8.onnx", "8 字滑行"),
]


def latest_checkpoint(experiment: str) -> Path | None:
    """Newest model_*.pt under logs/rsl_rl/<experiment>/<run>/, version-sorted."""
    root = REPO / "logs" / "rsl_rl" / experiment
    if not root.exists():
        return None
    runs = sorted((p for p in root.iterdir() if p.is_dir()),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        return None
    ckpts = list(runs[-1].glob("model_*.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(re.search(r"model_(\d+)", p.name).group(1)))


def train(task_id: str, envs: int, iters: int) -> int:
    log = REPO / f"train_{task_id}.log"
    print(f"\n{'=' * 70}\n[队列] 训练 {task_id}  ({envs} envs × {iters} iters)\n{'=' * 70}", flush=True)
    env = {**os.environ, "WANDB_MODE": os.environ.get("WANDB_MODE", "offline")}
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.run(
            ["uv", "run", "train", task_id,
             "--env.scene.num-envs", str(envs),
             "--agent.max_iterations", str(iters)],
            cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, env=env,
        )
    print(f"[队列] {task_id} 结束, exit={proc.returncode}, 日志 {log.name}", flush=True)
    return proc.returncode


def export(task_id: str, experiment: str, onnx_name: str) -> int:
    ckpt = latest_checkpoint(experiment)
    if ckpt is None:
        print(f"[队列] 找不到 {experiment} 的检查点，跳过导出", flush=True)
        return 1
    print(f"[队列] 导出 {ckpt.name} → {onnx_name}", flush=True)
    proc = subprocess.run(
        ["uv", "run", "scripts/export.py", task_id,
         "--checkpoint-file", str(ckpt), "--onnx-file", onnx_name],
        cwd=REPO,
    )
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=15_000)
    ap.add_argument("--only", choices=[q[0] for q in QUEUE], default=None)
    args = ap.parse_args()

    todo = [q for q in QUEUE if args.only in (None, q[0])]
    t0 = time.time()
    results = []
    for key, task_id, experiment, onnx_name, note in todo:
        print(f"\n[队列] === {note} ===", flush=True)
        rc_train = train(task_id, args.envs, args.iters)
        rc_export = export(task_id, experiment, onnx_name) if rc_train == 0 else 1
        results.append((key, task_id, rc_train, rc_export))

    print(f"\n{'=' * 70}\n[队列] 全部完成, 用时 {time.time() - t0:.0f}s\n{'=' * 70}")
    for key, task_id, rc_train, rc_export in results:
        print(f"  {key:<9} 训练 {'OK ' if rc_train == 0 else 'FAIL'}  导出 "
              f"{'OK' if rc_export == 0 else 'FAIL'}   {task_id}")
    return 0 if all(r[2] == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
