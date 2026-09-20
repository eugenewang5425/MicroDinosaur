"""Export an ONNX once the running training's reward reaches a threshold.

A tiny companion to train_supervisor.py: it watches the supervisor's status
lines, and the moment the instantaneous reward hits `--threshold` it exports the
newest checkpoint to `--onnx` and exits.  Used to hand over an intermediate
policy ("tell me when it reaches 160") without stopping the run.

Usage:
  uv run scripts/export_at_reward.py --task Mjlab-... --log duckrex_p2_supervisor.log \
      --onnx duckrex_p2_mid.onnx --threshold 160 --experiment velocity_microdinosaur
"""

import argparse
import glob
import os
import re
import subprocess
import time

REWARD_RE = re.compile(r"iter (\d+)\s+reward\s+([-\d.]+)\s+window-mean\s+([-\d.]+)")


def newest_ckpt(exp_dir):
    best = None
    for run in glob.glob(os.path.join(exp_dir, "*")):
        for f in glob.glob(os.path.join(run, "model_*.pt")):
            n = int(re.search(r"model_(\d+)\.pt$", f).group(1))
            if best is None or n > best[0]:
                best = (n, f)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--threshold", type=float, default=160.0)
    ap.add_argument("--experiment", default="velocity_microdinosaur")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--max-minutes", type=int, default=720)
    args = ap.parse_args()

    exp_dir = os.path.join("logs", "rsl_rl", args.experiment)
    t0 = time.time()
    last = None
    while (time.time() - t0) < args.max_minutes * 60:
        try:
            with open(args.log, encoding="utf-8", errors="ignore") as f:
                rows = REWARD_RE.findall(f.read())
        except OSError:
            rows = []
        if rows:
            it, rew, wmean = int(rows[-1][0]), float(rows[-1][1]), float(rows[-1][2])
            if last != it:
                print(f"[export-watch] iter {it}  reward {rew:.1f}  window-mean {wmean:.1f}",
                      flush=True)
                last = it
            if max(rew, wmean) >= args.threshold:
                ck = newest_ckpt(exp_dir)
                if ck:
                    print(f"[export-watch] threshold {args.threshold} reached -- "
                          f"exporting checkpoint {ck[1]}", flush=True)
                    subprocess.run(["uv", "run", "scripts/export.py", args.task,
                                    "--checkpoint-file", ck[1], "--onnx-file", args.onnx],
                                   check=False)
                    try:
                        import numpy as np
                        import onnxruntime as ort
                        s = ort.InferenceSession(args.onnx,
                                                 providers=["CPUExecutionProvider"])
                        i, o = s.get_inputs()[0], s.get_outputs()[0]
                        y = s.run(None, {i.name: np.random.randn(
                            1, i.shape[1]).astype(np.float32)})[0]
                        print(f"[export-watch] ONNX ok: {i.shape} -> {o.shape} "
                              f"nan={np.isnan(y).any()} "
                              f"size={os.path.getsize(args.onnx)}B", flush=True)
                    except Exception as exc:
                        print("[export-watch] ONNX check failed:", exc, flush=True)
                    return
        time.sleep(args.poll)
    print("[export-watch] timed out without reaching the threshold", flush=True)


if __name__ == "__main__":
    main()
