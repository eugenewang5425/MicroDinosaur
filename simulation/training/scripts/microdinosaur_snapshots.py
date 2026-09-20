"""Record one demo clip per checkpoint, so iterations can be compared side by side.

Same fixed command sequence every time (stand -> forward -> turn -> gestures),
one mp4 per checkpoint, plus a printed metric line so the clips can be compared
numerically as well as visually.

Usage (from the repo root):
  uv run scripts/microdinosaur_snapshots.py --iters 14000,16000,18000 \
      --out D:/项目/miro_dinosaur/renders/snap
  uv run scripts/microdinosaur_snapshots.py --latest 3 --out ...   # newest 3
"""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import microdinosaur_play as play  # noqa: E402


SEQ = [("stand", 0.0, 0.0, 0.0, 60),
       ("forward", 0.35, 0.0, 0.0, 150),
       ("turn", 0.0, 0.0, 0.8, 120),
       ("gesture", 0.0, 0.0, 0.0, 90)]


def find_ckpts(experiment, iters):
    found = {}
    for run in glob.glob(os.path.join("logs", "rsl_rl", experiment, "*")):
        for f in glob.glob(os.path.join(run, "model_*.pt")):
            n = int(os.path.basename(f).split("_")[1].split(".")[0])
            found[n] = f
    if not iters:
        return [found[k] for k in sorted(found)[-3:]]
    out = []
    for want in iters:
        best = min(found, key=lambda k: abs(k - want)) if found else None
        if best is not None:
            out.append(found[best])
    return out


def record(ckpt, out_path, model, kp, kv, iters, size):
    task = "Mjlab-Velocity-Flat-MicroDinosaur"
    tmp = os.path.join(os.path.dirname(out_path), "_tmp_policy.onnx")
    print(f"[snap] exporting {os.path.basename(ckpt)} ...")
    subprocess.run(["uv", "run", "scripts/export.py", task,
                    "--checkpoint-file", ckpt, "--onnx-file", tmp], check=False)
    sim = play.Sim(play.DEFAULT_XML[model], tmp, kp=kp, kv=kv, iters=iters)
    cam = play.mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 135, -18, 0.9
    cmd = np.zeros(18, dtype=np.float32)
    frames, stats = [], {}
    import time
    t = 0.0
    for name, vx, vy, wz, n in SEQ:
        vs, as_ = [], []
        for _ in range(n):
            t += 0.02
            cmd[:] = 0.0
            cmd[0], cmd[1], cmd[2] = vx, vy, wz
            if name == "gesture":
                cmd[13] = cmd[14] = 0.9          # arms up
                cmd[15] = 0.4 * np.sin(2 * np.pi * 0.8 * t)
                cmd[17] = 0.3
            sim.step(cmd, 1.0)
            frames.append(sim.render(cam, size))
            vs.append(float(sim.data.qvel[0]))
        stats[name] = float(np.mean(vs))
    import imageio
    imageio.mimsave(out_path, frames, fps=50)
    print(f"[snap] wrote {out_path}  ({len(frames)} frames)  "
          f"forward={stats['forward']:+.3f} m/s  "
          f"fwd_z_ok={'yes' if stats['forward'] > 0.15 else 'no'}")
    try:
        os.remove(tmp)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", default=None, help="comma-separated iteration numbers")
    ap.add_argument("--latest", type=int, default=0, help="take the newest N checkpoints")
    ap.add_argument("--experiment", default="velocity_microdinosaur")
    ap.add_argument("--model", default="gc", choices=list(play.DEFAULT_XML))
    ap.add_argument("--out", required=True, help="output directory (ASCII path)")
    ap.add_argument("--tag", default="p2")
    ap.add_argument("--kp", type=float, default=7.0)
    ap.add_argument("--kv", type=float, default=0.4)
    ap.add_argument("--iters-solver", type=int, default=20)
    ap.add_argument("--size", default="960x720")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    W, H = (int(x) for x in args.size.lower().split("x"))

    iters = [int(x) for x in args.iters.split(",")] if args.iters else []
    ckpts = find_ckpts(args.experiment, iters)
    if not ckpts:
        raise SystemExit("no checkpoints found")
    for ck in ckpts:
        n = int(os.path.basename(ck).split("_")[1].split(".")[0])
        out = os.path.join(args.out, f"{args.tag}_iter{n}.mp4")
        record(ck, out, args.model, args.kp, args.kv, args.iters_solver, (W, H))


if __name__ == "__main__":
    main()
