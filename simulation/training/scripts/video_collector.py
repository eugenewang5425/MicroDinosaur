"""Copy new training clips into one easy-to-open folder.

mjlab writes clips to <run_dir>/videos/train/rl-video-step-<N>.mp4, deep under
logs/rsl_rl/<experiment>/<timestamp>/.  This watcher mirrors any new clip into
--out with a run-prefixed name, so there is always one place to look.

Usage (runs alongside training; harmless to leave running):
  uv run scripts/video_collector.py --out ../../local_runs/train_videos
"""

import argparse
import glob
import os
import shutil
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="velocity_microdinosaur")
    ap.add_argument("--out", required=True)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--max-minutes", type=int, default=1440)
    args = ap.parse_args()

    root = os.path.join("logs", "rsl_rl", args.experiment)
    os.makedirs(args.out, exist_ok=True)
    seen = set()
    t0 = time.time()
    print(f"[videos] watching {root} -> {args.out}  (Ctrl+C to stop)")
    while (time.time() - t0) < args.max_minutes * 60:
        for f in glob.glob(os.path.join(root, "*", "videos", "train", "*.mp4")):
            run = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f))))
            run = run.replace("_velocity_microdinosaur", "")
            dst = os.path.join(args.out, f"{run}__{os.path.basename(f)}")
            if dst in seen or os.path.exists(dst):
                seen.add(dst)
                continue
            try:
                shutil.copy2(f, dst)
                seen.add(dst)
                print(f"[videos] copied -> {os.path.basename(dst)}  "
                      f"({os.path.getsize(dst) // 1024} KB)")
            except (OSError, shutil.Error) as exc:
                print("[videos] copy failed:", exc)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
