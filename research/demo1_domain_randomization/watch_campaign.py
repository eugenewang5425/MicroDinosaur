"""Watch several frozen-input actuator workers in a single lightweight process."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


TRAINERS = {
    "overnight_v2": "continue_training.py",
    "overnight_v3": "train_stop.py",
    "overnight_v4": "train_foot_capture.py",
    "overnight_v5": "train_speed_curriculum.py",
    "overnight_v6": "train_ready_to_stop.py",
    "overnight_v7": "train_paired_phases.py",
    "overnight_v8": "train_actuator_lead.py",
    "overnight_v9": "train_nominal_first.py",
}
SOURCE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def normal(path):
    return os.path.normcase(os.path.normpath(path))


def owns_worker(folder, trainer):
    try:
        worker = psutil.Process(read(folder / "status.json")["pid"])
        command = {normal(part) for part in worker.cmdline()}
        return normal(str(folder)) in command and normal(str(trainer)) in command
    except (FileNotFoundError, psutil.NoSuchProcess):
        return False
    except (PermissionError, psutil.AccessDenied, json.JSONDecodeError):
        return True  # Avoid starting a duplicate during an unverifiable read.


def log(destination, message):
    with destination.open("a", encoding="utf-8") as file:
        file.write(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}\n")


def start(args, out, trainer):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    command = [
        str(args.python), "-u", str(SOURCE / "run_with_source_lock.py"),
        "--entry", str(trainer), "--source-lock", str(args.source_lock), "--study", str(args.study),
        "--output", str(out), "--pilot", str(args.pilot),
    ]
    environment = os.environ | {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1"}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with (out / f"resume_{stamp}_stdout.log").open("ab") as stdout, (out / f"resume_{stamp}_stderr.log").open("ab") as stderr:
        process = subprocess.Popen(command, cwd=args.study, env=environment, stdout=stdout, stderr=stderr,
                                   stdin=subprocess.DEVNULL, creationflags=flags)
    return process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    for key in ("campaign-root", "study", "pilot", "source-lock"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", choices=tuple(TRAINERS), required=True)
    args = parser.parse_args()
    for name in ("python", "campaign_root", "study", "pilot", "source_lock"):
        setattr(args, name, getattr(args, name).resolve())
    jobs = {args.campaign_root / name: SOURCE / TRAINERS[name] for name in dict.fromkeys(args.runs)}
    for out in jobs:
        if not (out / "contract.json").is_file():
            raise ValueError(f"Expected an existing training contract in {out.name}")
    log_path = args.campaign_root / "campaign_watcher.log"
    workers = {}
    last_start = {}
    (args.campaign_root / "campaign_watcher.pid").write_text(str(os.getpid()) + "\n", encoding="ascii")
    log(log_path, "Attached to " + ", ".join(out.name for out in jobs))
    while any(not (out / "STOP").exists() for out in jobs):
        tick = time.monotonic()
        for out, trainer in jobs.items():
            if (out / "STOP").exists() or owns_worker(out, trainer):
                continue
            pending = workers.get(out)
            if (pending is not None and pending.poll() is None) or tick - last_start.get(out, -1e5) < 75.:
                continue
            if psutil.virtual_memory().available < 4.3 * 1024**3:
                continue
            last_start[out] = tick
            try:
                workers[out] = start(args, out, trainer)
                log(log_path, f"Started {out.name} via process {workers[out].pid}")
            except OSError as error:
                log(log_path, f"Could not start {out.name}: {type(error).__name__}")
        time.sleep(15.)
    log(log_path, "Observed stop markers for all monitored runs")


if __name__ == "__main__":
    main()
