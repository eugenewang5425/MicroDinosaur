"""Learn stopping with a lower-speed curriculum and unchanged full-speed validation."""

import argparse
from pathlib import Path

import numpy as np

import continue_training as base
from probe_speed_curriculum import SpeedWorkbench
from train_foot_capture import FootCapture
from train_stop import StopCampaign, braking_score


class SpeedCurriculum(StopCampaign):
    def __init__(self, args):
        args.output.mkdir(parents=True, exist_ok=True)
        seed = args.output / "curriculum_seed.json"
        identity = {
            "method": "vary_approach_speed",
            "sha256": base.sha(Path(__file__)),
            "speed_probe_sha256": base.sha(Path(__file__).with_name("probe_speed_curriculum.py")),
            "seed_sha256": base.sha(seed),
        }
        stamp = args.output / "speed_variant.json"
        if stamp.exists() and base.read(stamp) != identity:
            raise ValueError("Speed curriculum source or seed changed; use a new run directory")
        if not stamp.exists():
            base.replace_json(stamp, identity)
        super().__init__(args)
        if self.state["generation"] == 0:
            seeded = np.array(base.read(seed))
            if seeded.shape != (base.DIMENSIONS + 16,) or np.any(seeded[:base.DIMENSIONS]):
                raise ValueError("Expected a fixed-walking-prefix seed with 16 braking gains")
            self.state.update(center=seeded.tolist(), incumbent=seeded.tolist())
            base.replace_json(self.state_path, self.state)

    def evaluate(self, weights, cases, kind, candidate, trace_folder=None):
        generation = self.state["generation"]
        if kind == "stop_training":
            if generation < 8:
                peak = .09
            elif generation < 24:
                peak = (.09, .12, .15, .20)[generation % 4]
            else:
                peak = (.12, .15, .18, .20, .20)[generation % 5]
        else:
            peak = .20
        self.box.command_peak = peak
        self.status["forward_command_peak"] = peak
        return super().evaluate(weights, cases, kind, candidate, trace_folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("study", "output", "pilot"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    for name in ("study", "output", "pilot"):
        setattr(args, name, getattr(args, name).resolve())
    if (args.output / "STOP").exists():
        return
    base.Workbench = SpeedWorkbench
    base.StopResidual = FootCapture
    base.score = braking_score
    try:
        SpeedCurriculum(args).run()
    except base.StopRequested:
        return


if __name__ == "__main__":
    main()
