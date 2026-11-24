"""Train the exact native stop first, then admit broader phases and adverse domains."""

import argparse
from pathlib import Path

import numpy as np

import continue_training as base
from demo1_domain_randomization import train as simulator
from train_foot_capture import FootCapture
from train_ready_to_stop import ReadyCampaign, ready_metrics, ready_score


class NominalFirstCampaign(ReadyCampaign):
    def __init__(self, args):
        args.output.mkdir(parents=True, exist_ok=True)
        stamp = args.output / "nominal_first_variant.json"
        identity = dict(method="physics_progression_after_pass", sha256=base.sha(Path(__file__)))
        if stamp.exists() and base.read(stamp) != identity:
            raise ValueError("Nominal-first trainer changed")
        if not stamp.exists():
            base.replace_json(stamp, identity)
        super().__init__(args)

    def stage_cases(self):
        stage = self.state.get("course_stage", 0)
        level = ("nominal", "nominal", "mild", "moderate")[stage]
        times = (4.,) if stage == 0 else (3.76, 4., 4.24)
        cohorts = []
        rng = np.random.default_rng(base.SEED + 8863 + stage)
        for release in times:
            domain = base.draw_domain(level, rng)
            cohorts.extend((domain, release, dt, 10.) for dt in (base.BUS_SLOT, base.BUS_SLOT / 2))
        return cohorts

    def evaluate(self, weights, cases, kind, candidate, trace_folder=None):
        stage = self.state.get("course_stage", 0)
        self.status["course_stage"] = stage
        if kind in ("gait_readiness", "ready_development", "frozen_development"):
            cases = self.stage_cases()
        return super().evaluate(weights, cases, kind, candidate, trace_folder)

    def generation(self):
        super().generation()
        stage = self.state.get("course_stage", 0)
        champion = self.state["best"]
        if stage == 3 or champion["metrics"]["joint"] < len(self.stage_cases()):
            return
        stage_report = dict(stage=stage, generation=champion["generation"],
                            metrics=champion["metrics"], weights=champion["weights"])
        base.replace_json(self.output / f"passed_course_{stage}.json", stage_report)
        self.state["course_stage"] = stage + 1
        self.state["incumbent"] = champion["weights"]
        self.state["center"] = champion["weights"]
        self.state["best"] = dict(generation=champion["generation"], rank=(-1, -1, -1, -1, -1000.),
                                  metrics=dict(joint=0), weights=champion["weights"])
        base.replace_json(self.state_path, self.state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("study", "output", "pilot"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    for name in ("study", "output", "pilot"):
        setattr(args, name, getattr(args, name).resolve())
    if (args.output / "STOP").exists():
        return
    base.StopResidual = FootCapture
    base.score = ready_score
    simulator.metrics = ready_metrics
    try:
        NominalFirstCampaign(args).run()
    except base.StopRequested:
        return


if __name__ == "__main__":
    main()
