"""Train walking readiness against paired release phases and integration steps."""

import argparse
from pathlib import Path

import numpy as np

import continue_training as base
from demo1_domain_randomization import train as simulator
from train_foot_capture import FootCapture
from train_ready_to_stop import ReadyCampaign, ready_metrics, ready_score


class PairedPhaseCampaign(ReadyCampaign):
    def __init__(self, args):
        args.output.mkdir(parents=True, exist_ok=True)
        mixed = args.output / "mixed_seed.json"
        marker = args.output / "paired_phase_variant.json"
        identity = dict(method="paired_phases_minimum_score", sha256=base.sha(Path(__file__)),
                        mixed_seed_sha256=base.sha(mixed))
        if marker.exists() and base.read(marker) != identity:
            raise ValueError("Paired-phase training source or seed changed")
        if not marker.exists():
            base.replace_json(marker, identity)
        super().__init__(args)
        if self.state["generation"] == 0:
            start = np.array(base.read(mixed))
            if start.shape != (base.TOTAL,):
                raise ValueError("Expected a 32-parameter mixed seed")
            self.state.update(incumbent=start.tolist(), center=start.tolist())
            base.replace_json(self.state_path, self.state)

    def evaluate(self, weights, cases, kind, candidate, trace_folder=None):
        if kind == "gait_readiness":
            generation = self.state["generation"]
            # Share the input suite across a complete generation and retain one
            # batch for four generations to make rank changes attributable.
            block = generation // 4
            rng = np.random.default_rng(base.SEED + 13371 + block * 23)
            early = (3.54, 3.72, 3.88, 4.12, 4.28, 4.44)[block % 6]
            first = base.draw_domain("nominal" if block < 2 else "mild", rng)
            second = base.draw_domain("mild" if block < 4 else "moderate", rng)
            cases = [(domain, release, dt, 4.8)
                     for (domain, release) in ((first, 4.), (second, early))
                     for dt in (base.BUS_SLOT, base.BUS_SLOT / 2)]
        return super().evaluate(weights, cases, kind, candidate, trace_folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("study", "output", "pilot"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    for field in ("study", "output", "pilot"):
        setattr(args, field, getattr(args, field).resolve())
    if (args.output / "STOP").exists():
        return
    base.StopResidual = FootCapture
    base.score = ready_score
    simulator.metrics = ready_metrics
    try:
        PairedPhaseCampaign(args).run()
    except base.StopRequested:
        return


if __name__ == "__main__":
    main()
