"""Measure lower-speed approach phases before adding them to the stopping curriculum."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

import continue_training as base
from train_foot_capture import FootCapture
from train_stop import braking_score


class SpeedWorkbench(base.Workbench):
    command_peak = .20

    def rollout(self, *args, **kwargs):
        advance = self.box.advance

        def cap_forward(motion=None, *options, **named):
            if motion == "forward":
                target = self.box.plan["commands"]["forward"]
                self.box.plan["commands"]["forward"] = [min(self.command_peak, target[0]), *target[1:]]
            return advance(motion, *options, **named)

        with patch.object(self.box, "advance", cap_forward):
            return super().rollout(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("study", "output", "pilot", "reference"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    args = parser.parse_args()
    for option in ("study", "output", "pilot", "reference"):
        setattr(args, option, getattr(args, option).resolve())
    reference = base.read(args.reference)
    candidate = np.array(reference["validation"][0]["weights"])
    base.Workbench = SpeedWorkbench
    base.StopResidual = FootCapture
    base.score = braking_score
    campaign = base.Campaign(args)
    cases = base.paired_suite(base.VALIDATION_SEED, (4.,))
    collected = []
    try:
        for index, peak in enumerate((.20, .09, .13, .17)):
            campaign.box.command_peak = peak
            results = campaign.evaluate(candidate, cases, "speed_probe", index, args.output)
            collected.append(dict(command_peak=peak, summary=base.summary(results), cases=results))
            base.replace_json(args.output / "report.json", {
                "reference_report_sha256": base.sha(args.reference),
                "scenarios": collected,
            })
    except base.StopRequested:
        campaign.heartbeat("stopped")
        return
    campaign.heartbeat("completed")


if __name__ == "__main__":
    main()
