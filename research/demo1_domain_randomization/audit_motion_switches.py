"""Replay direct forward-to-backward and forward-to-left switches on frozen Demo 1."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

import continue_training as base
from demo1_domain_randomization import train as simulation
from train_command_switch import HOLD_S, assess, transition_differences
from train_foot_capture import FootCapture


TURN_AT_S = 4.
RELEASE_S = 6.
DIRECTIONS = {"backward": (-.12, 0., 0.), "left": (0., .12, 0.)}


class DirectionalWorkbench(simulation.Workbench):
    direction = "backward"

    def rollout(self, *args, **kwargs):
        advance = self.box.advance

        def directional_advance(motion=None):
            if motion == "forward" and self.box.env.d.time - 2.4 >= TURN_AT_S - base.EPSILON:
                motion = self.direction
                self.box.plan["commands"][motion] = list(DIRECTIONS[motion])
            return advance(motion)

        with patch.object(self.box, "advance", directional_advance):
            return super().rollout(*args, **kwargs)


class DirectionalCapture(FootCapture):
    def __call__(self, env, proposal, active):
        return super().__call__(env, proposal, active or np.any(self.box.active_command))


def motion_data(trace, direction):
    with np.load(trace) as replay:
        t = replay["time"]
        moved = t[-1] >= RELEASE_S - base.EPSILON
        xy = replay["root"][:, :2]
        axis = np.asarray(DIRECTIONS[direction][:2]) / .12
        start = np.searchsorted(t, TURN_AT_S)
        stop = np.searchsorted(t, RELEASE_S)
        progress = float(1000. * ((xy[stop] - xy[start]) @ axis)) if moved else None
        prefix = replay["angle_rad"][replay["action_time"] < RELEASE_S - base.EPSILON]
        return progress, prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("study", "output", "pilot", "seed"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    base.Workbench = DirectionalWorkbench
    base.StopResidual = DirectionalCapture
    base.score = lambda result, *_: float(result["parking"]["passed"])
    campaign = base.Campaign(args)
    original = np.zeros(base.TOTAL)
    candidate = np.asarray(base.read(args.seed)["weights"], dtype=float)
    if candidate.shape != original.shape or np.any(candidate[:base.DIMENSIONS]) or not np.isfinite(candidate).all():
        raise ValueError("Expected the v5 seed with unmodified direction switches")
    nominal = base.draw_domain("nominal", np.random.default_rng(0))
    cases = [(nominal, RELEASE_S, dt, HOLD_S) for dt in (base.BUS_SLOT, base.BUS_SLOT / 2)]
    rows = []
    try:
        with patch.object(simulation, "metrics", assess):
            for direction in DIRECTIONS:
                campaign.box.direction = direction
                baseline = campaign.evaluate(original, cases, f"{direction}_original", 0, args.output)
                revised = campaign.evaluate(candidate, cases, f"{direction}_v5", 0, args.output)
                for i, (frozen, trained) in enumerate(zip(baseline, revised)):
                    original_progress, original_prefix = motion_data(args.output / f"{direction}_original_0_{i}.npz",
                                                                     direction)
                    v5_progress, v5_prefix = motion_data(args.output / f"{direction}_v5_0_{i}.npz", direction)
                    rows.append(dict(direction=direction, dt_s=frozen["dt_s"],
                                     original_progress_mm=original_progress, v5_progress_mm=v5_progress,
                                     complete_command_prefix_matches=bool(original_progress is not None
                                                                          and v5_progress is not None
                                                                          and np.array_equal(original_prefix,
                                                                                             v5_prefix)),
                                     original=frozen["metrics"], v5=trained["metrics"],
                                     v5_minus_original=transition_differences(frozen, trained)))
                base.replace_json(args.output / "report.json", dict(
                    forward_peak=.20, changed_at_s=TURN_AT_S, zero_at_s=RELEASE_S,
                    settled_stance_s=10., test="nominal diagnostic; assess observed switching and safe stance",
                    results=rows,
                ))
    except base.StopRequested:
        campaign.heartbeat("stopped")
        return
    campaign.heartbeat("completed")


if __name__ == "__main__":
    main()
