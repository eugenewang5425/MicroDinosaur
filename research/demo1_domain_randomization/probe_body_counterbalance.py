"""Test upper-body assistance during stopping without changing the walking prefix."""

import argparse
from pathlib import Path

import numpy as np

import continue_training as base
from train_foot_capture import FootCapture
from train_stop import braking_score


class BodyCounterbalance(FootCapture):
    gain = 0.

    def __init__(self, bench, study):
        super().__init__(bench, study)
        names = list(bench.box.plant.names)
        self.axes = [names.index(name) for name in ("tail_pitch", "tail_yaw", "neck_pitch", "head_pitch")]

    def __call__(self, env, proposal, active):
        target = super().__call__(env, proposal, active)
        if (self.gain == 0. or active or self.box.plan["commands"]["forward"][0] > 0.
                or env.d.time < 2.4 - base.EPSILON):
            return target
        packet = env.selected
        gyro = packet["gyro"]
        gravity = packet["gravity_estimate"]
        pitch = np.clip((gravity[0] + .09 * gyro[1]) / .15, -1., 1.)
        roll = np.clip((gravity[1] - .09 * gyro[0]) / .15, -1., 1.)
        motion = np.clip(np.linalg.norm(gyro[:2]) / .8, 0., 1.)
        correction = self.gain * motion * np.array([pitch, roll, -.7 * pitch, -.4 * pitch])
        output = target.copy()
        output[self.axes] += correction
        return np.clip(output, env.p.bounds[:, 0], env.p.bounds[:, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("study", "output", "pilot", "reference"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    args = parser.parse_args()
    for option in ("study", "output", "pilot", "reference"):
        setattr(args, option, getattr(args, option).resolve())
    reference = base.read(args.reference)
    profile = np.array(reference["validation"][0]["weights"])
    base.StopResidual = BodyCounterbalance
    base.score = braking_score
    runner = base.Campaign(args)
    cases = base.paired_suite(908587, (4.,))
    variants = []
    try:
        for i, gain in enumerate((0., .10, -.10)):
            runner.stop.gain = gain
            results = runner.evaluate(profile, cases, "body_counterbalance", i, args.output)
            variants.append(dict(gain_rad=gain, summary=base.summary(results), cases=results))
            base.replace_json(args.output / "report.json", {
                "reference_sha256": base.sha(args.reference), "variants": variants,
            })
    except base.StopRequested:
        runner.heartbeat("stopped")
        return
    runner.heartbeat("completed")


if __name__ == "__main__":
    main()
