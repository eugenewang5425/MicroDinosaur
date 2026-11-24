"""Test a persistent two-foot horizontal reference using encoder/IMU odometry."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

import continue_training as base
from train_foot_capture import FootCapture
from train_stop import braking_score


class AnchorHold(FootCapture):
    """Compare both inferred foot positions to a retained stance reference."""

    gain = .4

    def reset(self, env):
        super().reset(env)
        self.old_xy = None
        self.old_near_floor = None
        self.sample_time = None
        self.reference_xy = None
        self.body_xy = np.zeros(2)
        self.both_confidence = 0.

    def __call__(self, env, proposal, active):
        target = super().__call__(env, proposal, active)
        if self.box.plan["commands"]["forward"][0] > 0. or active:
            return target
        if env.d.time < 2.4 - base.EPSILON or not np.any(self.coefficients):
            return target
        data = self.kinematics.data
        packet = env.selected
        data.qpos[env.p.qa] = packet["encoder_q"]
        mujoco.mj_kinematics(self.model, data)
        xy = data.geom_xpos[self.soles, :2].copy()
        height = self.lowest_points(data)[:, 2]
        near_floor = np.clip(1. - (height - min(height)) / .003, 0., 1.)
        timestamp = packet["t_s"]
        if self.sample_time is None or timestamp > self.sample_time + base.EPSILON:
            if self.old_xy is not None:
                common = np.minimum(near_floor, self.old_near_floor)
                if common.max() > .4:
                    movement = np.average(self.old_xy - xy, axis=0, weights=common)
                    self.body_xy = np.clip(self.body_xy + movement, -.035, .035)
            self.both_confidence += .35 * (near_floor.min() - self.both_confidence)
            self.old_xy = xy
            self.old_near_floor = near_floor
            self.sample_time = timestamp
        if self.reference_xy is None and self.both_confidence > .75:
            self.reference_xy = xy + self.body_xy
        if self.reference_xy is None or self.both_confidence <= .4:
            return target

        data.qpos[env.p.qa] = target
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_comPos(self.model, data)
        current_xy = data.geom_xpos[self.soles, :2] + self.body_xy
        result = target.copy()
        for side, geom in enumerate(self.soles):
            change = self.gain * self.both_confidence * (self.reference_xy[side] - current_xy[side])
            step = np.r_[np.clip(change, -.004, .004), 0.]
            mujoco.mj_jacGeom(self.model, data, self.jac, None, geom)
            j = self.jac[:, self.velocity_ids[side]]
            adjust = j.T @ np.linalg.solve(j @ j.T + self.regularizer, step)
            axes = self.legs[side]
            result[axes] = proposal[axes] + np.clip(target[axes] - proposal[axes] + adjust, -.17, .17)
        return np.clip(result, env.p.bounds[:, 0], env.p.bounds[:, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("study", "output", "pilot", "reference"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    args = parser.parse_args()
    for option in ("study", "output", "pilot", "reference"):
        setattr(args, option, getattr(args, option).resolve())
    reference = base.read(args.reference)
    weights = np.array(reference["validation"][0]["weights"])
    base.StopResidual = FootCapture
    base.score = braking_score
    campaign = base.Campaign(args)
    suite = base.paired_suite(base.VALIDATION_SEED, (4.,))
    results = []
    try:
        for i, gain in enumerate((0., .3, .6)):
            if gain:
                campaign.stop = AnchorHold(campaign.box, args.study)
                campaign.stop.gain = gain
            cases = campaign.evaluate(weights, suite, "anchor_hold", i, args.output)
            results.append(dict(gain=gain, summary=base.summary(cases), cases=cases))
            base.replace_json(args.output / "report.json", {
                "reference_report_sha256": base.sha(args.reference), "cases": results,
            })
    except base.StopRequested:
        campaign.heartbeat("stopped")
        return
    campaign.heartbeat("completed")


if __name__ == "__main__":
    main()
