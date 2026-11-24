"""Compare stance anchoring and lower swing clearance in paired physical replay."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

import continue_training as base
from train_foot_capture import FootCapture
from train_stop import braking_score


class SupportAnchor(FootCapture):
    def reset(self, env):
        super().reset(env)
        self.anchor = None

    def __call__(self, env, proposal, active):
        capture = super().__call__(env, proposal, active)
        if self.box.plan["commands"]["forward"][0] > 0. or active:
            self.anchor = None
            return capture
        if env.d.time < 2.4 - base.EPSILON or not np.any(self.coefficients):
            return capture

        data = self.kinematics.data
        data.qpos[env.p.qa] = env.selected["encoder_q"]
        mujoco.mj_kinematics(self.model, data)
        height = self.lowest_points(data)[:, 2]
        nearest = int(np.argmin(height))
        if self.anchor is None:
            self.anchor = nearest
        # When the original support rises clear of the other sole, transfer to
        # the lower foot for recovery. During close double support guard both.
        if height[self.anchor] - height[nearest] > .004:
            self.anchor = nearest
        both_close = abs(height[0] - height[1]) < .0015
        support_sides = tuple(range(2)) if both_close else (self.anchor,)

        data.qpos[env.p.qa] = capture
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_comPos(self.model, data)
        tips = self.lowest_points(data)
        floor = min(height)
        result = capture.copy()
        for side in support_sides:
            needed = np.clip(floor - .0007 - tips[side, 2], -.012, 0.)
            if needed == 0.:
                continue
            body = int(self.model.geom_bodyid[self.soles[side]])
            mujoco.mj_jac(self.model, data, self.jac, None, tips[side], body)
            jacobian = self.jac[:, self.velocity_ids[side]]
            correction = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + self.regularizer,
                                                      np.array([0., 0., needed]))
            axes = self.legs[side]
            result[axes] = proposal[axes] + np.clip(capture[axes] - proposal[axes] + correction, -.17, .17)
        return np.clip(result, env.p.bounds[:, 0], env.p.bounds[:, 1])


class BrakeReadyWalk(FootCapture):
    """Make each step less dependent on the airborne phase at command release."""

    def __call__(self, env, proposal, active):
        stop_target = super().__call__(env, proposal, active)
        if not active or not np.any(self.coefficients) or env.d.time < 2.4 - base.EPSILON:
            return stop_target
        data = self.kinematics.data
        observed_floor = self.lowest_points(data)[:, 2].min()
        data.qpos[env.p.qa] = proposal
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_comPos(self.model, data)
        tips = self.lowest_points(data)
        side = int(np.argmax(tips[:, 2]))
        lower_by = np.clip(tips[side, 2] - observed_floor - .005, 0., .010)
        if lower_by == 0.:
            return proposal
        body = int(self.model.geom_bodyid[self.soles[side]])
        mujoco.mj_jac(self.model, data, self.jac, None, tips[side], body)
        jacobian = self.jac[:, self.velocity_ids[side]]
        correction = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + self.regularizer,
                                                  np.array([0., 0., -.8 * lower_by]))
        result = proposal.copy()
        result[self.legs[side]] += np.clip(correction, -.10, .10)
        return np.clip(result, env.p.bounds[:, 0], env.p.bounds[:, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("study", "output", "pilot", "reference"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    parser.add_argument("--mode", choices=("anchor", "brake-ready"), default="anchor")
    args = parser.parse_args()
    for option in ("study", "output", "pilot", "reference"):
        setattr(args, option, getattr(args, option).resolve())
    reference = base.read(args.reference)
    weight = np.array(reference["validation"][0]["weights"])
    if not np.any(weight[base.DIMENSIONS:]):
        raise ValueError("The paired probe requires nonzero foot-capture weights")
    base.StopResidual = FootCapture
    base.score = braking_score
    campaign = base.Campaign(args)
    suite = base.paired_suite(base.VALIDATION_SEED, (4.,))
    variant = {"anchor": SupportAnchor, "brake-ready": BrakeReadyWalk}[args.mode]
    try:
        reference_cases = campaign.evaluate(weight, suite, "foot_capture", 0, args.output)
        campaign.stop = variant(campaign.box, args.study)
        anchor_cases = campaign.evaluate(weight, suite, args.mode, 0, args.output)
    except base.StopRequested:
        campaign.heartbeat("stopped")
        return
    output = {
        "mode": args.mode,
        "source_validation_generation": reference["generation"],
        "source_selected_generation": reference["selected_generation"],
        "reference_report_sha256": base.sha(args.reference),
        "same_walking_prefix": all(before["metrics"]["walk"] == after["metrics"]["walk"]
                                   for before, after in zip(reference_cases, anchor_cases, strict=True)),
        "reference": base.summary(reference_cases),
        "anchor": base.summary(anchor_cases),
        "reference_cases": reference_cases,
        "anchor_cases": anchor_cases,
    }
    base.replace_json(args.output / "report.json", output)
    campaign.heartbeat("completed")


if __name__ == "__main__":
    main()
