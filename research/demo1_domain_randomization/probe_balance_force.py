"""Test a horizontal braking correction from an estimated stance and COM motion."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

import continue_training as base
from train_foot_capture import FootCapture
from train_stop import braking_score


class BrakeForce(FootCapture):
    gain = 0.

    def __call__(self, env, proposal, active):
        target = super().__call__(env, proposal, active)
        if (self.gain == 0. or active or self.box.plan["commands"]["forward"][0] > 0.
                or env.d.time < 2.4 - base.EPSILON):
            return target

        data = self.kinematics.data
        data.qpos[env.p.qa] = env.selected["encoder_q"]
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_comPos(self.model, data)
        support = .02 + np.clip(self.kinematics.filtered, 0., 1.) ** 2
        share = support / support.sum()
        midpoint = share @ data.geom_xpos[self.soles, :2]
        com_error = data.subtree_com[self.body, :2] - midpoint
        rotation = data.xmat[self.body].reshape(3, 3)
        world_speed = rotation[:2, :2] @ self.velocity
        motion = np.clip(np.linalg.norm(world_speed) / .08, 0., 1.)
        acceleration = np.clip(-8. * world_speed - 16. * motion * com_error, -3., 3.)
        force = np.r_[self.model.body_subtreemass[self.body] * self.gain * acceleration, 0.]

        result = target.copy()
        for side, geom in enumerate(self.soles):
            mujoco.mj_jacGeom(self.model, data, self.jac, None, geom)
            jacobian = self.jac[:, self.velocity_ids[side]]
            # Convert a desired support reaction to a motor position offset
            # using nominal stiffness. The physical drive limits still apply.
            correction = -share[side] * jacobian.T @ force / 7.
            axes = self.legs[side]
            result[axes] = proposal[axes] + np.clip(target[axes] - proposal[axes] + correction, -.17, .17)
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
    base.StopResidual = BrakeForce
    base.score = braking_score
    runner = base.Campaign(args)
    suite = base.paired_suite(base.VALIDATION_SEED, (4.,))
    records = []
    try:
        for i, gain in enumerate((0., .5, 1.)):
            runner.stop.gain = gain
            trials = runner.evaluate(weights, suite, "balance_force", i, args.output)
            records.append(dict(gain=gain, summary=base.summary(trials), cases=trials))
            base.replace_json(args.output / "report.json", {
                "reference_sha256": base.sha(args.reference), "evaluations": records,
            })
    except base.StopRequested:
        runner.heartbeat("stopped")
        return
    runner.heartbeat("completed")


if __name__ == "__main__":
    main()
