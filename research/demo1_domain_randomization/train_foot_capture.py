"""Train a foot-placement correction for zero-command support recovery."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

import continue_training as base
from train_stop import StopCampaign, WarmStop, braking_score


class FootCapture(WarmStop):
    """Project bounded end-effector corrections through the sensed geometry."""

    def __init__(self, bench, study):
        super().__init__(bench, study)
        self.box = bench.box
        self.model = bench.model
        self.velocity_ids = bench.box.plant.va[self.legs]
        self.jac = np.zeros((3, self.model.nv))
        self.regularizer = (.012 ** 2) * np.eye(3)

    def lowest_points(self, data):
        tips = []
        for geom, vertices in self.kinematics.soles:
            rotation = data.geom_xmat[geom].reshape(3, 3)
            lowest = vertices[np.argmin(vertices @ rotation[2])]
            tips.append(data.geom_xpos[geom] + rotation @ lowest)
        return np.asarray(tips)

    def __call__(self, env, proposal, active):
        # Keep the complete externally commanded walking prefix unchanged.
        walking = self.box.plan["commands"]["forward"][0] > 0.
        result = super().__call__(env, proposal, active or walking)
        if walking or active or env.d.time < 2.4 - base.EPSILON or not np.any(self.coefficients):
            return result

        data = self.kinematics.data
        observed_xy = data.geom_xpos[self.soles, :2].copy()
        surface = self.lowest_points(data)[:, 2].min()
        support = self.kinematics.filtered
        # Use a separate kinematics buffer to evaluate the commanded pose.
        # The simulator's actual state and solver never see this trial pose.
        data.qpos[env.p.qa] = result
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_comPos(self.model, data)
        commanded = self.lowest_points(data)
        fix_xy, swing_xy, descend, keep_height = np.clip(4.5 * np.abs(self.coefficients[0]), 0., .95)
        output = result.copy()
        for side, geom in enumerate(self.soles):
            amount = support[side]
            direction = np.empty(3)
            direction[:2] = (fix_xy * amount + swing_xy * (1. - amount)) * (
                observed_xy[side] - data.geom_xpos[geom, :2]
            )
            gap = surface - commanded[side, 2]
            direction[2] = amount * keep_height * gap + (1. - amount) * descend * min(0., gap - .0007)
            direction = np.clip(direction, -.012, .012)
            mujoco.mj_jac(self.model, data, self.jac, None, commanded[side], int(self.model.geom_bodyid[geom]))
            jacobian = self.jac[:, self.velocity_ids[side]]
            correction = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + self.regularizer, direction)
            axes = self.legs[side]
            output[axes] = proposal[axes] + np.clip(result[axes] - proposal[axes] + correction, -.17, .17)
        return np.clip(output, env.p.bounds[:, 0], env.p.bounds[:, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    args = parser.parse_args()
    for name in ("study", "output", "pilot"):
        setattr(args, name, getattr(args, name).resolve())
    if (args.output / "STOP").exists():
        return
    args.output.mkdir(parents=True, exist_ok=True)
    identity_path = args.output / "capture_variant.json"
    identity = dict(method="encoder_sole_projection", source_sha256=base.sha(Path(__file__)))
    if identity_path.exists() and base.read(identity_path) != identity:
        raise ValueError("Foot capture source changed; use a new output directory")
    if not identity_path.exists():
        base.replace_json(identity_path, identity)
    base.StopResidual = FootCapture
    base.score = braking_score
    try:
        StopCampaign(args).run()
    except base.StopRequested:
        return


if __name__ == "__main__":
    main()
