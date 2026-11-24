"""Simulation-only test of whether early double support is reachable from the frozen gait."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

import continue_training as base
from train_foot_capture import FootCapture
from train_stop import braking_score


class ContactProbe(FootCapture):
    """Uses simulator contact and pose for feasibility diagnosis only."""

    def reset(self, env):
        super().reset(env)
        self.first_support = None
        self.acquired_dual = False

    def __call__(self, env, proposal, active):
        capture = super().__call__(env, proposal, active)
        walking = self.box.plan["commands"]["forward"][0] > 0.
        if walking or env.d.time < 2.4 - base.EPSILON or self.acquired_dual:
            return capture
        load = np.asarray(self.box.row["sole_N"])
        if self.first_support is None:
            self.first_support = int(np.argmax(load))
        if np.all(load > .5):
            self.acquired_dual = True
            return capture

        data = self.kinematics.data
        data.qpos[:] = env.d.qpos
        mujoco.mj_kinematics(self.model, data)
        floor = self.lowest_points(data)[:, 2].min()
        data.qpos[env.p.qa] = capture
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_comPos(self.model, data)
        tips = self.lowest_points(data)
        output = capture.copy()
        for side, geom in enumerate(self.soles):
            depth = self.preload if side == self.first_support else self.swing_depth
            requested_z = np.clip(floor - depth - tips[side, 2], -.018, 0.)
            if requested_z == 0.:
                continue
            mujoco.mj_jac(self.model, data, self.jac, None, tips[side], int(self.model.geom_bodyid[geom]))
            j = self.jac[:, self.velocity_ids[side]]
            dq = j.T @ np.linalg.solve(j @ j.T + self.regularizer, np.array([0., 0., requested_z]))
            axes = self.legs[side]
            output[axes] = proposal[axes] + np.clip(capture[axes] - proposal[axes] + dq, -.17, .17)
        return np.clip(output, env.p.bounds[:, 0], env.p.bounds[:, 1])


def measure(trace, release):
    with np.load(trace) as arrays:
        select = arrays["time"] >= release - base.EPSILON
        t = arrays["time"][select] - release
        load = arrays["loads"][select]
        foot = arrays["foot"][select]
    use = np.flatnonzero(np.all(load > .5, axis=1))
    first = int(use[0]) if len(use) else None
    return {
        "first_dual_support_s": round(float(t[first]), 6) if first is not None else None,
        "dual_within_60_ms": bool(first is not None and t[first] <= .06 + base.EPSILON),
        "foot_displacement_at_first_dual_mm": (np.linalg.norm(foot[first] - foot[0], axis=1) * 1000.).tolist()
                                              if first is not None else None,
        "first_support_preserved": bool(np.all(load[:first + 1, np.argmax(load[0])] > .5))
                                   if first is not None else False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("study", "output", "pilot", "reference"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    for name in ("study", "output", "pilot", "reference"):
        setattr(args, name, getattr(args, name).resolve())
    reference = base.read(args.reference)
    weights = np.array(reference["validation"][0]["weights"])
    base.StopResidual = FootCapture
    base.score = braking_score
    campaign = base.Campaign(args)
    suite = [(domain, release, dt, .60) for domain, release, dt, _ in
             base.paired_suite(base.VALIDATION_SEED, (4.,))]
    report = []
    designs = (None, (.0005, .004), (.001, .008), (.002, .013))
    journal = args.output / "trials.jsonl"
    completed = {}
    if journal.exists():
        for line in journal.read_bytes().splitlines():
            row = json.loads(line)
            completed.setdefault(row["candidate"], {})[row["case"]] = row
        campaign.count = sum(len(rows) for rows in completed.values())
    try:
        for i, shape in enumerate(designs):
            if shape is not None:
                campaign.stop = ContactProbe(campaign.box, args.study)
                campaign.stop.preload, campaign.stop.swing_depth = shape
            prior = completed.get(i, {})
            if len(prior) == len(suite) and all((args.output / f"contact_probe_{i}_{slot}.npz").is_file()
                                               for slot in prior):
                cases = [prior[slot] for slot in sorted(prior)]
            else:
                cases = campaign.evaluate(weights, suite, "contact_probe", i, args.output)
            report.append(dict(preload_and_swing_depth_m=shape, cases=[
                dict(case=case["case"], dt_s=case["dt_s"],
                     **measure(args.output / f"contact_probe_{i}_{case['case']}.npz", case["release_s"]))
                for case in cases
            ]))
            base.replace_json(args.output / "report.json", {
                "scope": "Simulation-only feasibility probe, using true contact and pose. Not a trained or deployable policy.",
                "reference_sha256": base.sha(args.reference),
                "completed_designs": len(report),
                "designs": report,
            })
    except base.StopRequested:
        campaign.heartbeat("stopped")
        return
    campaign.heartbeat("completed")


if __name__ == "__main__":
    main()
