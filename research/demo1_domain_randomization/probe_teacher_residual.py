"""Distill short-window simulated targets into delayed encoder and IMU feedback."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

import continue_training as base
from train_foot_capture import FootCapture
from train_stop import braking_score


def features(packet, plant, legs):
    ids = legs.ravel()
    return np.r_[
        (packet["encoder_q"][ids] - plant.home[ids]) / .3,
        packet["encoder_dq"][ids] / 5.,
        packet["gravity_estimate"][:2] / .25,
        packet["gyro"][:2] / 2.,
    ]


def fit_teacher(directory, bench):
    report = base.read(directory / "report.json")
    if not report["phase_bank"]:
        raise ValueError("Expected the separate release-phase teacher bank")
    x, y, case_ids = [], [], []
    for row in report["results"]:
        frozen, improved = row["frozen_actions"], row["optimized_actions"]
        improvement = frozen["per_tick_swing_clearance_mm"][-1] - improved["per_tick_swing_clearance_mm"][-1]
        if improvement <= .25 or not improved["old_support_all_three_ticks"] or improved["peak_nonsole_N"] > .5:
            continue
        with np.load(directory / f"teacher_packet_{row['case']}.npz") as packet:
            x.append(features(packet, bench.box.plant, bench.legs))
            # Fit the immediate correction. The control policy can reevaluate
            # it on every sensor packet; it never replays the teacher sequence.
            y.append(packet["target_offset_rad"][0])
        case_ids.append(row["case"])
    if len(case_ids) < 4:
        raise ValueError("Too few physically informative teacher samples")
    data = np.asarray(x)
    mean = data.mean(axis=0)
    scale = np.maximum(.15, data.std(axis=0))
    centered = np.clip((data - mean) / scale, -3., 3.)
    design = np.c_[np.ones(len(centered)), centered]
    regularization = 5. * np.eye(design.shape[1])
    regularization[0, 0] = .5
    model = np.linalg.solve(design.T @ design + regularization, design.T @ np.asarray(y))
    return mean, scale, model, case_ids


class LearnedTouchdown(FootCapture):
    def __call__(self, env, proposal, active):
        target = super().__call__(env, proposal, active)
        if active or self.box.plan["commands"]["forward"][0] > 0. or env.d.time < 2.4 - base.EPSILON:
            return target
        data = self.kinematics.data
        data.qpos[env.p.qa] = env.selected["encoder_q"]
        mujoco.mj_kinematics(self.model, data)
        height = self.lowest_points(data)[:, 2]
        single_support = np.clip((np.ptp(height) - .0015) / .005, 0., 1.)
        measured = features(env.selected, env.p, self.legs)
        normalized = np.clip((measured - self.mean) / self.scale, -3., 3.)
        inferred = np.r_[1., normalized] @ self.model_fit
        result = target.copy()
        ids = self.legs.ravel()
        offset = self.gain * single_support * inferred
        result[ids] = proposal[ids] + np.clip(target[ids] - proposal[ids] + offset, -.17, .17)
        return np.clip(result, env.p.bounds[:, 0], env.p.bounds[:, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("study", "output", "pilot", "reference", "teacher"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    args = parser.parse_args()
    for option in ("study", "output", "pilot", "reference", "teacher"):
        setattr(args, option, getattr(args, option).resolve())
    reference = base.read(args.reference)
    teacher_report = base.read(args.teacher / "report.json")
    if teacher_report["reference_sha256"] != base.sha(args.reference):
        raise ValueError("The teacher bank must correspond to the same frozen feedback controller")
    weights = np.array(reference["validation"][0]["weights"])
    base.StopResidual = FootCapture
    base.score = braking_score
    campaign = base.Campaign(args)
    learned = LearnedTouchdown(campaign.box, args.study)
    learned.mean, learned.scale, learned.model_fit, identifiers = fit_teacher(args.teacher, campaign.box)
    np.savez_compressed(args.output / "linear_fit.npz", mean=learned.mean, scale=learned.scale,
                        coefficients=learned.model_fit, source_cases=identifiers)
    suite = base.paired_suite(base.VALIDATION_SEED, (4.,))
    rows = []
    try:
        for i, gain in enumerate((0., .5, 1.)):
            if gain:
                campaign.stop = learned
                learned.gain = gain
            cases = campaign.evaluate(weights, suite, "teacher_residual", i, args.output)
            rows.append(dict(gain=gain, summary=base.summary(cases), cases=cases))
            base.replace_json(args.output / "report.json", {
                "fit_case_count": len(identifiers),
                "teacher_sha256": base.sha(args.teacher / "report.json"),
                "evaluations": rows,
            })
    except base.StopRequested:
        campaign.heartbeat("stopped")
        return
    campaign.heartbeat("completed")


if __name__ == "__main__":
    main()
