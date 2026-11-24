"""Identify a local servo-response model and test its bounded command lead."""

import argparse
from pathlib import Path

import numpy as np

import continue_training as base
from train_foot_capture import FootCapture
from train_stop import braking_score


def response_samples(trace, slots, joints):
    with np.load(trace) as source:
        t = source["action_time"][2:]
        angle = source["angle_rad"][:, 3, joints]
        held = source["angle_rad"][:, 2, joints]
    delayed_fraction = np.array(slots)[joints] / 4.
    effective = (1. - delayed_fraction) * held[2:] + delayed_fraction * held[1:-1]
    previous_increment = angle[1:-1] - angle[:-2]
    error = effective - angle[1:-1]
    observed = angle[2:] - angle[1:-1]
    select = (t >= 0.) & (t < 4.6)
    return np.stack((previous_increment[select], error[select]), axis=2), observed[select]


def fit_response(directory, joints, names):
    report = base.read(directory / "report.json")
    cases = [response_samples(directory / f"validation_{report['generation']}_{row['case']}.npz",
                              row["domain"]["slots"], joints) for row in report["validation"]]
    train_x = np.concatenate([row[0] for row in cases[:4]])
    train_y = np.concatenate([row[1] for row in cases[:4]])
    check_x = np.concatenate([row[0] for row in cases[4:]])
    check_y = np.concatenate([row[1] for row in cases[4:]])
    fitted = []
    for i, name in enumerate(names):
        x = train_x[:, i, :]
        model = np.linalg.solve(x.T @ x + .015 * np.eye(2), x.T @ train_y[:, i])
        error = check_x[:, i, :] @ model - check_y[:, i]
        velocity_error = check_x[:, i, 0] - check_y[:, i]
        lag_ticks = float(np.clip((1. - model[0]) / max(.10, model[1]) - 1., 0., 2.))
        fitted.append(dict(axis=name, velocity_retention=float(model[0]), drive_response=float(model[1]),
                           approximate_lead_ticks=lag_ticks,
                           heldout_rmse_deg=float(np.degrees(np.sqrt(np.mean(error ** 2)))),
                           constant_velocity_rmse_deg=float(np.degrees(np.sqrt(np.mean(velocity_error ** 2)))),
                           no_update_rmse_deg=float(np.degrees(np.sqrt(np.mean(check_y[:, i] ** 2))))))
    return fitted


class ServoLead(FootCapture):
    gain = 0.

    def reset(self, env):
        super().reset(env)
        self.previous_target = None

    def __call__(self, env, proposal, active):
        target = super().__call__(env, proposal, active)
        ids = self.legs.ravel()
        current = target[ids].copy()
        prior = self.previous_target
        self.previous_target = current
        if (prior is None or self.gain == 0. or active or not np.any(self.coefficients)
                or self.box.plan["commands"]["forward"][0] > 0. or env.d.time < 2.4 - base.EPSILON):
            return target
        # Use a smaller lead on whichever leg is inferred to be bearing weight.
        role = np.repeat(.35 + .65 * (1. - self.kinematics.filtered), self.legs.shape[1])
        advanced = np.clip(self.gain * role * self.lead_ticks * (current - prior), -.07, .07)
        output = target.copy()
        output[ids] = proposal[ids] + np.clip(current + advanced - proposal[ids], -.17, .17)
        return np.clip(output, env.p.bounds[:, 0], env.p.bounds[:, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("study", "output", "pilot", "reference", "fit-samples"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    args = parser.parse_args()
    for option in ("study", "output", "pilot", "reference", "fit_samples"):
        setattr(args, option, getattr(args, option).resolve())
    reference = base.read(args.reference)
    weights = np.array(reference["validation"][0]["weights"])
    base.StopResidual = ServoLead
    base.score = braking_score
    campaign = base.Campaign(args)
    indices = campaign.box.legs.ravel()
    names = [campaign.box.box.plant.names[i] for i in indices]
    model = fit_response(args.fit_samples, indices, names)
    campaign.stop.lead_ticks = np.array([row["approximate_lead_ticks"] for row in model])
    suite = base.paired_suite(913271, (4.,))
    results = []
    try:
        for i, gain in enumerate((0., .35, .7)):
            campaign.stop.gain = gain
            cases = campaign.evaluate(weights, suite, "actuator_prefilter", i, args.output)
            results.append(dict(gain=gain, summary=base.summary(cases), cases=cases))
            base.replace_json(args.output / "report.json", {
                "scope": "Local simulated actuator response; no real actuator tolerance inferred",
                "reference_sha256": base.sha(args.reference),
                "fit_cases_and_validation": "Fit nominal/mild from two integration steps; validate angle increments on moderate",
                "model": model,
                "evaluations": results,
            })
    except base.StopRequested:
        campaign.heartbeat("stopped")
        return
    campaign.heartbeat("completed")


if __name__ == "__main__":
    main()
