"""Test age-aware geometric feedback with the same 50 Hz actuator publication."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

import continue_training as base
from train_foot_capture import FootCapture
from train_stop import braking_score


class PredictiveFeedback(FootCapture):
    gain = .8
    predict_orientation = False

    def reset(self, env):
        super().reset(env)
        self.ages = []

    def __call__(self, env, proposal, active):
        source = env.selected
        age = float(np.clip(float(env.d.time) - source["t_s"], 0., .06))
        if env.d.time >= 2.4 - base.EPSILON:
            self.ages.append(age)
        if self.gain == 0. or age < 1e-6:
            return super().__call__(env, proposal, active)
        snapshot = env.last_x
        predicted = source.copy()
        delta = np.clip(self.gain * age * source["encoder_dq"], -.13, .13)
        predicted["encoder_q"] = np.clip(source["encoder_q"] + delta, env.p.bounds[:, 0], env.p.bounds[:, 1])
        predicted_observation = snapshot.copy()
        predicted_observation[12:31] = predicted["encoder_q"] - env.p.home
        if self.predict_orientation:
            orientation = source["body_orientation_estimated"].copy()
            angular_rate = np.clip(source["gyro"], -3., 3.)
            mujoco.mju_quatIntegrate(orientation, angular_rate, self.gain * age)
            predicted["body_orientation_estimated"] = orientation
            predicted_observation[129:133] = orientation
        env.selected = predicted
        env.last_x = predicted_observation
        try:
            return super().__call__(env, proposal, active)
        finally:
            env.selected = source
            env.last_x = snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("study", "output", "pilot", "reference"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    args = parser.parse_args()
    for option in ("study", "output", "pilot", "reference"):
        setattr(args, option, getattr(args, option).resolve())
    reference = base.read(args.reference)
    weights = np.array(reference["validation"][0]["weights"])
    base.StopResidual = PredictiveFeedback
    base.score = braking_score
    campaign = base.Campaign(args)
    suite = base.paired_suite(base.VALIDATION_SEED, (4.,))
    rows = []
    try:
        for i, (gain, orientation) in enumerate(((0., False), (.8, False), (.8, True))):
            campaign.stop.gain = gain
            campaign.stop.predict_orientation = orientation
            cases = []
            for j, case in enumerate(suite):
                data = campaign.evaluate(weights, [case], f"feedback_age_{j}", i, args.output)[0]
                data["age_ms_mean"] = 1e3 * float(np.mean(campaign.stop.ages))
                data["age_ms_max"] = 1e3 * max(campaign.stop.ages)
                cases.append(data)
            rows.append(dict(gain=gain, orientation=orientation, summary=base.summary(cases), cases=cases))
            base.replace_json(args.output / "report.json", {
                "reference_sha256": base.sha(args.reference),
                "evaluations": rows,
            })
    except base.StopRequested:
        campaign.heartbeat("stopped")
        return
    campaign.heartbeat("completed")


if __name__ == "__main__":
    main()
