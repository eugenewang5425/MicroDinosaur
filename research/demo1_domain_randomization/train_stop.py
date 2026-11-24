"""Train the zero-command transition while holding the Demo 1 walking prefix fixed."""

import argparse
from pathlib import Path
import time

import numpy as np

import continue_training as base


GROUPS = 16
SLOT = base.BUS_SLOT
WALK = np.zeros(base.DIMENSIONS)
RELEASE_PHASES = (4., 3.72, 4., 4.28, 2.46, 4.)


class WarmStop(base.StopResidual):
    def reset(self, env):
        super().reset(env)
        self.last_support = None

    def __call__(self, env, proposal, active):
        if env.d.time < 2.4 - base.EPSILON or not np.any(self.coefficients):
            return proposal
        packet = env.selected
        support, _ = self.kinematics.estimate(env.last_x)
        position = self.kinematics.data.geom_xpos[self.soles, :2].copy()
        t = packet["t_s"]
        if self.last_time is not None and t > self.last_time + base.EPSILON:
            displacement = self.last_position - position
            rotation = self.kinematics.data.xmat[self.body].reshape(3, 3)
            common_support = support * self.last_support
            estimated = np.average(displacement, axis=0, weights=.02 + common_support)
            estimated = estimated @ rotation[:2, :2] / (t - self.last_time)
            self.velocity += .35 * (np.clip(estimated, -.35, .35) - self.velocity)
        if self.last_time is None or t > self.last_time + base.EPSILON:
            self.last_time, self.last_position, self.last_support = t, position, support
        if active:
            return proposal

        target = proposal[self.legs]
        error = np.clip((target - packet["encoder_q"][self.legs]) / .20, -1., 1.)
        feature = np.clip(self.velocity[[1, 0, 0, 0]] / .20, -1., 1.)
        lateral_sign = np.array([[1., 1., 1., 1.], [-1., 1., 1., 1.]])
        inferred_support = support[:, None]
        rates = np.clip(packet["gyro"][[0, 1, 1, 1]] / 2., -1., 1.)
        motion = np.clip(np.linalg.norm(self.velocity) / .10, 0., 1.)
        grounded, swing, brake, gyro = self.coefficients
        residual = (-grounded * inferred_support * error - swing * (1. - inferred_support) * error
                    + lateral_sign * (brake * feature + gyro * rates))
        correction = np.clip(residual, -.15, .15) * (.30 + .70 * motion)
        output = proposal.copy()
        output[self.legs] = target + correction
        return np.clip(output, env.p.bounds[:, 0], env.p.bounds[:, 1])


def braking_score(result, release, hold):
    walk, park = result["walk"], result["parking"]
    if not walk["complete"]:
        return -1000. + np.clip(walk["speed_mm_s"] / base.SPEED, -4., 1.)
    if not park["complete"]:
        return -250. + 100. * park["duration_s"] / hold
    # The walking prefix is identical among candidates. Successful braking must
    # preserve its standard forward and support milestones in the paired replay.
    eligibility = 40. if walk["stable"] and walk["speed_mm_s"] >= base.SPEED else 0.
    excess = [
        (park["longest_unloaded_s"] / .06, 1.2),
        (park["foot_shift_mm"] / 6., 1.),
        (park["landings"] / 2., 1.1),
        (park["stance_width_change_mm"] / 8., .40),
        (park["drift_mm"] / 25., .65),
        (park["rms_m_s"] / .02, .65),
        ((1. - park["dual"]) / .02, .50),
        (park["peak_nonsole_N"] / .5, .8),
    ]
    cost = sum(weight * min(max(0., ratio - 1.), 8.) for ratio, weight in excess)
    margin = min(.10, park["rms_m_s"]) * 4. + min(60., park["foot_shift_mm"]) / 300.
    return float(eligibility - cost - margin)


class StopCampaign(base.Campaign):
    def __init__(self, args):
        super().__init__(args)
        variant = self.output / "variant.json"
        identity = {"method": "fixed_walking_prefix", "sha256": base.sha(Path(__file__))}
        if variant.exists() and base.read(variant) != identity:
            raise ValueError("The stopping trainer changed; use a new run directory")
        if not variant.exists():
            base.replace_json(variant, identity)

        self.starts = [np.zeros(base.DIMENSIONS + GROUPS)]
        if not self.state_path.exists():
            center = self.starts[0].copy()
            self.state.update(incumbent=center.tolist(), center=center.tolist(),
                              sigma=np.full(base.DIMENSIONS + GROUPS, .05).tolist(), started_s=time.time())
        if not (self.output / "frozen_development.json").exists():
            baseline_trials = self.evaluate(self.starts[0], self.development, "frozen_development", 0)
            metrics, rank = rank_cases(baseline_trials)
            self.state.update(best=dict(generation=-1, rank=rank, metrics=metrics, weights=self.starts[0].tolist()),
                              evaluations=self.count)
            base.replace_json(self.output / "frozen_development.json", metrics)
            base.replace_json(self.state_path, self.state)

    def generation(self):
        generation = self.state["generation"]
        rng = np.random.default_rng(base.SEED + 4201 + generation * 17)
        level = base.stage_for(generation)
        duration = RELEASE_PHASES[generation % len(RELEASE_PHASES)]
        domain = base.draw_domain(level, rng)
        cases = [(domain, duration, step, 3.2) for step in (SLOT, SLOT / 2)]
        if generation % 3 == 2:
            cases.append((base.draw_domain("nominal", rng), 4., SLOT, 3.2))

        center = np.array(self.state["center"])[base.DIMENSIONS:]
        sigma = np.array(self.state["sigma"])[base.DIMENSIONS:]
        incumbent = np.array(self.state["incumbent"])[base.DIMENSIONS:]
        population = [incumbent, np.zeros(GROUPS)]
        for j in range(base.POPULATION - 2):
            origin = center if j < 4 else np.zeros(GROUPS)
            population.append(np.clip(origin + rng.normal(size=GROUPS) * sigma, -.23, .23))
        ranked = []
        for i, gains in enumerate(population):
            results = self.evaluate(np.r_[WALK, gains], cases, "stop_training", i)
            values = [result["objective"] for result in results]
            ranked.append((.6 * float(np.mean(values)) + .4 * min(values), gains))
        ranked.sort(key=lambda item: item[0], reverse=True)
        best = ranked[0][1]
        elite = np.array([gains for _, gains in ranked[:3]])
        center = .4 * center + .6 * elite.mean(axis=0)
        sigma = np.clip(.72 * sigma + .28 * elite.std(axis=0), .016, .075)
        if generation % 12 == 11:
            sigma = np.maximum(sigma, .05)
        self.state.update(incumbent=np.r_[WALK, best].tolist(),
                          center=np.r_[WALK, center].tolist(), sigma=np.r_[WALK, sigma].tolist())

        if generation % 3 == 0:
            candidate = np.r_[WALK, best]
            dev = self.evaluate(candidate, self.development, "stop_development", generation)
            metrics, rank = rank_cases(dev)
            if rank > tuple(self.state["best"]["rank"]):
                self.state["best"] = dict(generation=generation, rank=rank, metrics=metrics, weights=candidate.tolist())
                base.replace_json(self.output / "candidate.json", self.state["best"])
        self.state.update(generation=generation + 1, evaluations=self.count)
        base.replace_json(self.state_path, self.state)
        if generation % 6 == 0:
            self.validate(generation)


def rank_cases(results):
    counts = base.summary(results)
    joint_horizon = sum(row["metrics"]["walk"]["stable"] and row["metrics"]["parking"]["complete"]
                        for row in results)
    rank = (counts["joint"], counts["parked"], joint_horizon, counts["walked"], counts["mean_score"])
    return counts, rank


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
    # Each trainer is its own process. Select the zero-command controller and
    # objective for the shared simulator and journal.
    base.StopResidual = WarmStop
    base.score = braking_score
    try:
        StopCampaign(args).run()
    except base.StopRequested:
        return


if __name__ == "__main__":
    main()
