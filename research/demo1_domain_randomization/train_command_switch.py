"""Train and validate zero-command recovery against frozen Demo 1 switching."""

import argparse
import itertools
from pathlib import Path
from unittest.mock import patch

import numpy as np

import continue_training as base
from demo1_domain_randomization import train as simulation
from train_foot_capture import FootCapture
from train_stop import WALK, rank_cases


SETTLE_S = 1.5
QUIET_S = 10.
HOLD_S = SETTLE_S + QUIET_S
REFERENCE_SEED = 38411179
TRAIN_SEED = 73317209
RELEASES = (4., 3.72, 4.28)
ORIGINAL_METRICS = simulation.metrics
VELOCITY = "speed_mm_s"
TRANSITION_FIELDS = ("landings", "foot_shift_mm", "drift_mm", "longest_unloaded_s")


def assess(rows, release, hold, ended, load_transfers):
    metrics = ORIGINAL_METRICS(rows, release, hold, ended, load_transfers)
    park = metrics["parking"]
    park["legacy_passed"] = park["passed"]
    stance_start = release + SETTLE_S
    # Reuse the standing check after the original policy's allowed transfer.
    # Start-of-window foot position is the stance anchor, not the release pose.
    stance_rows = [row for row in rows if row["t"] >= stance_start - base.EPSILON]
    stance = ORIGINAL_METRICS(
        stance_rows, stance_start, hold - SETTLE_S, ended,
        lambda *_: {"sustained_alternations": 0},
    )["parking"]
    clear = (ended == "completed" and metrics["walk"]["peak_nonsole_N"] <= .5
             and park.get("peak_nonsole_N", 100.) <= .5)
    park["passed"] = bool(park["complete"] and clear and stance["passed"])
    metrics["established_stance"] = stance
    return metrics


def transition_differences(frozen, candidate):
    original = frozen["metrics"]["parking"]
    revised = candidate["metrics"]["parking"]
    return {key: revised[key] - original[key] for key in TRANSITION_FIELDS
            if key in original and key in revised}


def paired_result(frozen, candidate):
    original_speed = frozen["metrics"]["walk"][VELOCITY]
    return dict(
        original_eligible=frozen["combined"],
        candidate_eligible=candidate["combined"],
        speed_ratio=(candidate["metrics"]["walk"][VELOCITY] / original_speed
                     if original_speed > 0. else None),
        transition_delta=transition_differences(frozen, candidate),
    )


class CommandSwitchCampaign(base.Campaign):
    def __init__(self, args):
        self.seed = np.asarray(base.read(args.seed)["weights"], dtype=float)
        if self.seed.shape != (base.TOTAL,) or not np.isfinite(self.seed).all() or np.any(self.seed[:base.DIMENSIONS]):
            raise ValueError("Expected a finite v5 seed with the frozen walking prefix")
        args.output.mkdir(parents=True, exist_ok=True)
        identity = {
            "method": "demo1_command_switch", "source": base.sha(Path(__file__)),
            "capture_source": base.sha(Path(__file__).with_name("train_foot_capture.py")),
            "seed": base.sha(args.seed), "reference_seed": REFERENCE_SEED, "train_seed": TRAIN_SEED,
        }
        stamp = args.output / "switch_contract.json"
        if stamp.exists() and base.read(stamp) != identity:
            raise ValueError("The command-switch contract changed; use a new run directory")
        if not stamp.exists():
            base.replace_json(stamp, identity)
        super().__init__(args)
        self.starts = [np.r_[WALK, np.zeros(16)], self.seed]
        self.development = [(domain, release, dt, HOLD_S) for domain, release, dt, _ in self.development]
        self.validation = [(domain, release, dt, HOLD_S) for domain, release, dt, _ in self.validation]
        self.phase_validation = [(domain, release, dt, HOLD_S) for domain, release, dt, _ in self.phase_validation]
        self.nominal_cases = self.cases("nominal", 4., np.random.default_rng(REFERENCE_SEED), HOLD_S)

    @staticmethod
    def cases(severity, release, rng, hold=HOLD_S):
        domain = base.draw_domain(severity, rng)
        return [(domain, release, step, hold) for step in (base.BUS_SLOT, base.BUS_SLOT / 2)]

    def run(self):
        try:
            self.audit()
            if self.state["best"] is None:
                evaluated = self.evaluate(self.seed, self.development, "seed_development", 0)
                metrics, rank = rank_cases(evaluated)
                self.state.update(center=self.seed.tolist(), incumbent=self.seed.tolist(), evaluations=self.count,
                                  best=dict(generation=-1, rank=rank, metrics=metrics, weights=self.seed.tolist()))
                base.replace_json(self.state_path, self.state)
            super().run()
        except (base.StopRequested, KeyboardInterrupt):
            self.heartbeat("stopped")
        except Exception as exc:
            self.heartbeat("error", error=f"{type(exc).__name__}: {exc}")
            raise

    def audit(self):
        report = self.output / "paired_review.json"
        if report.exists():
            self.reference = base.read(report)["nominal_reference"]
            return
        nominal = self.evaluate(self.starts[0], self.nominal_cases, "original_nominal", 0, self.output)
        if not all(row["combined"] for row in nominal):
            raise RuntimeError("Frozen Demo 1 must pass the revised nominal acceptance at both integration steps")
        self.reference = {str(row["dt_s"]): row["metrics"] for row in nominal}
        base.replace_json(self.output / "nominal_reference.json", self.reference)
        reports = []
        rng = np.random.default_rng(REFERENCE_SEED)
        for severity, release in (("nominal", 4.), ("mild", 4.), ("moderate", 4.), ("mild", 3.72),
                                  ("moderate", 4.28)):
            cases = self.nominal_cases if severity == "nominal" else self.cases(severity, release, rng)
            frozen = nominal if severity == "nominal" else self.evaluate(
                self.starts[0], cases, "original_held_out", len(reports))
            trained = self.evaluate(self.seed, cases, "v5_held_out", len(reports), self.output)
            reports.extend(dict(severity=severity, dt_s=old["dt_s"], release_s=release,
                                original=old, v5=new, paired=paired_result(old, new))
                           for old, new in zip(frozen, trained))
        base.replace_json(report, dict(
            reference_seed=REFERENCE_SEED, nominal_reference=self.reference, pairs=reports,
            original_combined=sum(row["original"]["combined"] for row in reports),
            v5_combined=sum(row["v5"]["combined"] for row in reports),
        ))

    def evaluate(self, weights, cases, kind, candidate, trace_folder=None):
        reference = getattr(self, "reference", {})

        def measured(rows, release, hold, ended, load_transfers):
            result = assess(rows, release, hold, ended, load_transfers)
            nominal = reference.get(str(self.box.model.opt.timestep))
            if nominal:
                walk = result["walk"]
                walk["nominal_speed_ratio"] = walk[VELOCITY] / nominal["walk"][VELOCITY]
                result["speed_floor_mm_s"] = .8 * nominal["walk"][VELOCITY]
                result["nominal_transition"] = nominal["parking"]
            return result

        with patch.object(simulation, "metrics", measured):
            return super().evaluate(weights, cases, kind, candidate, trace_folder)

    def generation(self):
        generation = self.state["generation"]
        rng = np.random.default_rng(TRAIN_SEED + generation)
        curriculum = ("mild", "nominal", "mild", "moderate") if generation < 4 else (
            "mild", "moderate", "nominal", "moderate")
        release = RELEASES[generation % len(RELEASES)] if generation >= 2 else 4.
        cases = self.cases(curriculum[generation % len(curriculum)], release, rng, SETTLE_S + 2.)
        incumbent = np.asarray(self.state["incumbent"])
        center = np.asarray(self.state["center"])[base.DIMENSIONS:]
        sigma = np.asarray(self.state["sigma"])[base.DIMENSIONS:]
        variants = np.clip(center + rng.normal(size=(4, 16)) * sigma, -.23, .23)
        population = itertools.chain((incumbent, self.seed), (np.r_[WALK, v] for v in variants))
        ranking = []
        for index, weights in enumerate(population):
            rows = self.evaluate(weights, cases, "switch_training", index)
            scores = [row["objective"] for row in rows]
            ranking.append((.6 * np.mean(scores) + .4 * min(scores), weights))
        ranking.sort(key=lambda item: item[0], reverse=True)
        winner = ranking[0][1]
        elite = np.asarray([weights[base.DIMENSIONS:] for _, weights in ranking[:3]])
        self.state.update(
            generation=generation + 1, evaluations=self.count, incumbent=winner.tolist(),
            center=np.r_[WALK, .5 * center + .5 * elite.mean(axis=0)].tolist(),
            sigma=np.r_[WALK, np.clip(.75 * sigma + .25 * elite.std(axis=0), .012, .055)].tolist(),
        )
        if generation % 2 == 0:
            rows = self.evaluate(winner, self.development, "switch_development", generation)
            metrics, rank = rank_cases(rows)
            if rank > tuple(self.state["best"]["rank"]):
                self.state["best"] = dict(generation=generation, rank=rank, metrics=metrics, weights=winner.tolist())
                base.replace_json(self.output / "candidate.json", self.state["best"])
        base.replace_json(self.state_path, self.state)
        if generation % 6 == 0:
            self.validate(generation)


def score(result, release, hold):
    walk, stop = result["walk"], result["parking"]
    stance = result["established_stance"]
    if not walk["complete"]:
        return -200. + min(1., walk[VELOCITY] / base.SPEED)
    if result["ended"] != "completed" or not stop["complete"] or max(
        walk["peak_nonsole_N"], stop.get("peak_nonsole_N", 100.)
    ) > .5:
        return -100. + 10. * stop["duration_s"] / hold
    reference = result.get("nominal_transition", stop)
    excess = sum(min(3., max(0., stop[key] / max(reference[key], .01) - 1.))
                 for key in TRANSITION_FIELDS)
    eligible = walk["stable"] and walk[VELOCITY] >= result.get("speed_floor_mm_s", base.SPEED)
    return (20. * eligible + 10. * stance["passed"] - .25 * excess
            - .05 * min(10., stance.get("drift_mm", 10.)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("study", "output", "pilot", "seed"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "STOP").exists():
        return
    base.StopResidual = FootCapture
    base.score = score
    CommandSwitchCampaign(args).run()


if __name__ == "__main__":
    main()
