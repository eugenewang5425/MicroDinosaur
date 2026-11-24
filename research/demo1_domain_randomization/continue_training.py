"""Resumable actuator-randomization training on the frozen Demo 1 plant."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo1_domain_randomization.train import BUS_SLOT, DIMENSIONS, PERIOD, Workbench, draw_domain, sha


SEED = 624211
DEVELOPMENT_SEED = 720491
VALIDATION_SEED = 829517
POPULATION = 9
RELEASES = (1.48, 1.82, 2.46, 3.72, 4., 4.28)
SPEED = .8 * 79.03858833104131
TOTAL = DIMENSIONS + 16
EPSILON = 1e-8


def read(path):
    return json.loads(path.read_bytes())


def replace_json(path, value):
    temp = path.with_name(path.name + ".writing")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    for attempt in range(20):
        try:
            os.replace(temp, path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1)


class StopRequested(Exception):
    pass


class StopResidual:
    """Use an encoder/IMU sole-height proxy and anchored-foot velocity estimate."""

    def __init__(self, bench, study):
        sys.path.insert(0, str(study / "getup/unified_motion/turn_stop_pid"))
        from support_brake import EstimatedSupport

        self.kinematics = EstimatedSupport(bench.box.plant)
        self.legs = bench.legs
        self.soles = [geom for geom, _ in self.kinematics.soles]
        self.body = bench.box.root_body
        self.coefficients = np.zeros((4, 4))

    def reset(self, env):
        self.last_time = None
        self.last_position = None
        self.velocity = np.zeros(2)
        self.kinematics.filtered.fill(0.)
        self.kinematics.last.fill(0.)

    def __call__(self, env, proposal, active):
        if active or env.d.time < 2.4 - EPSILON or not np.any(self.coefficients):
            return proposal
        packet = env.selected
        support, _ = self.kinematics.estimate(env.last_x)
        position = self.kinematics.data.geom_xpos[self.soles, :2].copy()
        sample_time = packet["t_s"]
        if self.last_time is not None and sample_time > self.last_time + EPSILON:
            displacement = self.last_position - position
            rotation = self.kinematics.data.xmat[self.body].reshape(3, 3)
            velocity = (np.average(displacement, axis=0, weights=.05 + support) @ rotation[:2, :2]
                        / (sample_time - self.last_time))
            self.velocity += .35 * (np.clip(velocity, -.35, .35) - self.velocity)
        if self.last_time is None or sample_time > self.last_time + EPSILON:
            self.last_time = sample_time
            self.last_position = position

        q = packet["encoder_q"][self.legs]
        target = proposal[self.legs]
        error = np.clip((target - q) / .22, -1., 1.)
        grounded = support[:, None]
        speed = np.clip(self.velocity[[1, 0, 0, 0]] / .20, -1., 1.)
        rate = np.clip(packet["gyro"][[0, 1, 1, 1]] / 2., -1., 1.)
        firm, free, braking, damping = self.coefficients
        residual = (-firm * grounded * error - free * (1. - grounded) * error
                    + braking * speed * (.25 + .75 * grounded) + damping * rate)
        output = proposal.copy()
        output[self.legs] = target + np.clip(residual, -.14, .14)
        return np.clip(output, env.p.bounds[:, 0], env.p.bounds[:, 1])


def stage_for(generation):
    if generation < 4:
        return "nominal"
    if generation < 10:
        return "mild"
    return ("moderate", "mild", "moderate")[generation % 3]


def training_cases(generation):
    rng = np.random.default_rng(SEED + generation * 19)
    stage = stage_for(generation)
    release = RELEASES[generation % len(RELEASES)]
    other = (3.72, 4., 4.28)[generation % 3]
    slots = (BUS_SLOT, BUS_SLOT / 2)
    return (
        (draw_domain("nominal" if generation % 3 else stage, rng), other,
         slots[generation % 2], 2.6),
        (draw_domain(stage, rng), release, slots[(generation + 1) % 2], 2.6),
    )


def paired_suite(seed, durations, include_nominal=True):
    rng = np.random.default_rng(seed)
    stages = ("nominal", "mild", "moderate") if include_nominal else ("mild", "moderate")
    cases = []
    for name in stages:
        for release in durations:
            domain = draw_domain(name, rng)
            cases.extend((domain, release, dt, 10.) for dt in (BUS_SLOT, BUS_SLOT / 2))
    return cases


def score(row, release, horizon):
    walking, stop = row["walk"], row["parking"]
    progress = np.clip(walking["speed_mm_s"] / SPEED, -.5, 1.1)
    moves = np.clip(walking["sustained_alternations"] / (14 * release / 4), 0., 1.)
    cross = np.clip(abs(walking["cross_mm"]) / (77.2527 * release / 4), 0., 2.)
    walk = .8 * progress + .65 * moves - .35 * cross + .4 * walking["complete"]
    walk -= min(1., .2 * walking["peak_nonsole_N"])
    if stop["duration_s"] <= 0.:
        return float(walk - 3.)
    loss = (3.0 * (1. - stop["duration_s"] / horizon) + 1.15 * (1. - stop["dual"])
            + .25 * min(stop["longest_unloaded_s"] / .06, 7.)
            + .28 * min(stop["rms_m_s"] / .02, 5.)
            + .25 * min(stop["drift_mm"] / 25., 6.)
            + .6 * min(stop["foot_shift_mm"] / 6., 7.)
            + .25 * min(stop["stance_width_change_mm"] / 8., 6.)
            + .42 * min(stop["landings"], 7.) + min(1.5, .3 * stop["peak_nonsole_N"]))
    return float(walk - loss)


def summary(rows):
    return dict(
        count=len(rows),
        walked=sum(item["metrics"]["walk"]["stable"] for item in rows),
        parked=sum(item["metrics"]["parking"]["passed"] for item in rows),
        joint=sum(item["combined"] for item in rows),
        completed_holds=sum(item["metrics"]["parking"]["complete"] for item in rows),
        worst_speed_mm_s=min(item["metrics"]["walk"]["speed_mm_s"] for item in rows),
        mean_score=float(np.mean([item["objective"] for item in rows])),
    )


class Campaign:
    def __init__(self, args):
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output / "checkpoint.json"
        self.count = 0
        self.status = {}
        contract = {
            "script_sha256": sha(Path(__file__)),
            "simulation_sha256": sha(Path(__file__).with_name("train.py")),
            "pilot_sha256": sha(args.pilot),
            "seeds": [SEED, DEVELOPMENT_SEED, VALIDATION_SEED],
            "policy_hz": int(round(1 / PERIOD)),
            "population": POPULATION,
            "first_review_target_hours": 18,
        }
        file = self.output / "contract.json"
        if file.exists():
            if read(file) != contract:
                raise ValueError("Resuming with a changed script, source, or configuration requires a new directory")
        else:
            replace_json(file, contract)
        bank = read(args.pilot)["weights"]
        self.starts = [np.r_[np.asarray(bank[name]), np.zeros(16)]
                       for name in ("frozen", "fixed", "randomized")]
        self.state = read(self.state_path) if self.state_path.exists() else dict(
            generation=0, evaluations=0, incumbent=self.starts[-1].tolist(),
            center=self.starts[-1].tolist(), sigma=np.r_[np.full(16, .022), np.full(16, .048)].tolist(),
            best=None, started_s=time.time(),
        )
        self.count = self.state["evaluations"]
        self.development = paired_suite(DEVELOPMENT_SEED, (4.,))
        self.validation = paired_suite(VALIDATION_SEED, (4.,))
        self.phase_validation = paired_suite(VALIDATION_SEED + 1, (3.72, 4.28), include_nominal=False)
        self.heartbeat("loading")
        self.guard_ram()
        self.box = Workbench(args.study)
        self.stop = StopResidual(self.box, args.study)
        self.verify_imports(args.study)

    def verify_imports(self, study):
        imported = {}
        for module in tuple(sys.modules.values()):
            source = getattr(module, "__file__", None)
            if source:
                path = Path(source).resolve()
                if path.suffix == ".py" and path.is_relative_to(study):
                    imported[path.relative_to(study).as_posix()] = sha(path)
        contract = self.output / "native_sources.json"
        if contract.exists() and read(contract) != imported:
            raise ValueError("Frozen Demo 1 runtime imports changed since this campaign started")
        if not contract.exists():
            replace_json(contract, imported)

    def heartbeat(self, phase, **fields):
        self.status.update(fields)
        replace_json(self.output / "status.json", dict(
            pid=os.getpid(), phase=phase, updated_s=time.time(), generation=self.state["generation"],
            evaluations=self.count, started_s=self.state["started_s"], **self.status,
        ))

    def guard_ram(self):
        while psutil.virtual_memory().available < 3 * 1024**3:
            if (self.output / "STOP").exists():
                raise StopRequested
            self.heartbeat("waiting_for_memory")
            time.sleep(5.)

    def check(self):
        if (self.output / "STOP").exists():
            raise StopRequested
        if shutil.disk_usage(self.output).free < 2 * 1024**3:
            raise OSError("Less than 2 GiB free on the run drive")
        self.guard_ram()

    def evaluate(self, weights, cases, kind, candidate, trace_folder=None):
        rows = []
        for i, (domain, release, dt, hold) in enumerate(cases):
            self.check()
            self.heartbeat(kind, candidate=candidate, case=i, case_count=len(cases), dt_s=dt)
            self.stop.coefficients = weights[DIMENSIONS:].reshape(4, 4)
            trace = trace_folder / f"{kind}_{candidate}_{i}.npz" if trace_folder else None
            result = self.box.rollout(weights[:DIMENSIONS], domain, release, hold, dt, trace, self.stop)
            row = dict(
                generation=self.state["generation"], suite=kind, candidate=candidate, case=i, dt_s=dt,
                release_s=release, hold_s=hold, domain=asdict(domain), metrics=result,
                objective=score(result, release, hold),
                combined=bool(result["walk"]["stable"] and result["parking"]["passed"]
                              and result["walk"]["speed_mm_s"] >= result.get("speed_floor_mm_s", SPEED)),
                weights=weights.tolist(),
            )
            with (self.output / "trials.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(row, allow_nan=False) + "\n")
            self.count += 1
            rows.append(row)
        return rows

    def generation(self):
        i = self.state["generation"]
        rng = np.random.default_rng(SEED + 100003 + i * 23)
        center = np.array(self.state["center"])
        sigma = np.array(self.state["sigma"])
        incumbent = np.array(self.state["incumbent"])
        anchor = self.starts[i % len(self.starts)]
        population = [incumbent, anchor]
        origin_pool = (center, center, center, incumbent, anchor, self.starts[0], self.starts[2])
        for j, origin in enumerate(origin_pool):
            radius = 1.7 if j >= 4 and i % 4 == 3 else 1.
            population.append(np.clip(origin + rng.normal(size=TOTAL) * sigma * radius, -.20, .20))
        cases = training_cases(i)
        ranking = []
        for j, weights in enumerate(population):
            trials = self.evaluate(weights, cases, "training", j)
            outcomes = [item["objective"] for item in trials]
            utility = .6 * float(np.mean(outcomes)) + .4 * min(outcomes) - .025 * np.linalg.norm(weights)
            ranking.append((utility, weights))
        ranking.sort(key=lambda item: item[0], reverse=True)
        challenger = ranking[0][1]
        elite = np.asarray([item[1] for item in ranking[:3]])
        new_center = .5 * center + .5 * elite.mean(axis=0)
        spread = np.clip(.75 * sigma + .25 * elite.std(axis=0), .014, .06)
        if i % 12 == 11:
            spread = np.maximum(spread, .040)
        self.state.update(center=new_center.tolist(), sigma=spread.tolist(), incumbent=challenger.tolist())

        if i % 3 == 0:
            assessed = self.evaluate(challenger, self.development, "development", i)
            current = summary(assessed)
            previous = self.state["best"]
            rank = (current["joint"], current["parked"], current["walked"], current["mean_score"])
            if previous is None or rank > tuple(previous["rank"]):
                self.state["best"] = dict(generation=i, rank=rank, metrics=current, weights=challenger.tolist())
                replace_json(self.output / "candidate.json", self.state["best"])
        self.state.update(generation=i + 1, evaluations=self.count)
        replace_json(self.state_path, self.state)
        if i % 6 == 0 and self.state["best"] is not None:
            self.validate(i)

    def validate(self, generation):
        weights = np.array(self.state["best"]["weights"])
        milestone = self.output / f"validation_{generation:05d}"
        milestone.mkdir(exist_ok=True)
        rows = self.evaluate(weights, self.validation, "validation", generation, milestone if generation % 18 == 0 else None)
        phase = self.evaluate(weights, self.phase_validation, "release_phase", generation)
        baseline_file = self.output / "baseline.json"
        if not baseline_file.exists():
            baseline = self.evaluate(self.starts[0], self.validation, "frozen", 0)
            replace_json(baseline_file, summary(baseline))
        results = dict(generation=generation, selected_generation=self.state["best"]["generation"],
                       comparison=summary(rows), phase=summary(phase), frozen=read(baseline_file),
                       validation=rows, phase_validation=phase)
        replace_json(milestone / "report.json", results)
        replace_json(self.output / "latest_validation.json", results)
        self.state["evaluations"] = self.count
        replace_json(self.state_path, self.state)

    def run(self):
        try:
            while True:
                self.check()
                self.generation()
        except (StopRequested, KeyboardInterrupt):
            self.heartbeat("stopped")
        except Exception as exc:
            self.heartbeat("error", error=f"{type(exc).__name__}: {exc}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot", required=True, type=Path)
    args = parser.parse_args()
    args.study = args.study.resolve()
    args.output = args.output.resolve()
    args.pilot = args.pilot.resolve()
    if (args.output / "STOP").exists():
        return
    try:
        Campaign(args).run()
    except StopRequested:
        return


if __name__ == "__main__":
    main()
