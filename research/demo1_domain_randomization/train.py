"""Train a bounded residual on the frozen Demo 1 flat workbench."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np


PERIOD = .02
BUS_SLOT = .005
LIMIT = .12
MODEL_SHA = "213df9505f01cbfdf89df2e60aab5f372a79f0b9bd4f379748dc13533ba64dbf"
ACTOR_SHA = "034b60a759d4f5c3c3d2b600e03e327ddaa219d2a048f2bb9e041a104da5c67a"
TRAIN_SEED = 82431
HELDOUT_SEED = 296701
AXES = ("hip_roll", "hip_pitch", "knee", "ankle")
DIMENSIONS = 16
STAGES = {  # stiffness floor, sliding-friction floor, maximum added 5 ms slots
    "nominal": (1., 1., 0),
    "mild": (.94, .9, 1),
    "moderate": (.84, .76, 2),
}


@dataclass(frozen=True)
class Domain:
    stiffness: tuple[float, ...]
    friction: float
    slots: tuple[int, ...]


def draw_domain(stage, rng, axes=19):
    stiffness_min, friction_min, delay_max = STAGES[stage]
    stiffness = rng.uniform(stiffness_min, 1.)
    stiffness = stiffness + rng.uniform(-1., 1., axes) * (1. - stiffness_min) * .17
    return Domain(
        tuple(np.clip(stiffness, stiffness_min, 1.)),
        float(rng.uniform(friction_min, 1.)),
        tuple(rng.integers(0, delay_max + 1, axes).tolist()),
    )


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Workbench:
    def __init__(self, study):
        sys.path.insert(0, str(study / "getup/unified_motion/demo1_getup"))
        import learn_slope_balance as source

        self.bench = source.bench
        model_path = self.bench.verified_cache_path()
        actor_path = source.ROOT / "after/demo1_actor.onnx"
        if sha(model_path) != MODEL_SHA or sha(actor_path) != ACTOR_SHA:
            raise RuntimeError("Demo 1 model or policy differs from frozen source")
        plant, _, cases, minimum, _ = self.bench.load_sources()
        actor = source.unified.UnifiedActor(actor_path)
        source.configure(actor, True)
        self.box = self.bench.Sandbox(plant, actor, cases, minimum)
        self.model = plant.model
        if self.model.opt.timestep != BUS_SLOT or len(plant.ai) != 19:
            raise RuntimeError("Unexpected actuator count or native integration step")
        self.gain = self.model.actuator_gainprm[plant.ai].copy()
        self.bias = self.model.actuator_biasprm[plant.ai].copy()
        self.friction = self.model.geom_friction[:, 0].copy()
        self.soles = [self.model.geom(name).id for name in ("ground_p064", "ground_p002")]
        self.legs = np.array([[list(plant.names).index(f"{side}_{axis}")
                               for axis in AXES] for side in ("left", "right")])
        self.step = self.bench.recovery.base.step
        sys.path.insert(0, str(study / "deliveries/s288_transmission_model_20261113"))
        from walk_contact_quality import load_transfers

        self.load_transfers = load_transfers

    def rollout(self, weights, domain, release, hold, dt=BUS_SLOT, trace=None):
        box, model = self.box, self.model
        plant = box.plant
        repeats = round(BUS_SLOT / dt)
        if repeats * dt != BUS_SLOT:
            raise ValueError("Physics validation step must divide the 5 ms bus slot")
        model.opt.timestep = dt
        scale = np.asarray(domain.stiffness)
        model.actuator_gainprm[plant.ai, 0] = self.gain[:, 0] * scale
        model.actuator_biasprm[plant.ai, 1] = self.bias[:, 1] * scale
        model.geom_friction[:, 0] = self.friction * domain.friction
        box.reset("直立", "平地")
        box.plan = dict(box.plan, commands=dict(box.plan["commands"]))
        env = box.env
        if env.d.eq_active.any():
            raise RuntimeError("The base must be free of equality constraints")

        actuator_step = mujoco.mj_step
        previous = None
        delayed = np.asarray(domain.slots)
        correction_enabled = bool(np.any(weights))
        records = []
        logged = []
        ended = "completed"

        def apply(env, requested):
            nonlocal previous
            proposal = requested.copy()
            active = box.active_command[0] > 0.
            if correction_enabled and env.d.time >= 2.4 - 1e-8:
                packet = env.selected
                rates = packet["gyro"]
                gravity = packet["gravity_estimate"]
                shift = gravity[[1, 0]] + np.array([-.12, .12]) * rates[[0, 1]]
                feedback = np.clip(shift / .20, -1., 1.)[[0, 1, 1, 1]]
                original = proposal[self.legs]
                velocity_feature = 0. if previous is None else np.clip(
                    (original - previous) / .12, -1., 1.,
                )
                previous = original.copy()
                sensed = packet["encoder_q"]
                error = np.clip((original - sensed[self.legs]) / .25, -1., 1.)
                walking, posture, parking, bias = weights.reshape(4, 4)
                residual = (walking * velocity_feature + posture * feedback + bias * error
                            if active else parking * feedback + .35 * bias * error)
                proposal[self.legs] = original + np.clip(residual, -LIMIT, LIMIT)
                proposal = np.clip(proposal, plant.bounds[:, 0], plant.bounds[:, 1])
            held = env.d.ctrl[plant.ai].copy()
            slot = 0

            def apply_slot(m, d):
                nonlocal slot
                d.ctrl[plant.ai] = np.where(delayed <= slot, env.applied, held)
                for _ in range(repeats):
                    actuator_step(m, d)
                slot += 1

            if repeats != 1 or np.any(delayed):
                with patch.object(mujoco, "mj_step", apply_slot):
                    row = self.step(env, proposal)
            else:
                row = self.step(env, proposal)
            if trace is not None:
                logged.append(np.stack((requested, proposal, env.applied, env.d.qpos[plant.qa])))
            return row

        final_tick = round((release + hold) / PERIOD)
        with patch.object(self.bench.recovery.base, "step", apply):
            for tick in range(-120, final_tick + 1):
                t = tick * PERIOD
                command = min(.2, .02 * (tick + 1)) if 0 <= tick < round(release / PERIOD) else 0.
                if tick >= 0:
                    d = env.d
                    records.append(dict(
                        t=t,
                        root=d.qpos[:3].copy(),
                        loads=np.array(box.row["sole_N"]),
                        foot=d.geom_xpos[self.soles, :2].copy(),
                        rotation=d.xmat[box.root_body].reshape(3, 3).copy(),
                        root_speed_m_s=float(np.linalg.norm(d.qvel[:2])),
                        tilt=float(box.row["tilt_deg"]),
                        peak_other=float(box.row["nonsole_peak_5ms_N"]),
                    ))
                if box.row is not None and box.row["tilt_deg"] >= 45.:
                    ended = "tilted_45_deg"
                    break
                if tick == final_tick:
                    break
                box.plan["commands"]["forward"] = [command, 0., 0.]
                try:
                    box.advance("forward" if command else None)
                except (RuntimeError, ValueError) as exc:
                    if str(exc) != "NATIVE_PHYSICS_WARNING" and type(exc).__name__ != "RetargetDomainError":
                        raise
                    ended = type(exc).__name__ if isinstance(exc, ValueError) else str(exc)
                    break
                if box.recovery_tick is not None:
                    ended = "automatic_getup_detected"
                    break
        if trace is not None and records:
            np.savez_compressed(
                trace, time=np.array([r["t"] for r in records]),
                **{key: np.asarray([r[key] for r in records]) for key in
                   ("root", "loads", "foot", "rotation", "tilt", "peak_other", "root_speed_m_s")},
                action_time=np.arange(len(logged)) * PERIOD - 2.4,
                angle_rad=np.asarray(logged), axes=np.array(plant.names),
                angle_series=["actor_request", "residual_proposal", "published", "output_at_end_of_tick"],
            )
        return metrics(records, release, hold, ended, self.load_transfers)


def metrics(rows, release, hold, ended, load_transfers):
    active = [row for row in rows if row["t"] <= release + 1e-8]
    stop = [row for row in rows if row["t"] >= release - 1e-8]
    transfer = load_transfers(rows, release)
    alternations = transfer["sustained_alternations"]
    displacement = 1000. * (active[-1]["root"] - active[0]["root"]) if active else np.zeros(3)
    tilt = max((r["tilt"] for r in active), default=90.)
    walk_contact = max((r["peak_other"] for r in active), default=100.)
    completed_walk = bool(active and active[-1]["t"] >= release - 1e-8)
    walk = dict(
        complete=completed_walk, forward_mm=float(displacement[0]), cross_mm=float(displacement[1]),
        speed_mm_s=float(displacement[0] / release), sustained_alternations=alternations,
        peak_tilt_deg=tilt, peak_nonsole_N=walk_contact,
    )
    walk["stable"] = bool(completed_walk and displacement[0] > 0.
                          and alternations >= np.ceil(14 * release / 4.)
                          and abs(displacement[1]) <= 77.2527 * release / 4.
                          and tilt < 20. and walk_contact <= .5)
    parking = dict(complete=False, passed=False, duration_s=0.)
    if stop:
        loads = np.array([r["loads"] for r in stop]) > .5
        foot = np.array([r["foot"] for r in stop])
        root = np.array([r["root"][:2] for r in stop])
        dual = loads.all(axis=1)
        anchors = np.flatnonzero(np.r_[True, dual, True])
        landing = 0
        airborne = [None, None]
        # A swing that started while walking still counts if it lands after zero.
        for row in rows:
            t = row["t"]
            for side, load in enumerate(row["loads"]):
                if load <= .5 and airborne[side] is None:
                    airborne[side] = t
                elif load > .5 and airborne[side] is not None:
                    if t >= release - 1e-8 and t - airborne[side] >= .04 - 1e-8:
                        landing += 1
                    airborne[side] = None
        width = np.linalg.norm(foot[:, 0] - foot[:, 1], axis=1)
        duration = stop[-1]["t"] - stop[0]["t"]
        parking = dict(
            complete=duration >= hold - 1e-8, duration_s=duration, dual=float(dual.mean()),
            longest_unloaded_s=float((np.diff(anchors).max() - 1) * PERIOD),
            drift_mm=float(1000. * np.linalg.norm(root - root[0], axis=1).max()),
            rms_m_s=float(np.sqrt(np.mean([r["root_speed_m_s"] ** 2 for r in stop]))),
            foot_shift_mm=float(1000. * np.where(loads, np.linalg.norm(foot - foot[0], axis=2), 0.).max()),
            stance_width_change_mm=float(1000. * abs(width - width[0]).max()),
            peak_tilt_deg=max(r["tilt"] for r in stop),
            peak_nonsole_N=max(r["peak_other"] for r in stop), landings=landing,
        )
        parking["passed"] = bool(
            parking["complete"] and parking["dual"] >= .98 and parking["longest_unloaded_s"] <= .06
            and parking["drift_mm"] <= 25. and parking["rms_m_s"] <= .02
            and parking["foot_shift_mm"] <= 6. and parking["stance_width_change_mm"] <= 8.
            and parking["peak_tilt_deg"] <= 6. and parking["peak_nonsole_N"] <= .5 and landing <= 2
        )
    return dict(walk=walk, parking=parking, ended=ended)


def utility(result, release, hold):
    w, p = result["walk"], result["parking"]
    walk = (1.1 * np.clip(w["forward_mm"] / (250. * release / 4.), -.6, 1.1)
            + .75 * np.clip(w["sustained_alternations"] / (14 * release / 4.), 0., 1.15)
            - .35 * np.clip(abs(w["cross_mm"]) / 65., 0., 2.)
            - .45 * min(w["peak_nonsole_N"], 4.) / 4.
            - .30 * np.clip((w["peak_tilt_deg"] - 6.) / 20., 0., 2.) + .30 * w["complete"])
    if p["duration_s"] > 0.:
        park = (1.4 * p["duration_s"] / hold + .6 * p["dual"]
                - .42 * min(p["drift_mm"] / 65., 2.) - .38 * min(p["foot_shift_mm"] / 45., 2.)
                - .38 * min(p["peak_tilt_deg"] / 21., 2.) - .16 * min(p["landings"], 5)
                - .35 * min(p["peak_nonsole_N"], 4.) / 4. + .50 * p["passed"])
    else:
        park = -1.
    return float(walk + park)


def train(box, output, rounds, population, arm):
    rng = np.random.default_rng(TRAIN_SEED)
    domains_rng = np.random.default_rng(TRAIN_SEED + 1)
    mean = np.zeros(DIMENSIONS)
    incumbent = mean.copy()
    sigma = np.full(DIMENSIONS, .036)
    with (output / f"{arm}.jsonl").open("x", encoding="utf-8") as journal:
        for stage in STAGES:
            for iteration in range(rounds):
                release = (1.48, 1.82, 2.46)[(list(STAGES).index(stage) + iteration) % 3]
                group = stage if arm == "randomized" else "nominal"
                cohort = [draw_domain("nominal", domains_rng), draw_domain(group, domains_rng)]
                candidates = [incumbent, np.zeros(DIMENSIONS)]
                candidates.extend(np.clip(rng.normal(mean, sigma, (population - 2, DIMENSIONS)), -.10, .10))
                ranking = []
                for candidate, weights in enumerate(candidates):
                    results = [box.rollout(weights, domain, release, 2.2) for domain in cohort]
                    scores = [utility(result, release, 2.2) for result in results]
                    score = .6 * float(np.mean(scores)) + .4 * min(scores) - .025 * np.linalg.norm(weights / .12)
                    ranking.append((float(score), weights))
                    journal.write(json.dumps(dict(stage=stage, generation=iteration, candidate=candidate,
                                                  domains=[asdict(d) for d in cohort], release_s=release,
                                                  weights=weights.tolist(), scores=scores, objective=score,
                                                  results=results), allow_nan=False) + "\n")
                    journal.flush()
                ranking.sort(key=lambda entry: entry[0], reverse=True)
                incumbent = ranking[0][1].copy()
                elite = np.array([weights for _, weights in ranking[:max(2, population // 3)]])
                mean = .35 * mean + .65 * elite.mean(axis=0)
                sigma = np.clip(.75 * sigma + .25 * elite.std(axis=0), .012, .05)
                save(output / f"{arm}_weights.json", incumbent.tolist())
                print(arm, stage, iteration, "score", round(ranking[0][0], 3), flush=True)
    return incumbent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True, help="Local frozen Demo 1 study directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.population < 4:
        parser.error("require positive rounds and population >= 4")
    output = args.output
    output.mkdir(parents=True)
    box = Workbench(args.study.resolve())
    rng = np.random.default_rng(HELDOUT_SEED)
    zero = np.zeros(DIMENSIONS)
    nominal = draw_domain("nominal", rng)
    native = box.rollout(zero, nominal, 4., 10., trace=output / "native_reference.npz")
    save(output / "native_reference.json", native)
    print("native", json.dumps(native), flush=True)
    if args.smoke:
        return
    reference = native["walk"]["speed_mm_s"]
    arms = dict(frozen=zero)
    for arm in ("fixed", "randomized"):
        arms[arm] = train(box, output, args.rounds, args.population, arm)
    scenarios = [("nominal", 4., nominal)]
    for stage in ("mild", "moderate"):
        for release in (3.72, 4., 4.28):
            scenarios.append((stage, release, draw_domain(stage, rng)))
    checks = []
    for i, (stage, release, domain) in enumerate(scenarios):
        for dt in (.005, .0025):
            for arm, weights in arms.items():
                evaluation = box.rollout(weights, domain, release, 10., dt,
                                         output / f"heldout_{i}_{arm}_{dt:g}.npz")
                walk, stop = evaluation["walk"], evaluation["parking"]
                check = dict(scenario=i, stage=stage, release_s=release, dt_s=dt, arm=arm,
                             domain=asdict(domain), **evaluation)
                check["combined_pass"] = bool(walk["stable"] and stop["passed"]
                                              and walk["speed_mm_s"] >= .8 * reference)
                checks.append(check)
                save(output / "heldout.json", checks)
                print("heldout", i, dt, arm, round(walk["speed_mm_s"], 2),
                      walk["stable"], stop["passed"], check["combined_pass"], flush=True)
    save(output / "protocol.json", dict(
        model_sha256=MODEL_SHA, actor_sha256=ACTOR_SHA, script_sha256=sha(Path(__file__)),
        mujoco=mujoco.__version__, numpy=np.__version__, names=list(box.box.plant.names),
        training_seed=TRAIN_SEED, heldout_seed=HELDOUT_SEED, stages=STAGES,
        substep_s=BUS_SLOT, policy_s=PERIOD, population=args.population, rounds=args.rounds,
        weights={key: array.tolist() for key, array in arms.items()},
    ))


if __name__ == "__main__":
    main()
