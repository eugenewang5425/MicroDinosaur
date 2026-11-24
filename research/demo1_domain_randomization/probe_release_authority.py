"""Optimize three actuator updates from exact release states for feasibility diagnosis."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

import continue_training as base
from train_foot_capture import FootCapture


STATE = mujoco.mjtState.mjSTATE_INTEGRATION
TICKS = 3


class ReleaseFixture(FootCapture):
    def reset(self, env):
        super().reset(env)
        self.moving = False
        self.initial = None
        self.held = None
        self.actions = []

    def __call__(self, env, proposal, active):
        target = super().__call__(env, proposal, active)
        if self.box.plan["commands"]["forward"][0] > 0.:
            self.moving = True
        elif self.moving and len(self.actions) < TICKS:
            if self.initial is None:
                self.initial = np.empty(mujoco.mj_stateSize(self.model, STATE))
                mujoco.mj_getState(self.model, env.d, self.initial, STATE)
                self.held = env.applied.copy()
                self.first_load = np.asarray(self.box.row["sole_N"])
            self.actions.append(target.copy())
        return target


class ScriptedRelease(FootCapture):
    """Repeat an isolated feasibility witness, then return to closed-loop control."""

    def reset(self, env):
        super().reset(env)
        self.moving = False
        self.next_action = 0

    def __call__(self, env, proposal, active):
        target = super().__call__(env, proposal, active)
        if self.box.plan["commands"]["forward"][0] > 0.:
            self.moving = True
        elif self.moving and self.next_action < len(self.witness):
            reference = self.witness[self.next_action]
            self.next_action += 1
            return reference
        return target


class Replay:
    def __init__(self, campaign, fixture, domain, dt):
        self.m = campaign.box.model
        self.p = campaign.box.box.plant
        self.d = mujoco.MjData(self.m)
        self.soles = campaign.box.soles
        self.legs = campaign.box.legs.ravel()
        self.initial = fixture.initial
        self.held = fixture.held
        self.reference = np.array(fixture.actions)
        self.slots = np.asarray(domain.slots)
        self.repeats = round(base.BUS_SLOT / dt)
        self.old_foot = int(np.argmax(fixture.first_load))
        self.force = np.zeros(6)
        swing_geom = self.soles[1 - self.old_foot]
        mesh = int(self.m.geom_dataid[swing_geom])
        start = self.m.mesh_vertadr[mesh]
        self.swing_vertices = self.m.mesh_vert[start:start + self.m.mesh_vertnum[mesh]]
        self.swing_geom = swing_geom

    def advance(self, offset):
        m, d = self.m, self.d
        mujoco.mj_setState(m, d, self.initial, STATE)
        mujoco.mj_forward(m, d)
        initial_foot = d.geom_xpos[self.soles, :2].copy()
        applied = self.held.copy()
        old_ctrl = d.ctrl[self.p.ai].copy()
        loads, foot, clearance, peak_other = [], [], [], 0.
        for tick in range(TICKS):
            proposal = self.reference[tick].copy()
            proposal[self.legs] += offset[tick]
            proposal = np.clip(proposal, self.p.bounds[:, 0], self.p.bounds[:, 1])
            applied += np.clip(.9 * (proposal - applied), -.6, .6)
            for slot in range(4):
                d.ctrl[self.p.ai] = np.where(self.slots <= slot, applied, old_ctrl)
                for _ in range(self.repeats):
                    mujoco.mj_step(m, d)
                _, other = self.contact()
                peak_other = max(peak_other, other)
            old_ctrl = d.ctrl[self.p.ai].copy()
            mujoco.mj_forward(m, d)
            soles, other = self.contact()
            peak_other = max(peak_other, other)
            loads.append(soles)
            foot.append(d.geom_xpos[self.soles, :2].copy())
            z = d.geom_xmat[self.swing_geom, 6:9]
            clearance.append(float((self.swing_vertices @ z).min() + d.geom_xpos[self.swing_geom, 2]))
        return np.asarray(loads), np.asarray(foot), initial_foot, peak_other, clearance

    def contact(self):
        loads = np.zeros(2)
        other = 0.
        for i, c in enumerate(self.d.contact):
            if 0 not in (c.geom1, c.geom2):
                continue
            which = int(c.geom1 or c.geom2)
            mujoco.mj_contactForce(self.m, self.d, i, self.force)
            value = max(0., float(self.force[0]))
            if which in self.soles:
                loads[self.soles.index(which)] += value
            else:
                other += value
        return loads, other

    def loss(self, raw):
        loads, measured, start, other, clearance = raw
        old = loads[:, self.old_foot]
        swing = loads[-1, 1 - self.old_foot]
        foot_mm = float(1e3 * np.linalg.norm(measured - start, axis=2).max())
        old_shortfall = np.maximum(0., 1. - old).sum()
        both = bool(np.all(old > .5) and swing > .5)
        gap_cost = 160. * max(0., clearance[-1]) + 40. * max(0., min(clearance))
        score = (10. * both - 2.5 * old_shortfall + .4 * min(8., swing)
                 - .25 * max(0., foot_mm - 6.) - 8. * other - gap_cost)
        return score, {
            "old_support_all_three_ticks": bool(np.all(old > .5)),
            "swing_support_at_60_ms": float(swing),
            "dual_with_old_support_retained": both,
            "foot_max_displacement_mm": foot_mm,
            "peak_nonsole_N": other,
            "short_window_gate": both and foot_mm <= 6. and other <= .5,
            "per_tick_soles_N": loads.tolist(),
            "per_tick_swing_clearance_mm": [z * 1000. for z in clearance],
        }


def search(replay, seed, stop):
    rng = np.random.default_rng(seed)
    shape = (TICKS, len(replay.legs))
    zero = np.zeros(shape)
    native_score, baseline = replay.loss(replay.advance(zero))
    incumbent, best = zero, (native_score, baseline)
    center, sigma = zero.copy(), .075
    tested = 1
    unique = {zero.tobytes(): baseline}
    for iteration in range(10):
        population = [incumbent]
        for candidate in range(17):
            # Some candidates share all 3 ticks, and some can vary each update.
            noise = rng.normal(size=(1 if candidate < 8 else TICKS, shape[1]))
            population.append(np.clip(center + sigma * noise, -.17, .17))
        rows = []
        for target in population:
            if stop.exists():
                raise base.StopRequested
            score, metric = replay.loss(replay.advance(target))
            rows.append((score, metric, target))
            unique[target.tobytes()] = metric
            tested += 1
        rows.sort(key=lambda row: (row[1]["short_window_gate"], row[0]), reverse=True)
        score, metric, incumbent = rows[0]
        best = score, metric
        center = np.mean([row[2] for row in rows[:4]], axis=0)
        sigma = max(.024, sigma * .84)
    return {"frozen_actions": baseline, "optimized_actions": best[1],
            "candidates_replayed": tested, "distinct_targets": len(unique),
            "distinct_with_overlap": sum(x["dual_with_old_support_retained"] for x in unique.values()),
            "distinct_short_window_passes": sum(x["short_window_gate"] for x in unique.values()),
            "offset_rad": incumbent.tolist()}


def extend_witnesses(campaign, args, reference, profile):
    source = base.read(args.extend_witness / "report.json")
    if source["reference_sha256"] != base.sha(args.reference):
        raise ValueError("The feasibility witness must use the exact reference rollout")
    controller = ScriptedRelease(campaign.box, args.study)
    controller.coefficients = profile[base.DIMENSIONS:].reshape(4, 4)
    legs = campaign.box.legs.ravel()
    suite = base.paired_suite(base.VALIDATION_SEED, (4.,))
    results = []
    for witness in source["results"]:
        if not witness["optimized_actions"]["short_window_gate"]:
            continue
        case = witness["case"]
        domain, release, dt, hold = suite[case]
        with np.load(args.extend_witness / f"source_{case}.npz") as arrays:
            offsets = np.flatnonzero(arrays["action_time"] >= release - base.EPSILON)[:TICKS]
            controls = arrays["angle_rad"][offsets, 1].copy()
        controls[:, legs] += witness["offset_rad"]
        bounds = campaign.box.box.plant.bounds
        controller.witness = np.clip(controls, bounds[:, 0], bounds[:, 1])
        campaign.check()
        campaign.heartbeat("extend_witness", candidate=case, case=case, case_count=len(suite), dt_s=dt)
        trace = args.output / f"witness_{case}.npz"
        actual = campaign.box.rollout(profile[:base.DIMENSIONS], domain, release, hold, dt, trace, controller)
        with np.load(trace) as arrays:
            select = np.flatnonzero(arrays["time"] > release + base.EPSILON)[:TICKS]
            forces = arrays["loads"][select]
        expected = np.asarray(witness["optimized_actions"]["per_tick_soles_N"])
        if not np.allclose(forces, expected, rtol=0., atol=1e-6):
            raise RuntimeError("Witness did not reproduce its measured contact forces")
        assert actual["walk"] == reference["validation"][case]["metrics"]["walk"]
        results.append(dict(case=case, dt_s=dt, three_tick_soles_N=forces.tolist(), **actual))
        campaign.count += 1
        base.replace_json(args.output / "report.json", {
            "scope": "Long replay of a scenario-specific three-tick witness; offline feasibility only",
            "results": results,
        })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "output", "pilot", "reference"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--extend-witness", type=Path)
    args = p.parse_args()
    for name in ("study", "output", "pilot", "reference"):
        setattr(args, name, getattr(args, name).resolve())
    reference = base.read(args.reference)
    profile = np.array(reference["validation"][0]["weights"])
    base.StopResidual = ReleaseFixture
    campaign = base.Campaign(args)
    fixture = campaign.stop
    results = []
    try:
        if args.extend_witness is not None:
            extend_witnesses(campaign, args, reference, profile)
            campaign.heartbeat("completed")
            return
        for case, (domain, release, dt, _) in enumerate(base.paired_suite(base.VALIDATION_SEED, (4.,))):
            campaign.check()
            campaign.heartbeat("capture_release", candidate=case, case=0, case_count=6, dt_s=dt)
            fixture.coefficients = profile[base.DIMENSIONS:].reshape(4, 4)
            trace = args.output / f"source_{case}.npz"
            campaign.box.rollout(profile[:base.DIMENSIONS], domain, release, .12, dt, trace, fixture)
            replay = Replay(campaign, fixture, domain, dt)
            raw = replay.advance(np.zeros((TICKS, len(replay.legs))))
            with np.load(trace) as f:
                k = np.flatnonzero(f["time"] > release + base.EPSILON)[:TICKS]
                force_error = float(abs(f["loads"][k] - raw[0]).max())
                foot_error = float(abs(f["foot"][k] - raw[1]).max())
            if force_error > 1e-6 or foot_error > 1e-9:
                raise RuntimeError(f"Exact held-target replay mismatch for case {case}: {force_error:g} N, {foot_error:g} m")
            campaign.heartbeat("optimize_release", candidate=case, case=case, case_count=6, dt_s=dt)
            result = search(replay, 16421 + case, args.output / "STOP")
            campaign.count += result["candidates_replayed"]
            results.append(dict(case=case, dt_s=dt, replay_force_error_N=force_error,
                                replay_foot_error_m=foot_error, **result))
            base.replace_json(args.output / "report.json", {
                "scope": "Optimized three actuator updates from exact release states; offline feasibility only",
                "reference_sha256": base.sha(args.reference),
                "results": results,
            })
    except base.StopRequested:
        campaign.heartbeat("stopped")
        return
    campaign.heartbeat("completed")


if __name__ == "__main__":
    main()
