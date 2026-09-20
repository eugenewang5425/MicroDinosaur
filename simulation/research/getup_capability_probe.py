"""Is the fallen get-up start escapable by *any* open-loop action sequence?

The released get-up candidate produced literally zero body motion for 12 s from
the back/left/right starts (1e-7 m/s, tilt constant to five decimals). Two very
different causes fit that trace:

  (a) the policy sits in a do-nothing basin it cannot leave, or
  (b) the robot is physically trapped in that pose and no action can move it.

This probe separates them by brute force: from an identical physical fall start
it applies random and structured open-loop action sequences and records how far
the body actually travels. If wide random search moves nothing, the plant/pose
is the problem; if it moves easily, learning/exploration is.

Read-only with respect to training. Nothing here is a skill acceptance gate.
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hardware_sim import HardwareCase  # noqa: E402
from evaluate_run_jump import MotionExperiment  # noqa: E402
from imu_owned_head import configure_owned  # noqa: E402
from build_contact_motion_bank import prepare_fall, constrain_recovery_head  # noqa: E402
from run_heading_stable_start import CALIBRATOR  # noqa: E402
from evaluate_policy import sha  # noqa: E402

OUT = ROOT / '20260914_contact_motion'
POLICY = OUT / 'training_runs/getup/20260914_dense_train_512x201/candidate.onnx'
DIRECTIONS = {'left': [0, 1, 0], 'right': [0, -1, 0], 'front': [-1, 0, 0], 'back': [1, 0, 0]}
# Contract action order: 5 left leg, 4 head, 5 right leg, 2 tail, 2 arm.
LEG_L = [0, 1, 2, 3, 4]
LEG_R = [10, 11, 12, 13, 14]
HEADS = [5, 6, 7, 8]


class Scripted:
    """Injected in place of the ONNX session; returns raw 19-d actions."""

    def __init__(self, fn):
        self._fn = fn

    def get_inputs(self):
        return [SimpleNamespace(name='obs')]

    def run(self, unused, inputs):
        obs = inputs[list(inputs)[0]][0]
        return [np.asarray(self._fn(obs), dtype=np.float32)[None]]


def tilt_deg(sim):
    r = sim.data.xmat[sim.body].reshape(3, 3)
    return float(np.rad2deg(np.arccos(np.clip(r[2, 2], -1, 1))))


def support(sim):
    """Loaded foot force total and non-foot (proxy) force total."""
    import mujoco
    floor = sim.model.geom('terrain').id
    feet = 0.0
    other = 0.0
    w = np.zeros(6)
    for i, c in enumerate(sim.data.contact):
        if c.efc_address < 0:
            continue
        mujoco.mj_contactForce(sim.model, sim.data, i, w)
        if floor not in (c.geom1, c.geom2):
            continue
        if w[0] <= 0:
            continue
        pair = (c.geom1, c.geom2)
        if any(g in pair for g in sim.foot_geoms):
            feet += float(w[0])
        else:
            other += float(w[0])
    return feet, other


def build_experiment(plant, torsion, delay):
    case = HardwareCase(motor_curve=True, voltage=12., physics_dt=.00125,
                        command_ms=delay, foot_torsional_friction_m=torsion)
    e = configure_owned(MotionExperiment(CALIBRATOR, 'run', 0., case,
                                         plant=plant, operating_envelope=True), (1,))
    constrain_recovery_head(e)
    previous_target = e.sim.transform_target

    def bounded_recovery(target, obs):
        limits = e.sim.model.jnt_range[e.sim.jids]
        return previous_target(np.clip(target, limits[:, 0] + .07, limits[:, 1] - .07), obs)

    e.sim.transform_target = bounded_recovery
    return e, case


def fallen_start(e, direction, seed):
    e.reset(seed)
    pre, _ = prepare_fall(e, direction)
    ok = bool(pre[-1, 2] > 25 and pre[-1, 3] < .05 and pre[-1, 4] < .5
              and pre[-1, 5] > 1 and pre[-1, 6] < .002)
    return ok


def rollout(e, action_fn, ticks=100):
    e.sim.session = Scripted(action_fn)
    e.rows = []
    e.qpos_frames = []
    sim = e.sim
    heights = []
    tilts = []
    feet_all = []
    other_all = []
    start_xy = sim.data.qpos[:2].copy()
    for _ in range(ticks):
        e.poll()
        sim.step(np.zeros(18))
        heights.append(float(sim.data.qpos[2]))
        tilts.append(tilt_deg(sim))
        f, o = support(sim)
        feet_all.append(f)
        other_all.append(o)
    xy = sim.data.qpos[:2] - start_xy
    return dict(tilt_start=tilts[0], tilt_end=tilts[-1], tilt_min=float(np.min(tilts)),
                tilt_max=float(np.max(tilts)), tilt_excursion=float(np.max(tilts) - np.min(tilts)),
                height_start=heights[0], height_end=heights[-1], height_max=float(np.max(heights)),
                xy_travel_mm=float(np.linalg.norm(xy) * 1000),
                foot_force_max=float(np.max(feet_all)), nonfoot_force_max=float(np.max(other_all)),
                finite=bool(np.isfinite(sim.data.qpos).all()))


def strategies(rng, n_random):
    """Named open-loop action generators, all in raw-action space."""
    def const(v):
        return lambda obs: np.full(19, v, np.float32)

    def legs(sign, amp):
        def fn(obs):
            a = np.zeros(19, np.float32)
            a[LEG_L] = sign * amp
            a[LEG_R] = sign * amp
            return a
        return fn

    def alternate(amp, freq):
        """Asymmetric leg drive: the pattern that could roll a body over."""
        state = {'t': 0}

        def fn(obs):
            t = state['t'] / 50.
            state['t'] += 1
            a = np.zeros(19, np.float32)
            s = np.sin(2 * np.pi * freq * t)
            a[LEG_L] = amp * s
            a[LEG_R] = -amp * s
            a[[1, 11]] = amp * 0.6 * np.cos(2 * np.pi * freq * t)
            return a
        return fn

    def random_walk(amp, freq):
        state = {'i': 0, 'v': np.zeros(19, np.float32)}

        def fn(obs):
            state['i'] += 1
            if state['i'] % 25 == 1:
                state['v'] = rng.uniform(-amp, amp, 19).astype(np.float32)
            return state['v']
        return fn

    def random_hold(amp):
        v = rng.uniform(-amp, amp, 19).astype(np.float32)
        return lambda obs: v

    out = {
        'hold_zero': const(0.),
        'hold_positive': const(1.),
        'hold_negative': const(-1.),
        'legs_flex_both': legs(1., 1.),
        'legs_extend_both': legs(-1., 1.),
        'alternate_1hz': alternate(1., 1.),
        'alternate_2hz': alternate(1., 2.),
        'random_walk': random_walk(1., 1.),
    }
    for i in range(n_random):
        out[f'random_hold_{i:02d}'] = random_hold(1.)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plant', default=str(OUT / 'normal_contact_plant'))
    p.add_argument('--torsional-friction', type=float, default=.015)
    p.add_argument('--delay', type=int, default=10)
    p.add_argument('--seed', type=int, default=941)
    p.add_argument('--directions', nargs='+', default=['back', 'left', 'right', 'front'])
    p.add_argument('--random', type=int, default=12)
    p.add_argument('--ticks', type=int, default=120)
    p.add_argument('--policy', type=Path, default=POLICY)
    p.add_argument('--out', default='getup_capability')
    a = p.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')

    dest = OUT / a.out
    dest.mkdir(parents=True, exist_ok=False)
    e, case = build_experiment(a.plant, a.torsional_friction, a.delay)
    rng = np.random.default_rng(a.seed)
    plan = dict(purpose='Separate do-nothing policy basin from physically trapped pose',
                arguments={k: str(v) for k, v in vars(a).items()},
                policy_sha256=sha(a.policy), plant_sha256=sha(Path(a.plant) / 'nominal.mjb'),
                hardware_case=asdict(case), ticks=a.ticks, control_dt_seconds=float(e.sim.dt),
                policy_is_graded_as_skill=False)
    (dest / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    report = {}
    for name in a.directions:
        direction = DIRECTIONS[name]
        ok = fallen_start(e, direction, a.seed)
        if not ok:
            report[name] = {'status': 'FALL_START_REJECTED'}
            print(f'--- {name}: fall start rejected', flush=True)
            continue
        obs = e.sim.obs(np.zeros(18))[0]
        started = dict(tilt_deg=tilt_deg(e.sim), height_mm=float(e.sim.data.qpos[2] * 1000),
                       obs_ang_vel=obs[0:3].tolist(), obs_gravity=obs[3:6].tolist(),
                       obs_joint_offset_max=float(np.abs(obs[6:25]).max()),
                       obs_joint_vel_max=float(np.abs(obs[25:44]).max()),
                       feet_force=support(e.sim)[0], nonfoot_force=support(e.sim)[1])
        rows = {}
        for label, fn in strategies(rng, a.random).items():
            if not fallen_start(e, direction, a.seed):
                rows[label] = {'status': 'FALL_START_REJECTED'}
                continue
            rows[label] = rollout(e, fn, a.ticks)
            r = rows[label]
            print(f'--- {name} {label}: tilt {r["tilt_start"]:.1f}->{r["tilt_end"]:.1f} '
                  f'(exc {r["tilt_excursion"]:.1f}) h {r["height_start"]*1000:.1f}->{r["height_end"]*1000:.1f} '
                  f'xy {r["xy_travel_mm"]:.1f}mm feet {r["foot_force_max"]:.1f}N', flush=True)
        # Baseline: the released policy itself, for a same-run comparison.
        if a.policy.exists() and fallen_start(e, direction, a.seed):
            session = ort.InferenceSession(str(a.policy), providers=['CPUExecutionProvider'])
            input_name = session.get_inputs()[0].name

            def released(obs, _s=session, _n=input_name):
                return _s.run(None, {_n: obs[None]})[0][0]

            rows['released_policy'] = rollout(e, released, a.ticks)
        report[name] = dict(start=started, trials=rows)
        (dest / f'{name}.json').write_text(json.dumps(report[name], indent=2), encoding='utf-8')
    (dest / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    # Compact verdict, printed last so it is the visible conclusion.
    verdict = {}
    for name, block in report.items():
        trials = block.get('trials', {})
        movers = {k: round(v['tilt_excursion'], 1) for k, v in trials.items()
                  if 'tilt_excursion' in v and v['tilt_excursion'] > 15}
        verdict[name] = dict(start_tilt=round(block.get('start', {}).get('tilt_deg', float('nan')), 1),
                             escapable_by_some_action=bool(movers), movers=movers,
                             best_excursion=round(max([v['tilt_excursion'] for v in trials.values()
                                                       if 'tilt_excursion' in v], default=0.), 1))
    (dest / 'verdict.json').write_text(json.dumps(verdict, indent=2), encoding='utf-8')
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == '__main__':
    main()
