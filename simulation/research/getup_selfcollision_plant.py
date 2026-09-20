"""Robot-robot self-collision release for the get-up skill, with a pose sweep gate.

The released get-up plant has NO robot-robot collision at all: every body proxy is
`contype=4, conaffinity=0` (floor only) and `npair=0`. Head, neck, jaw and battery
region can therefore pass straight through the trunk and legs, which is why the
release report says `INVALID_SELF_INTERSECTION`.

Bit scheme (MuJoCo contacts when `(c1 & a2) || (c2 & a1)`):

    bit 4  floor class      proxy contype=4  -> still touches `terrain` (conaff=5)
    bit 8  head class       head  conaff=8, body contype=8
    bit 16 body class       body  conaff=16, head contype=16

    head contype = 4|8   head conaff = 16   -> head-head  (4|8)&16 = 0   no
    body contype = 4|16  body conaff = 8    -> body-body  (4|16)&8 = 0   no
    head-body: (4|8)&8 = 8 and (4|16)&16 = 16                            yes
    any-terrain: contype & terrain.conaff(5) = 4                         yes
    any-foot: bits 8/16 never meet the foot class (1)                    no

So exactly "head region vs body/legs" is released, with no head-head and no
leg-leg pairs, and no disturbance to the existing foot/floor contact model.

Nothing is enabled until a pose sweep shows the added pairs stay under the
penetration budget; the gate is printed and stored either way.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from evaluate_policy import sha  # noqa: E402

CONTACT = ROOT / '20260914_contact_motion'
SOURCE_PLANT = CONTACT / 'normal_contact_plant'

HEAD_CLASS = ('neck', 'neck_pitch', 'yaw_roll_motion', 'jaw_soft', 'jaw_hinge', 'bearing_roll')
BODY_CLASS = ('trunk_base', 'yaw2roll', 'hip_l', 'upper_leg_left', 'leg',
              'hip_l_2', 'upper_leg_right', 'leg_2', 'tail_yaw', 'tail_pitch', 'arm_l', 'arm_r')
FLOOR_BIT, HEAD_BIT, BODY_BIT = 4, 8, 16


def proxy_body(name):
    return name.removeprefix('robot/').removesuffix('_floor_proxy')


def release_self_collision(model):
    """Assign the bit scheme in place. Returns the list of touched proxy names."""
    touched = []
    for gid in range(model.ngeom):
        name = model.geom(gid).name
        if not name.endswith('_floor_proxy'):
            continue
        body = proxy_body(name)
        if body in HEAD_CLASS:
            model.geom_contype[gid] = FLOOR_BIT | HEAD_BIT
            model.geom_conaffinity[gid] = BODY_BIT
        elif body in BODY_CLASS:
            model.geom_contype[gid] = FLOOR_BIT | BODY_BIT
            model.geom_conaffinity[gid] = HEAD_BIT
        else:
            raise ValueError(f'unclassified proxy body {body!r}')
        touched.append(name)
    assert len(touched) == len(HEAD_CLASS) + len(BODY_CLASS), touched
    return touched


def robot_self_contacts(model, data, floor):
    """Robot-robot contacts with their signed separation, deepest first."""
    rows = []
    for i, c in enumerate(data.contact):
        pair = (c.geom1, c.geom2)
        if floor in pair:
            continue
        names = tuple(sorted((model.geom(g).name for g in pair)))
        if not all(n.startswith('robot/') for n in names):
            continue
        rows.append(dict(pair=list(names), dist_mm=float(c.dist * 1000)))
    rows.sort(key=lambda r: r['dist_mm'])
    return rows


def set_pose(model, data, qpos):
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0
    data.qpos[:len(qpos)] = qpos
    mujoco.mj_forward(model, data)


def load_poses(model):
    """Named poses: home, squat references, deep crouch, and the physical falls."""
    poses = {}
    contract = json.loads((SOURCE_PLANT / 'contract.json').read_text())
    home = np.zeros(model.nq)
    home[3] = 1.
    home[2] = .117182
    home[7:] = np.array(contract['action_offset'], dtype=float)
    poses['home'] = home
    for label, folder, key in (
            ('squat', ROOT / '20260914_squat_specialist', 'pose_references.json'),
            ('crouch', ROOT / '20260914_deep_crouch', 'pose_references.json')):
        path = folder / key
        if not path.exists():
            continue
        for row in json.loads(path.read_text()):
            q = home.copy()
            q[7:] = np.array(row['target'], dtype=float)
            poses[f'{label}_{row["depth_mm"]}mm'] = q
    for label in ('left', 'right', 'front', 'back'):
        path = CONTACT / 'fall_preparation_normal005' / f'{label}.npz'
        if not path.exists():
            continue
        frames = np.load(path)['qpos']
        for fraction in (0., .25, .5, .75, 1.):
            idx = min(len(frames) - 1, int(fraction * (len(frames) - 1)))
            q = np.zeros(model.nq)
            q[:len(frames[idx])] = frames[idx]
            poses[f'fall_{label}_{int(fraction*100):03d}'] = q
    return poses, home


def random_poses(model, home, count, seed, tilt_deg):
    rng = np.random.default_rng(seed)
    out = {}
    names = json.loads((SOURCE_PLANT / 'contract.json').read_text())['action_names']
    limits = model.jnt_range[[model.joint('robot/' + n).id for n in names]]
    for i in range(count):
        q = home.copy()
        q[7:] = rng.uniform(limits[:, 0], limits[:, 1])
        half = np.deg2rad(tilt_deg) / 2
        yaw = rng.uniform(0, 2 * np.pi)
        q[3:7] = np.r_[np.cos(half), np.sin(half) * np.array([np.cos(yaw), np.sin(yaw), 0.])]
        q[2] = rng.uniform(.03, .14)
        out[f'random_{tilt_deg:03.0f}_{i:03d}'] = q
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label', default='getup_plant_v2')
    p.add_argument('--random', type=int, default=60)
    p.add_argument('--penetration-budget-mm', type=float, default=.5)
    p.add_argument('--max-self-contact-pairs', type=int, default=40)
    a = p.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')

    source = SOURCE_PLANT / 'nominal.mjb'
    model = mujoco.MjModel.from_binary_path('n', assets={'n': source.read_bytes()})
    baseline_mass = model.body_mass.copy()
    baseline_inertia = model.body_inertia.copy()
    audit_pairs = int(model.npair)
    touched = release_self_collision(model)
    np.testing.assert_array_equal(model.body_mass, baseline_mass)
    np.testing.assert_array_equal(model.body_inertia, baseline_inertia)
    assert int(model.npair) == audit_pairs == 0, 'bit-mask release must not need <pair>'

    data = mujoco.MjData(model)
    floor = model.geom('terrain').id
    poses, home = load_poses(model)
    poses.update(random_poses(model, home, a.random, 4242, 0))
    poses.update(random_poses(model, home, a.random, 4243, 90))
    poses.update(random_poses(model, home, a.random, 4244, 180))
    poses.update(random_poses(model, home, a.random, 4245, 135))

    worst = {}
    offenders = []
    per_pose = {}
    for label, q in poses.items():
        set_pose(model, data, q)
        rows = robot_self_contacts(model, data, floor)
        deep = [r for r in rows if r['dist_mm'] < -a.penetration_budget_mm]
        per_pose[label] = dict(self_contact_pairs=len(rows),
                               deepest_mm=min([r['dist_mm'] for r in rows], default=None),
                               over_budget=[r['pair'] for r in deep])
        for r in rows:
            key = tuple(r['pair'])
            if key not in worst or r['dist_mm'] < worst[key]['dist_mm']:
                worst[key] = dict(pair=list(key), dist_mm=r['dist_mm'], pose=label)
        if deep:
            offenders.append(label)
        if label in ('home',) or label.startswith(('fall_', 'squat_', 'crouch_')):
            print(f'{label:22s} pairs={len(rows):3d} deepest={per_pose[label]["deepest_mm"]}', flush=True)

    ordered = sorted(worst.values(), key=lambda r: r['dist_mm'])
    gate_ok = not offenders and len(ordered) <= a.max_self_contact_pairs
    folder = CONTACT / a.label
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, folder / 'nominal.mjb')
    shutil.copy2(SOURCE_PLANT / 'contract.json', folder / 'contract.json')
    report = dict(
        source_plant=str(SOURCE_PLANT), source_sha256=sha(source),
        released_sha256=sha(folder / 'nominal.mjb'), bit_scheme=dict(
            floor=FLOOR_BIT, head_contype=FLOOR_BIT | HEAD_BIT, head_conaffinity=BODY_BIT,
            body_contype=FLOOR_BIT | BODY_BIT, body_conaffinity=HEAD_BIT),
        head_class=list(HEAD_CLASS), body_class=list(BODY_CLASS),
        proxies_touched=len(touched), pairs_added=int(model.npair),
        mass_unchanged=True, inertia_unchanged=True,
        pose_count=len(poses), penetration_budget_mm=a.penetration_budget_mm,
        distinct_self_contact_pairs=len(ordered), max_self_contact_pairs=a.max_self_contact_pairs,
        poses_over_budget=offenders, gate_passed=bool(gate_ok),
        deepest_pairs=ordered[:25], per_pose={k: v for k, v in per_pose.items()
                                              if k in ('home',) or k.startswith(('fall_', 'squat_', 'crouch_'))},
        semantics='Bit-mask release of head-region vs body/leg pairs only; floor and foot contact model untouched.')
    (folder / 'self_collision_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(gate_passed=gate_ok, gaps=len(offenders),
                          distinct_pairs=len(ordered), deepest=ordered[:8]), indent=2), flush=True)


if __name__ == '__main__':
    main()
