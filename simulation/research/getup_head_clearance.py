"""Head/battery clearance limit on the neck chain, derived from CAD contact geometry.

The released get-up gate failed with 6839 mm^3 of `S288_head_roll_CASE` (body
`jaw_soft`) inside `JMP_3S4000_90x42x27_ENVELOPE` (body `trunk_base`). Two facts
have to be separated before any plant change is justified:

  1. Are the flagged part pairs between bodies that MuJoCo would even collide?
     MuJoCo filters contacts between a body and its ancestors, so if every flagged
     pair is ancestor-descendant the engine can never represent the constraint and
     adding collision proxies is a no-op.
  2. If the constraint is unreachable by collision, what joint motion violates it,
     and what limit removes it?

This script answers both, prints the derived cap, and writes the evidence.
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from evaluate_policy import sha  # noqa: E402

CONTACT = ROOT / '20260914_contact_motion'
PLANT = CONTACT / 'normal_contact_plant'
CAD = CONTACT / 'getup_dense_final_cad'
HEAD_JOINTS = ('neck_pitch', 'head_pitch', 'head_yaw', 'head_roll')


def ancestor_chain(model, body_id):
    out = []
    while body_id:
        out.append(model.body(body_id).name)
        body_id = model.body_parentid[body_id]
    return out


def ancestry_audit(model, part_rows, frames):
    """Classify every flagged inter-body part pair by kinematic relationship."""
    owner = {r['name']: r['body'] for r in part_rows}
    pairs = {}
    for frame in frames:
        for row in frame['pairs']:
            if row['intersection_mm3'] <= 1e-9:
                continue
            a, b = (owner[n] for n in row['parts'])
            if a == b or a is None or b is None:
                continue
            pairs[tuple(sorted((a, b)))] = pairs.get(tuple(sorted((a, b))), 0.) + row['intersection_mm3']
    rows = []
    for (a, b) in sorted(pairs):
        id_a = model.body('robot/' + a).id
        id_b = model.body('robot/' + b).id
        chain_a = ancestor_chain(model, id_a)
        chain_b = ancestor_chain(model, id_b)
        if 'robot/' + b in chain_a:
            relation = f'{b}->{a}'
        elif 'robot/' + a in chain_b:
            relation = f'{a}->{b}'
        else:
            relation = 'NON_ADJACENT'
        rows.append(dict(bodies=[a, b], total_mm3=round(pairs[(a, b)], 4), relation=relation))
    return rows


def clearance(model, data, qpos, ga, gb, maxdist=.2):
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return float(mujoco.mj_geomDistance(model, data, ga, gb, maxdist, np.zeros(6)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--coarse', type=float, default=2.)
    p.add_argument('--fine', type=float, default=.25)
    p.add_argument('--out', default='getup_head_clearance')
    a = p.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')

    model = mujoco.MjModel.from_binary_path('n', assets={'n': (PLANT / 'nominal.mjb').read_bytes()})
    data = mujoco.MjData(model)
    contract = json.loads((PLANT / 'contract.json').read_text())
    home = np.zeros(model.nq)
    home[3] = 1.
    home[2] = .117182
    home[7:] = np.array(contract['action_offset'], dtype=float)
    names = contract['action_names']
    ja = model.geom('robot/jaw_soft_floor_proxy').id
    jb = model.geom('robot/trunk_base_floor_proxy').id
    addr = {n: model.jnt_qposadr[model.joint('robot/' + n).id] for n in HEAD_JOINTS}
    rng = {n: np.rad2deg(model.jnt_range[model.joint('robot/' + n).id]) for n in HEAD_JOINTS}
    nominal = {n: float(np.rad2deg(home[addr[n]])) for n in HEAD_JOINTS}

    # 1) Ancestry classification of the gate's own failures.
    parts = json.loads((CAD / 'collision_parts_indexed.json').read_text())
    frames = json.loads((CAD / 'solid_intersections.json').read_text())
    anc = ancestry_audit(model, parts, frames)
    nonadjacent = [r for r in anc if r['relation'] == 'NON_ADJACENT']
    print('--- flagged inter-body pairs, by kinematic relation ---')
    for r in anc:
        print(f"  {r['bodies'][0]:18s} {r['bodies'][1]:18s} {r['relation']:20s} {r['total_mm3']:10.3f} mm3")
    print(f'non-adjacent (engine-representable) pairs: {len(nonadjacent)}')

    # 2) Which head-chain motion drives the skull into the battery envelope.
    print('--- neck_pitch clearance sweep at nominal yaw/roll ---')
    lo, hi = rng['neck_pitch']
    sweep = []
    for angle in np.arange(lo, hi + 1e-9, a.coarse):
        q = home.copy()
        q[addr['neck_pitch']] = np.deg2rad(angle)
        sweep.append(dict(neck_pitch_deg=float(angle), clearance_mm=clearance(model, data, q, ja, jb) * 1000))
    for row in sweep:
        print(f"  neck_pitch {row['neck_pitch_deg']:7.2f}  clearance {row['clearance_mm']:8.2f} mm")

    # 3) Refine the first zero crossing, then re-check at the yaw/roll extremes.
    breach = next((r['neck_pitch_deg'] for r in sweep if r['clearance_mm'] < 0), None)
    crossings = {}
    for label, yaw, roll in (('nominal', nominal['head_yaw'], nominal['head_roll']),
                             ('yaw_centre', 0., 0.), ('yaw_min', rng['head_yaw'][0], 0.),
                             ('yaw_max', rng['head_yaw'][1], 0.),
                             ('roll_min', 0., rng['head_roll'][0]),
                             ('roll_max', 0., rng['head_roll'][1])):
        previous = None
        zero = None
        for angle in np.arange(lo, hi + 1e-9, a.fine):
            q = home.copy()
            q[addr['neck_pitch']] = np.deg2rad(angle)
            q[addr['head_yaw']] = np.deg2rad(yaw)
            q[addr['head_roll']] = np.deg2rad(roll)
            c = clearance(model, data, q, ja, jb) * 1000
            if c < 0 and previous is not None and zero is None:
                zero = float(previous[0] + (0 - previous[1]) * (angle - previous[0]) / (c - previous[1]))
            previous = (angle, c)
        crossings[label] = dict(zero_crossing_neck_pitch_deg=zero, head_yaw_deg=float(yaw), head_roll_deg=float(roll))
        print(f'  {label:11s} yaw={yaw:7.1f} roll={roll:6.1f}  first breach at neck_pitch={zero}')

    caps = [c['zero_crossing_neck_pitch_deg'] for c in crossings.values()
            if c['zero_crossing_neck_pitch_deg'] is not None]
    cap = float(np.floor(min(caps))) if caps else None
    report = dict(
        plant_sha256=sha(PLANT / 'nominal.mjb'), cad_frames=str(CAD),
        ancestry=anc, non_adjacent_pairs=len(nonadjacent),
        mu_joco_filters_ancestor_contacts=True,
        neck_pitch_range_deg=list(rng['neck_pitch']), nominal_neck_pitch_deg=nominal['neck_pitch'],
        coarse_sweep=sweep, first_breach_deg=breach, crossings=crossings,
        recommended_neck_pitch_cap_deg=cap,
        semantics='Conservative: proxy hulls contain the CAD parts, so a proxy-clearance '
                  'cap is stricter than the CAD-solid limit.')
    folder = CONTACT / a.out
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'clearance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(non_adjacent_pairs=len(nonadjacent),
                          recommended_neck_pitch_cap_deg=cap), indent=2), flush=True)


if __name__ == '__main__':
    main()
