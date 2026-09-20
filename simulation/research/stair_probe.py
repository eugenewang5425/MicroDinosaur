"""上台阶诊断探针:能力边界 + 四个假设的逐帧证据。

探针只测量,不训练。对每个台阶高度(h mm,3 级,总高 3h)记录:
  - 通过/卡住(前进 >=0.8m 为门),卡住时的前进距离
  - 逐帧:左右脚底最低点高度(相对当前台阶面)、髋/膝/踝角、根高、根俯仰、
          双足受力、足尖与台阶立面的最近距离
用来回答(用户提出的四个假设):
  ① 抬脚高度不够     -> 摆动腿脚底峰值高度 vs 台阶高 h
  ② 双腿协同不足     -> 两腿的抬脚时刻差/是否只有一条腿在干活
  ③ 折叠策略不够极限 -> 摆动相膝角峰值(收腿紧不紧)
  ④ 平衡没利用好     -> 支撑期 CoP/受力分布、头尾是否参与(尾巴受力)
"""
from pathlib import Path
import sys
import json
import argparse

import numpy as np
import mujoco

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from corrective_common import OUT  # noqa: E402
from evaluate_owned_terrain import experiment, evaluate  # noqa: E402
from terrain_skill_eval import ground_height  # noqa: E402
from evaluate_policy import sha  # noqa: E402

POLICY = Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_corrective_v2'
              '/20260917_train_512x201/candidate.onnx')
DELAY = 10
SPEED = .35


def foot_low(model, data, geom_id):
    """足底碰撞网格的世界最低点 z。"""
    mid = model.geom_dataid[geom_id]
    a = model.mesh_vertadr[mid]
    n = model.mesh_vertnum[mid]
    v = model.mesh_vert[a:a + n] @ data.geom_xmat[geom_id].reshape(3, 3).T \
        + data.geom_xpos[geom_id]
    return float(v[:, 2].min())


def run_case(height_mm, seed, speed=SPEED, delay=DELAY, seconds=14.0):
    kind = f'steps_{height_mm}'
    case = dict(terrain=kind, scenario='straight', program='legacy', depth=0,
                delay=delay, speed=speed, seconds=seconds)
    e = experiment(str(POLICY), case)
    s = e.sim
    m = s.model
    fl = m.geom('robot/left_foot_collision').id
    fr = m.geom('robot/right_foot_collision').id
    jn = s.names
    rows = []
    prev = s.substep_callback

    def cap():
        prev()
        if not e.active:
            return
        t = s.data.time - 6.
        if t <= 1e-8 or (rows and t - rows[-1]['t'] < .02 - 1e-9):
            return
        d = s.data
        q = d.qpos[s.jadr]
        rot = d.xmat[s.body].reshape(3, 3)
        x = d.qpos[0]
        floor = ground_height(kind, x, d.qpos[1])
        rows.append(dict(
            t=t, x=float(x), root_z=float(d.qpos[2] - floor),
            tilt=float(np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1)))),
            pitch=float(np.degrees(np.arctan2(rot[2, 0], rot[2, 2]))),
            foot_l=foot_low(m, d, fl) - floor,
            foot_r=foot_low(m, d, fr) - floor,
            hip_l=float(np.degrees(q[jn.index('left_hip_pitch')])),
            knee_l=float(np.degrees(q[jn.index('left_knee')])),
            ankle_l=float(np.degrees(q[jn.index('left_ankle')])),
            hip_r=float(np.degrees(q[jn.index('right_hip_pitch')])),
            knee_r=float(np.degrees(q[jn.index('right_knee')])),
            ankle_r=float(np.degrees(q[jn.index('right_ankle')])),
            tail=float(np.degrees(q[jn.index('tail_pitch')])),
        ))

    s.substep_callback = cap
    metrics, trace = evaluate(e, case, seed)
    s.substep_callback = prev
    return metrics, rows


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--heights', type=int, nargs='+', default=[10, 15, 20, 25, 30])
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--dump-height', type=int, default=None,
                    help='打印该高度 case 的逐帧时间线,看崩的瞬间')
    ap.add_argument('--dump-seed', type=int, default=1)
    a = ap.parse_args()
    if a.dump_height is not None:
        m, rows = run_case(a.dump_height, a.dump_seed)
        print(f'== steps_{a.dump_height} s{a.dump_seed} 前进'
              f'{m.get("forward_displacement_m"):.2f}m fell={m["fell"]}')
        print(f"{'t':>5s}{'x':>7s}{'根高':>7s}{'倾角':>7s}{'俯仰':>7s}"
              f"{'脚L':>8s}{'脚R':>8s}{'膝L':>7s}{'膝R':>7s}{'髋L':>7s}")
        step = max(1, len(rows) // 60)
        for r in rows[::step]:
            print(f"{r['t']:5.1f}{r['x']:7.3f}{r['root_z'] * 1000:7.1f}"
                  f"{r['tilt']:7.1f}{r['pitch']:7.1f}{r['foot_l'] * 1000:8.1f}"
                  f"{r['foot_r'] * 1000:8.1f}{r['knee_l']:7.1f}{r['knee_r']:7.1f}"
                  f"{r['hip_l']:7.1f}")
        return
    print(f'策略 {POLICY.name} sha={sha(POLICY)[:12]} 速度 {SPEED} 延迟 {DELAY}ms',
          flush=True)
    summary = []
    for h in a.heights:
        for sd in a.seeds:
            try:
                m, rows = run_case(h, sd)
            except Exception as exc:
                print(f'steps_{h} s{sd}: 异常 {exc!r}'[:100], flush=True)
                continue
            fwd = float(m.get('forward_displacement_m', 0.))
            ok = (not m['fell']) and fwd >= .8
            peak_l = max((r['foot_l'] for r in rows), default=0.)
            peak_r = max((r['foot_r'] for r in rows), default=0.)
            knee_min = min((min(r['knee_l'], r['knee_r']) for r in rows), default=0.)
            tail_load = m.get('foot_contact_fraction')
            summary.append(dict(height_mm=h, seed=sd, passed=bool(ok),
                                fell=bool(m['fell']), forward_m=fwd,
                                peak_foot_l_mm=peak_l * 1000,
                                peak_foot_r_mm=peak_r * 1000,
                                knee_fold_min_deg=knee_min,
                                tilt_max_deg=float(m.get('body_tilt_max_deg', 0.))))
            print(f'steps_{h:2d}(总高{h * 3:3d}mm) s{sd}: '
                  f'{"通过" if ok else "卡住"} 前进{fwd:5.2f}m '
                  f'抬脚峰值 L{peak_l * 1000:5.1f} R{peak_r * 1000:5.1f}mm '
                  f'最紧膝{knee_min:6.1f}° 倾角max{m.get("body_tilt_max_deg", 0):5.1f}°',
                  flush=True)
    dest = OUT / 'stair_probe.json'
    dest.write_text(json.dumps(dict(policy=str(POLICY), policy_sha256=sha(POLICY),
                                    speed=SPEED, delay=DELAY, summary=summary),
                               ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[write] {dest}')


if __name__ == '__main__':
    main()
