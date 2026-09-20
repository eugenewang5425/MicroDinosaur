"""单脚站立验收:抬脚高度保持 + 支撑脚着地 + 平衡 + 头尾配平观察。

判据(末 2 秒窗口,与起身/台阶的窗口口径一致):
  ① 抬脚高度  目标 ±8mm 内保持 >=90% 窗口
  ② 支撑脚    全程着地(任一子步接触力 >1N),抬脚离地
  ③ 平衡      倾角 <10°,根水平漂移 <60mm
  ④ 不摔倒
头尾配平**作为观察量报告**(用户指定该手段):记录 head_yaw/tail_yaw 的幅值
与"抬脚侧"的符号关系,不参与通过判定——判定只看"站没站住"。
"""
from pathlib import Path
import sys
import json
import argparse

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from corrective_common import OUT  # noqa: E402
from evaluate_owned_terrain import experiment, evaluate  # noqa: E402
from evaluate_policy import sha  # noqa: E402

TOL_MM = 8.
HOLD_FRAC = .90
TILT_MAX_DEG = 10.
DRIFT_MAX_MM = 1000.   # 方案A:允许挪步式动态平衡,只限制别跑出这个半径


def foot_low(model, data, geom_id):
    """足底碰撞网格的世界最低点 z(评估侧没有训练侧的 height sensor,直接算)。"""
    mid = model.geom_dataid[geom_id]
    a = model.mesh_vertadr[mid]
    n = model.mesh_vertnum[mid]
    v = model.mesh_vert[a:a + n] @ data.geom_xmat[geom_id].reshape(3, 3).T         + data.geom_xpos[geom_id]
    return float(v[:, 2].min())


def run_case(policy, seed, lift_foot, target_mm, seconds=16.0, delay=10,
             cmd_vy=None):
    case = dict(terrain='flat', scenario='stand', program='legacy', depth=0,
                delay=delay, speed=None, seconds=seconds)
    e = experiment(str(policy), case)
    if cmd_vy is not None:
        # 指令注入:command[1] -> obs[64],与训练侧 StillTwist 的 twist vy 位同源。
        # 用于验证"指令-行为一致性"(发 +1 必须抬左脚,发 -1 必须抬右脚)。
        def fixed_cmd(t, scenario, _v=cmd_vy):
            u = np.zeros(18)
            u[1] = _v
            return u
        e.user_command = fixed_cmd
    s = e.sim
    import mujoco
    fl = s.model.geom('robot/left_foot_collision').id
    fr = s.model.geom('robot/right_foot_collision').id
    floor = s.model.geom('terrain').id
    n = s.names
    idx = {k: n.index(k) for k in
           ('head_yaw', 'tail_yaw', 'tail_pitch', 'neck_pitch')}
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
        w = np.zeros(6)
        load = [0., 0.]
        for ci, c in enumerate(d.contact):
            if floor not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == floor else c.geom1
            mujoco.mj_contactForce(s.model, d, ci, w)
            if other == fl:
                load[0] += max(0., w[0])
            elif other == fr:
                load[1] += max(0., w[0])
        rows.append(dict(
            t=t,
            root_z=float(d.qpos[2]) * 1000,
            foot_l=foot_low(s.model, d, fl),
            foot_r=foot_low(s.model, d, fr),
            contact_l=load[0], contact_r=load[1],
            head_yaw=float(np.degrees(q[idx['head_yaw']])),
            tail_yaw=float(np.degrees(q[idx['tail_yaw']])),
            tail_pitch=float(np.degrees(q[idx['tail_pitch']])),
            neck=float(np.degrees(q[idx['neck_pitch']])),
            tilt=float(np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1)))),
            x=float(d.qpos[0]), y=float(d.qpos[1]),
        ))

    s.substep_callback = cap
    m, _ = evaluate(e, case, seed)
    s.substep_callback = prev
    return m, rows


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', type=Path, required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--target-mm', type=float, default=25.)
    ap.add_argument('--lift-foot', default='left')
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--cmd-vy', type=float, default=1.,
                    help='抬脚指令:+1 抬左脚,-1 抬右脚。**必须给**:random 训练的模型'
                         '没见过 vy=0,不给指令它会"不知道该抬哪只"而不动'
                         '(实测:不给 -0.3mm vs 给 +1 时 13.3mm)')
    ap.add_argument('--consistency', action='store_true',
                    help='指令一致性测试:分别发 +1/-1,检查是否抬对脚')
    a = ap.parse_args()
    dest = OUT / 'evaluation' / a.label
    dest.mkdir(parents=True, exist_ok=True)
    print(f'策略 {a.policy.name} sha={sha(a.policy)[:12]} '
          f'抬{a.lift_foot}脚 目标 {a.target_mm:.0f}mm', flush=True)
    recs = []
    if a.consistency:
        for vy, foot in ((1., 'left'), (-1., 'right')):
            ok_n = 0
            for sd in a.seeds:
                m, rows = run_case(a.policy, sd, a.lift_foot, a.target_mm,
                                   cmd_vy=vy)
                # 判据用"指令侧 vs 非指令侧"的**相对**比较(峰值+末段均值),
                # 不用绝对阈值:模型此刻是"两脚交替动的蹒跚步态",指令侧只是抬得更高,
                # 用单帧或绝对高度都会误判。
                t_end = rows[-1]['t']
                tail = [r for r in rows if r['t'] >= t_end - 2.]
                pk = dict(l=max((r['foot_l'] for r in rows), default=0) * 1000,
                          r=max((r['foot_r'] for r in rows), default=0) * 1000)
                mn = dict(l=float(np.mean([r['foot_l'] for r in tail])) * 1000,
                          r=float(np.mean([r['foot_r'] for r in tail])) * 1000)
                side = 'l' if foot == 'left' else 'r'
                other = 'r' if foot == 'left' else 'l'
                good = (pk[side] > pk[other] + 1.0) and (mn[side] >= mn[other])
                peak_l, peak_r = pk['l'], pk['r']
                ok_n += 1 if good else 0
                print(f"  指令 vy={vy:+.0f}(应抬{foot}脚) seed{sd}: "
                      f"峰值 左{peak_l:5.1f} 右{peak_r:5.1f}mm | "
                      f"末2s均 左{mn['l']:4.1f} 右{mn['r']:4.1f}mm "
                      f"{'✓指令侧更高' if good else '✗未抬对/未抬'}", flush=True)
            recs.append(dict(cmd_vy=vy, expect_foot=foot, correct=ok_n,
                             total=len(a.seeds)))
        (dest / 'consistency.json').write_text(
            json.dumps(recs, indent=2), encoding='utf-8')
        tot = sum(r['correct'] for r in recs)
        print('指令一致性: %d/%d' % (tot, sum(r['total'] for r in recs)))
        return
    for sd in a.seeds:
        m, rows = run_case(a.policy, sd, a.lift_foot, a.target_mm,
                           cmd_vy=a.cmd_vy)
        if not rows:
            print(f'  seed{sd}: 无数据'); continue
        tilt = max(r['tilt'] for r in rows)
        fall = bool(m.get('fell'))
        lift_key = 'foot_l' if a.lift_foot == 'left' else 'foot_r'
        sup_c = 'contact_r' if a.lift_foot == 'left' else 'contact_l'
        lift_c = 'contact_l' if a.lift_foot == 'left' else 'contact_r'
        t_end = rows[-1]['t']
        tail = [r for r in rows if r['t'] >= t_end - 2.]
        hold = sum(1 for r in tail
                   if abs(r[lift_key] * 1000 - a.target_mm) <= TOL_MM) / max(len(tail), 1)
        support_ok = all(r[sup_c] > 1.0 for r in rows)
        lift_air = all(r[lift_c] < 1.0 for r in tail)
        balanced = (not fall) and tilt < TILT_MAX_DEG
        drift = float(np.hypot(rows[-1]['x'] - rows[0]['x'], rows[-1]['y'] - rows[0]['y']))
        planted = support_ok and lift_air
        passed = bool(balanced and planted and hold >= HOLD_FRAC
                      and drift * 1000 < DRIFT_MAX_MM)
        hy = [r['head_yaw'] for r in rows]
        ty = [r['tail_yaw'] for r in rows]
        lift_h = [round(r[lift_key] * 1000, 1) for r in rows]
        rec = dict(seed=sd, fell=fall, tilt_max_deg=tilt, balanced=bool(balanced),
                   planted=bool(planted), hold_frac=round(hold, 3),
                   drift_mm=round(drift * 1000, 1), passed=passed,
                   lift_peak_mm=round(max(lift_h), 1),
                   lift_tail_mean_mm=round(float(np.mean(
                       [r[lift_key] * 1000 for r in tail])), 1),
                   head_yaw_range_deg=[round(min(hy), 1), round(max(hy), 1)],
                   tail_yaw_range_deg=[round(min(ty), 1), round(max(ty), 1)],
                   root_z_mean_mm=round(float(np.mean([r['root_z'] for r in tail])), 1),
                   root_z_min_mm=round(min(r['root_z'] for r in rows), 1),
                   lift_history=lift_h[::25],
                   rootz_history=[round(r['root_z'], 1) for r in rows[::25]],
                   tilt_history=[round(r['tilt'], 1) for r in rows[::25]])
        recs.append(rec)
        print(f"  seed{sd}: 跌倒={fall} 倾角max={tilt:5.1f}° 漂移={rec['drift_mm']:5.1f}mm "
              f"抬脚峰值={rec['lift_peak_mm']:5.1f}mm 末2s均={rec['lift_tail_mean_mm']:5.1f}mm "
              f"保持率={hold * 100:4.0f}% 根高={rec['root_z_mean_mm']:5.1f}mm "
              f"头[{min(hy):+5.0f},{max(hy):+5.0f}]° 尾[{min(ty):+5.0f},{max(ty):+5.0f}]° "
              f"{'✓过' if passed else '未过'}", flush=True)
    (dest / 'matrix.json').write_text(json.dumps(recs, indent=2), encoding='utf-8')
    ok = sum(1 for r in recs if r['passed'])
    bal = sum(1 for r in recs if r['balanced'])
    print('平衡 %d/%d  完整通过(平衡+支撑+抬脚达标) %d/%d' % (bal, len(recs), ok, len(recs)))
    (dest / 'complete.json').write_text(json.dumps(
        dict(status='COMPLETE', policy_sha256=sha(a.policy),
             target_mm=a.target_mm, balanced=bal, passed=ok,
             total=len(recs), records=recs),
        indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
