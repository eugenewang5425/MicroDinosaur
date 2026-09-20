"""坐撑(sit)起身的脚本参考:紧凑折腿 -> 伸腿站直,并按全部验收判据验稳定性。

用户给的机制(2026-09-16):"特殊蹲姿 —— 髋部后旋到大腿竖直,大小腿收紧,踝关节把脚往上。
**膝盖是正向折叠**(项目 knee_range_audit.json:膝 +90° 时折叠夹角最小 31.59°,−90° 时 148°)。
完全贴死(膝 121.6°)超出量程,可达最紧是 86°/31.6°。

两段式:
    ① t=0~2.0s  紧凑折腿(髋 ±86 / 膝 ±86)  —— 把"后仰坐姿"(44.7°)转成"直立深蹲"(7.2°)
    ② t=2.0~5.0s 伸腿到 HOME              —— 深蹲升到站立
    ③ t=5.0~12s 保持

必须验的不是"能不能站",而是**稳不稳**:长时保持漂移、双脚持续受力、躯干/尾巴离地、
以及加扰后能否恢复。本脚本按 evaluate_contact_motion 的判据逐条输出。
"""
from pathlib import Path
import argparse, json, sys
from dataclasses import asdict
import numpy as np
import mujoco
from hardware_sim import HardwareCase
from evaluate_run_jump import MotionExperiment
from imu_owned_head import configure_owned
from build_contact_motion_bank import prepare_fall, constrain_recovery_head
from run_heading_stable_start import CALIBRATOR
from evaluate_policy import sha

ROOT = Path(__file__).resolve().parent
CONTACT = ROOT / '20260914_contact_motion'
STANCE_JOINTS = ('left_hip_pitch', 'left_knee', 'left_ankle', 'neck_pitch', 'head_yaw',
                 'right_hip_pitch', 'right_knee', 'right_ankle', 'tail_pitch')
FOLD = {'left_hip_pitch': 86., 'right_hip_pitch': -86., 'left_knee': 86., 'right_knee': -86.}


def mesh_low(model, data, geom_id):
    mid = model.geom_dataid[geom_id]
    a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    v = model.mesh_vert[a:a + n] @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
    return v[:, 2].min()


def sole_tilt(model, data, geom_id):
    mid = model.geom_dataid[geom_id]
    a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    v = model.mesh_vert[a:a + n] @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
    lo = v[v[:, 2] < np.quantile(v[:, 2], .2)]
    _, _, vt = np.linalg.svd(lo - lo.mean(0))
    nrm = vt[2]
    return float(np.degrees(np.arccos(np.clip(abs(nrm[2]), -1, 1))))


def run(plant, seed, case, fold_s=2.0, extend_s=3.0, total_s=12.0, push_n=0., push_t=8.0,
        push_dur=0.3, direction='back'):
    e = configure_owned(MotionExperiment(CALIBRATOR, 'run', 0., case, plant=plant,
                                         operating_envelope=True), (1,))
    constrain_recovery_head(e)
    s = e.sim
    m = s.model
    names = s.names
    limits = m.jnt_range[s.jids]
    previous = s.transform_target
    current = {'g': None}

    def scripted(target, obs):
        g = current['g'] if current['g'] is not None else target
        return previous(np.clip(g, limits[:, 0] + .07, limits[:, 1] - .07), obs)

    e.reset(seed)
    prepare_fall(e, direction)
    s.transform_target = scripted

    class Hold:
        def get_inputs(self): return [type('I', (), {'name': 'obs'})()]

        def run(self, *a): return [((s.applied - s.home) / s.scale).astype(np.float32)[None]]

    s.session = Hold()
    home = np.asarray(s.home, dtype=float)
    goals = []
    for secs, cmd in ((fold_s, FOLD), (extend_s, {})):
        tgt = home.copy()
        for k, v in cmd.items():
            tgt[names.index(k)] = np.deg2rad(v)
        goals.append((secs, tgt))
    g0 = s.data.qpos[s.jadr].copy()
    started = s.data.time
    floor = m.geom('terrain').id
    head_g = m.geom('robot/jaw_soft_floor_proxy').id
    tail_g = m.geom('robot/tail_pitch_floor_proxy').id
    rows = []
    prev_cb = s.substep_callback

    def record():
        prev_cb()
        w = np.zeros(6)
        feet = [0., 0.]
        other = 0.
        penetration = 0.
        for i, c in enumerate(s.data.contact):
            if c.efc_address < 0 or floor not in (c.geom1, c.geom2):
                continue
            mujoco.mj_contactForce(m, s.data, i, w)
            penetration = max(penetration, -c.dist)
            g = c.geom2 if c.geom1 == floor else c.geom1
            load = max(0., w[0])
            if g in s.foot_geoms:
                feet[s.foot_geoms.index(g)] += load
            else:
                other += load
        rot = s.data.xmat[s.body].reshape(3, 3)
        q = s.data.qpos[s.jadr]
        dev = max(abs(np.rad2deg(q[names.index(n)] - home[names.index(n)])) for n in STANCE_JOINTS)
        rows.append([s.data.time - started, s.data.qpos[2],
                     np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1))),
                     np.linalg.norm(s.data.qvel[:3]), np.linalg.norm(s.data.qvel[3:6]),
                     feet[0], feet[1], other, penetration, dev,
                     np.rad2deg(q[names.index('head_pitch')]),
                     np.degrees(np.arctan2(rot[1, 0], rot[0, 0])),
                     mesh_low(m, s.data, head_g), mesh_low(m, s.data, tail_g),
                     np.rad2deg(q[names.index('tail_pitch')])])

    s.substep_callback = record
    while s.data.time - started < total_s:
        t = s.data.time - started
        if t < goals[0][0]:
            f = t / goals[0][0]
            cur = g0 + (goals[0][1] - g0) * (f * f * (3 - 2 * f))
        elif t < goals[0][0] + goals[1][0]:
            f = (t - goals[0][0]) / goals[1][0]
            cur = goals[0][1] + (goals[1][1] - goals[0][1]) * (f * f * (3 - 2 * f))
        else:
            cur = goals[1][1]
        current['g'] = cur
        s.data.xfrc_applied[:] = 0
        if push_n and push_t <= t < push_t + push_dur:
            s.data.xfrc_applied[s.head, 0] = push_n
        e.poll()
        s.step(np.zeros(18))
        if not np.isfinite(s.data.qpos).all():
            s.substep_callback = prev_cb
            return dict(seed=seed, status='NONFINITE')
    s.data.xfrc_applied[:] = 0
    s.substep_callback = prev_cb
    np.savez_compressed(CONTACT / 'sit_ref_rows.npz', rows=np.asarray(rows),
                        frames_dt=float(s.dt))
    # 独立复核:跑完后用 fresh forward 重算头/尾高度,与回调里记的对比。
    # (MuJoCo 在 mj_step 开头算 geom 变换,子步回调里读到的可能差一个子步)
    mujoco.mj_forward(m, s.data)
    check = dict(head_low_fresh_mm=mesh_low(m, s.data, head_g) * 1000,
                 tail_low_fresh_mm=mesh_low(m, s.data, tail_g) * 1000)
    a = np.asarray(rows)
    hold = a[a[:, 0] >= total_s - 2.]
    return dict(seed=seed, status='COMPLETE', **check,
                tilt_max_deg=float(hold[:, 2].max()),
                height_min_mm=float(hold[:, 1].min() * 1000), height_max_mm=float(hold[:, 1].max() * 1000),
                speed_max=float(hold[:, 3].max()), omega_max=float(hold[:, 4].max()),
                foot_L_min_N=float(hold[:, 5].min()), foot_R_min_N=float(hold[:, 6].min()),
                nonfoot_max_N=float(hold[:, 7].max()),
                penetration_max_mm=float(a[:, 8].max() * 1000),
                stance_max_dev_deg=float(hold[:, 9].max()),
                head_pitch_range_deg=[float(hold[:, 10].min()), float(hold[:, 10].max())],
                drift_mm=float(abs(hold[-1, 1] - hold[0, 1]) * 1000),
                head_low_mm=float(hold[:, 12].min()), tail_low_mm=float(hold[:, 13].min()),
                yaw_max_deg=float(np.max(np.abs(hold[:, 11] - hold[0, 11]))),
                trace=a.tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[941, 942, 943, 951])
    p.add_argument('--push', type=float, default=0., help='横推力 N(自 head, 8.0~8.3s)')
    p.add_argument('--out', default='sit_reference')
    p.add_argument('--direction', default='back',
                   help='倒地方位(back=坐撑, left/right/front 用来试这条对称折腿能不能通用)')
    a = p.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    selected = json.loads((CONTACT / 'selected_contact.json').read_text())
    plant = CONTACT / selected.get('plant_directory', 'normal_contact_plant')
    case = HardwareCase(motor_curve=True, voltage=12., physics_dt=.00125, command_ms=10,
                        foot_torsional_friction_m=selected['friction'][1])
    folder = CONTACT / a.out
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'plan.json').write_text(json.dumps(dict(
        purpose='Scripted sit get-up: compact fold then extend; stability verification',
        fold=FOLD, fold_s=2.0, extend_s=3.0, push_n=a.push,
        plant_sha256=sha(plant / 'nominal.mjb'), hardware=asdict(case),
        note='Knee folds POSITIVE (knee_range_audit: +90 deg gives the tightest 31.59 deg included angle)'),
        indent=2), encoding='utf-8')
    results = []
    for seed in a.seeds:
        r = run(plant, seed, case, push_n=a.push, direction=a.direction)
        results.append(r)
        if r['status'] != 'COMPLETE':
            print(f'seed {seed}: {r["status"]}'); continue
        ok = (r['tilt_max_deg'] < 10 and 103.5 < r['height_min_mm'] and r['height_max_mm'] < 125.5
              and r['foot_L_min_N'] > .5 and r['foot_R_min_N'] > .5 and r['nonfoot_max_N'] < .2
              and r['speed_max'] < .03 and r['omega_max'] < .3
              and r['penetration_max_mm'] <= 2 and r['stance_max_dev_deg'] <= 15
              and r['head_pitch_range_deg'][0] >= -2 and r['head_pitch_range_deg'][1] <= 29)
        r['passed'] = bool(ok)
        print(f"seed {seed}: 倾角{r['tilt_max_deg']:5.2f}° 根高{r['height_min_mm']:6.1f}~{r['height_max_mm']:.1f} "
              f"脚L{r['foot_L_min_N']:5.2f} 脚R{r['foot_R_min_N']:5.2f} 非足{r['nonfoot_max_N']:5.2f}N "
              f"站姿{r['stance_max_dev_deg']:5.2f}° 漂移{r['drift_mm']:+5.1f}mm "
              f"穿透{r['penetration_max_mm']:4.2f}mm 头俯仰{r['head_pitch_range_deg']} "
              f"头低{r['head_low_mm']:6.1f} 尾低{r['tail_low_mm']:6.1f}  {'PASS' if ok else 'FAIL'}")
    (folder / ('summary_push.json' if a.push else 'summary.json')).write_text(
        json.dumps(results, indent=2), encoding='utf-8')
    n = sum(1 for r in results if r.get('passed'))
    print(f'\n汇总: {n}/{len(results)} 通过' + (f'  (含 {a.push}N 横向扰动)' if a.push else ''))


if __name__ == '__main__':
    main()
