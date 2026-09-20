"""脚本化起身参考轨迹可行性验证(第 0 步,纯 CPU,不烧 GPU)。

按 GETUP_MOTION_PLAN.md 第一节的几何结论设计关键帧:

    仰卧时头已经着地(0.0mm) -> 腿向前上摆、把重心甩到头支点前方
    -> 腿落下、脚在 90° 俯仰附近踩地 -> 伸腿站直、尾巴抬起、头回到 +20°

本脚本**只驱动位置目标**(与验收同一条动作通路:recovery_bounds -> jnt_range±0.07
夹取 -> 站位包络 -> MotionExperiment 的踝/颌包络 -> HeadExperiment 的头 IMU 适配器),
躯干怎么翻由物理决定 —— 这正是要验的东西。奖励学不出来的相(P1 以头为支点、P2 荡腿),
先用脚本证明"物理上做得到",做得到就把关键帧变成训练用的参考表(与蹲技能同构)。

用法:
    python try_getup_reference.py --direction back --out ref_probe
    python try_getup_reference.py --direction back --out ref_probe --sweep
"""
from pathlib import Path
import argparse, json, itertools, time
from dataclasses import asdict
import numpy as np
from hardware_sim import HardwareCase
from evaluate_run_jump import MotionExperiment
from imu_owned_head import configure_owned
from build_contact_motion_bank import prepare_fall, constrain_recovery_head
from run_heading_stable_start import CALIBRATOR
from heading_sim import WARMUP_SECONDS
from evaluate_policy import sha
from mjlab_microduck.tasks import recovery_bounds

ROOT = Path(__file__).resolve().parent
CONTACT = ROOT / '20260914_contact_motion'
COLUMNS = ['time', 'root_z', 'tilt_deg', 'speed', 'omega', 'left_N', 'right_N', 'nonfoot_N',
           'head_N', 'tail_N', 'penetration_m', 'limit_excess_rad', 'neck_load_Nm']
DIRECTIONS = {'left': [0, 1, 0], 'right': [0, -1, 0], 'front': [-1, 0, 0], 'back': [1, 0, 0]}


def build_reference(names, home, fallen, kf):
    """关键帧 -> 绝对关节角轨迹。kf 是 [(t_start, {关节: 角}), ...],末帧之后保持。"""
    home = np.asarray(home, dtype=float)
    keys = []
    for t0, over in kf:
        v = home.copy()
        for n, a in over.items():
            if n in names:
                v[names.index(n)] = a
        keys.append((t0, v))
    # 首帧用实测倒地姿态,避免目标阶跃
    keys[0] = (keys[0][0], np.asarray(fallen, dtype=float).copy())

    def smooth(s):
        s = min(max(s, 0.), 1.)
        return s * s * (3 - 2 * s)

    def goal(t):
        if t <= keys[0][0]:
            return keys[0][1].copy()
        for i in range(len(keys) - 1):
            t0, v0 = keys[i]
            t1, v1 = keys[i + 1]
            if t0 <= t < t1:
                return v0 + (v1 - v0) * smooth((t - t0) / max(1e-6, t1 - t0))
        return keys[-1][1].copy()
    return goal


def default_keyframes(p):
    """p 是参数字典;角度单位度。"""
    D = np.deg2rad
    kf = []
    kf.append((0.0, {}))
    # P1 支点相:腿向前上摆(髋伸展=腿甩向头侧)、膝伸直做长杠杆;尾巴下压接力;脖子顶住
    kf.append((p['t_swing'], {
        'left_hip_pitch': D(p['swing_hip']), 'right_hip_pitch': D(p['swing_hip']),
        'left_knee': D(p['swing_knee']), 'right_knee': D(p['swing_knee']),
        'left_ankle': D(p['swing_ankle']), 'right_ankle': D(p['swing_ankle']),
        'tail_pitch': D(p['tail_plant']),
        'neck_pitch': D(p['neck_pivot']),
    }))
    # P2 收腿相:腿落下、脚踩到重心下方(膝屈)
    kf.append((p['t_plant'], {
        'left_hip_pitch': D(p['plant_hip']), 'right_hip_pitch': D(p['plant_hip']),
        'left_knee': D(p['plant_knee']), 'right_knee': D(p['plant_knee']),
        'left_ankle': D(p['plant_ankle']), 'right_ankle': D(p['plant_ankle']),
        'tail_pitch': D(p['tail_plant'] * .5),
        'neck_pitch': D(p['neck_release']),
    }))
    # P3 站立相:伸腿站直、尾巴抬起、头回 +20
    kf.append((p['t_stand'], {
        'left_hip_pitch': D(p['stand_hip']), 'right_hip_pitch': D(p['stand_hip']),
        'left_knee': D(p['stand_knee']), 'right_knee': D(p['stand_knee']),
        'left_ankle': D(p['stand_ankle']), 'right_ankle': D(p['stand_ankle']),
        'tail_pitch': 0.0, 'neck_pitch': D(20.0),
    }))
    kf.append((p['t_hold'], {}))     # 全部回 HOME
    return kf


BASE = dict(t_swing=.7, t_plant=1.7, t_stand=3.0, t_hold=4.2,
            swing_hip=60., swing_knee=0., swing_ankle=0., tail_plant=-40., neck_pivot=20.,
            plant_hip=-30., plant_knee=-60., plant_ankle=10., neck_release=10.,
            stand_hip=-26., stand_knee=0., stand_ankle=26.)


def run_case(plant, direction, seed, case, params, folder, tag, policy_onnx=None):
    e = configure_owned(MotionExperiment(CALIBRATOR, 'run', 0., case, plant=plant,
                                         operating_envelope=True), (1,))
    constrain_recovery_head(e)
    previous = e.sim.transform_target
    limits = e.sim.model.jnt_range[e.sim.jids]
    s = e.sim
    result = dict(key=tag, direction=direction, seed=seed, params=params,
                  plant_sha256=sha(plant / 'nominal.mjb'), hardware=asdict(case))
    e.reset(seed)
    pre, _ = prepare_fall(e, DIRECTIONS[direction])
    result['start_tilt_deg'] = float(pre[-1, 2])
    result['fallen_start_valid'] = bool(pre[-1, 2] > 25 and pre[-1, 3] < .05
                                        and pre[-1, 4] < .5 and pre[-1, 5] > 1 and pre[-1, 6] < .002)
    fallen = s.data.qpos[s.jadr].copy()
    goal = build_reference(s.names, s.home, fallen, default_keyframes(params))
    t0 = s.data.time

    def scripted(target, obs):
        g = goal(s.data.time - t0)
        g = recovery_bounds.apply_numpy(g, s.names)
        g = np.clip(g, limits[:, 0] + .07, limits[:, 1] - .07)
        rot = s.data.xmat[s.body].reshape(3, 3)
        phi = recovery_bounds.phi_numpy(rot, s.data.qpos[2])
        g = recovery_bounds.apply_band_numpy(g, s.names, s.home, phi)
        return previous(g, obs)

    s.transform_target = scripted
    class Hold:
        def get_inputs(self): return [type('I', (), {'name': 'obs'})()]
        def run(self, *a): return [((s.applied - s.home) / s.scale).astype(np.float32)[None]]
    s.session = Hold()

    floor = s.model.geom('terrain').id
    head_geom = s.model.geom('robot/jaw_soft_floor_proxy').id
    tail_geom = s.model.geom('robot/tail_pitch_floor_proxy').id
    rows = []
    prev_cb = s.substep_callback

    def record():
        prev_cb()
        import mujoco
        w = np.zeros(6)
        head_n = tail_n = 0.
        for i, c in enumerate(s.data.contact):
            if c.efc_address < 0 or floor not in (c.geom1, c.geom2):
                continue
            mujoco.mj_contactForce(s.model, s.data, i, w)
            other = c.geom2 if c.geom1 == floor else c.geom1
            if other == head_geom: head_n += max(0., w[0])
            if other == tail_geom: tail_n += max(0., w[0])
        rot = s.data.xmat[s.body].reshape(3, 3)
        q = s.data.qpos[s.jadr]
        excess = float(np.maximum(limits[:, 0] - q, q - limits[:, 1]).clip(0).max())
        rows.append([s.data.time - WARMUP_SECONDS, s.data.qpos[2],
                     np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1))),
                     np.linalg.norm(s.data.qvel[:3]), np.linalg.norm(s.data.qvel[3:6]),
                     0., 0., 0., head_n, tail_n, 0., excess,
                     float(abs(s.data.actuator_force[s.aids[s.names.index('neck_pitch')]]))])

    s.substep_callback = record
    for _ in range(round(12.0 / s.dt)):
        e.poll(); s.step(np.zeros(18))
        if not np.isfinite(s.data.qpos).all():
            result.update(status='NONFINITE'); return result
    s.substep_callback = prev_cb
    a = np.asarray(rows)
    tail = a[a[:, 0] >= 10.0]
    q = s.data.qpos[s.jadr]
    rot = s.data.xmat[s.body].reshape(3, 3)
    result.update(status='COMPLETE',
                  final_tilt_deg=float(tail[-1, 2]), final_height_mm=float(tail[-1, 1] * 1000),
                  final_speed=float(np.linalg.norm(s.data.qvel[:3])),
                  neck_torque_peak_Nm=float(np.abs(a[:, 12]).max()),
                  head_force_peak_N=float(a[:, 8].max()), tail_force_peak_N=float(a[:, 9].max()),
                  head_touch_fraction=float((a[:, 8] > .2).mean()),
                  tail_touch_fraction=float((a[:, 9] > .2).mean()),
                  joint_limit_excess_deg=float(np.rad2deg(a[:, 11].max())),
                  final_joint_deg={n: float(np.rad2deg(q[i])) for i, n in enumerate(s.names)},
                  final_yaw_deg=float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))))
    np.savez_compressed(folder / f'{tag}.npz', trace=a, columns=np.array(COLUMNS),
                        qpos=np.asarray(e.qpos_frames))
    return result


def score(r):
    """越接近"站起来"分数越高;没站起来给很低的基线。仅用于排序脚本化候选。"""
    if r.get('status') != 'COMPLETE':
        return 1e9
    tilt = abs(r['final_tilt_deg'])
    h = abs(r['final_height_mm'] - 114.0)
    return tilt * 2.0 + h * 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plant', default=None)
    p.add_argument('--direction', default='back', choices=list(DIRECTIONS))
    p.add_argument('--seed', type=int, default=941)
    p.add_argument('--delay', type=int, default=10)
    p.add_argument('--out', default='ref_probe')
    p.add_argument('--sweep', action='store_true')
    a = p.parse_args()
    selected = json.loads((CONTACT / 'selected_contact.json').read_text())
    plant = CONTACT / (a.plant or selected.get('plant_directory', 'normal_contact_plant'))
    case = HardwareCase(motor_curve=True, voltage=12., physics_dt=.00125, command_ms=a.delay,
                        foot_torsional_friction_m=selected['friction'][1])
    folder = CONTACT / a.out
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'plan.json').write_text(json.dumps(dict(
        purpose='Scripted reference feasibility for the pivot-and-swing get-up (CPU only)',
        arguments={k: str(v) for k, v in vars(a).items()}, plant_sha256=sha(plant / 'nominal.mjb'),
        hardware=asdict(case), base_params=BASE,
        note='Keyframes from GETUP_MOTION_PLAN.md; physics decides the trunk rotation.'), indent=2),
        encoding='utf-8')
    grid = [dict(BASE)]
    if a.sweep:
        grid = []
        for swing_hip in (40., 70., 100.):
            for swing_knee in (0., -30.):
                for tail_plant in (-40., 0.):
                    grid.append(dict(BASE, swing_hip=swing_hip, swing_knee=swing_knee,
                                     tail_plant=tail_plant))
    results = []
    for i, params in enumerate(grid):
        tag = f'{a.direction}_s{a.seed}_c{a.delay}_k{i:02d}'
        t = time.time()
        r = run_case(plant, a.direction, a.seed, case, params, folder, tag)
        r['score'] = score(r)
        results.append(r)
        (folder / f'{tag}.json').write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(dict(tag=tag, swing_hip=params['swing_hip'], swing_knee=params['swing_knee'],
                              tail_plant=params['tail_plant'], status=r.get('status'),
                              tilt=round(r.get('final_tilt_deg', -1), 1),
                              h=round(r.get('final_height_mm', -1), 1),
                              neck_peak=round(r.get('neck_torque_peak_Nm', -1), 3),
                              head_pk=round(r.get('head_force_peak_N', -1), 2),
                              tail_pk=round(r.get('tail_force_peak_N', -1), 2),
                              yaw=round(r.get('final_yaw_deg', 0), 0),
                              dt=round(time.time() - t, 1))), flush=True)
    best = min(results, key=lambda r: r['score'])
    (folder / 'summary.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nBEST', json.dumps({k: best.get(k) for k in
          ('key', 'score', 'status', 'final_tilt_deg', 'final_height_mm', 'neck_torque_peak_Nm',
           'head_force_peak_N', 'tail_force_peak_N', 'final_yaw_deg')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
