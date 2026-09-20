"""仰卧姿态下的控制权限图(第 0 步的诊断,纯 CPU)。

第一次脚本化关键帧 12 个组合全部原地不动(98-102°),说明我对"哪个关节能推动躯干"
的符号/量级判断是错的。别再猜关键帧了 —— 直接量:从实测倒地姿态出发,给单个关节组
一个常量偏置,看躯干倾角怎么变。这张"权限图"才是设计关键帧的依据。

关键背景(实测):仰卧(180°)时头已经着地(0.0mm),躯干离地 84.8mm,脚翘空 253mm;
颈舵机上限 0.6 N·m、尾 0.226 N·m;头+颈占 21.9% 质量、尾 7.4%、腿 22%、臂 0.8%。
"""
from pathlib import Path
import argparse, json
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
DIRECTIONS = {'left': [0, 1, 0], 'right': [0, -1, 0], 'front': [-1, 0, 0], 'back': [1, 0, 0]}


def measure(plant, direction, seed, case, offsets, hold_s=2.5):
    e = configure_owned(MotionExperiment(CALIBRATOR, 'run', 0., case, plant=plant,
                                         operating_envelope=True), (1,))
    constrain_recovery_head(e)
    previous = e.sim.transform_target
    limits = e.sim.model.jnt_range[e.sim.jids]
    s = e.sim
    e.reset(seed)
    pre, _ = prepare_fall(e, DIRECTIONS[direction])
    fallen = s.data.qpos[s.jadr].copy()
    home = np.asarray(s.home, dtype=float).copy()
    start_tilt = float(np.degrees(np.arccos(np.clip(s.data.xmat[s.body].reshape(3, 3)[2, 2], -1, 1))))

    def make_target(goal_vec):
        def scripted(target, obs):
            g = recovery_bounds.apply_numpy(goal_vec, s.names)
            g = np.clip(g, limits[:, 0] + .07, limits[:, 1] - .07)
            rot = s.data.xmat[s.body].reshape(3, 3)
            phi = recovery_bounds.phi_numpy(rot, s.data.qpos[2])
            g = recovery_bounds.apply_band_numpy(g, s.names, s.home, phi)
            return previous(g, obs)
        return scripted

    s.transform_target = make_target(fallen)      # 先保持倒地姿态
    class Hold:
        def get_inputs(self): return [type('I', (), {'name': 'obs'})()]
        def run(self, *a): return [((s.applied - s.home) / s.scale).astype(np.float32)[None]]
    s.session = Hold()
    for _ in range(round(.3 / s.dt)):
        e.poll(); s.step(np.zeros(18))
    goal = home.copy()
    for n, d in offsets.items():
        goal[s.names.index(n)] = home[s.names.index(n)] + np.deg2rad(d)
    s.transform_target = make_target(goal)
    import mujoco
    floor = s.model.geom('terrain').id
    head_g = s.model.geom('robot/jaw_soft_floor_proxy').id
    tail_g = s.model.geom('robot/tail_pitch_floor_proxy').id
    peak = dict(head=0., tail=0.)
    w = np.zeros(6)
    for _ in range(round(hold_s / s.dt)):
        e.poll(); s.step(np.zeros(18))
        for i, c in enumerate(s.data.contact):
            if c.efc_address < 0 or floor not in (c.geom1, c.geom2):
                continue
            mujoco.mj_contactForce(s.model, s.data, i, w)
            other = c.geom2 if c.geom1 == floor else c.geom1
            if other == head_g: peak['head'] = max(peak['head'], float(w[0]))
            if other == tail_g: peak['tail'] = max(peak['tail'], float(w[0]))
        if not np.isfinite(s.data.qpos).all():
            break
    rot = s.data.xmat[s.body].reshape(3, 3)
    tilt = float(np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1))))
    q = np.rad2deg(s.data.qpos[s.jadr])
    return dict(start_tilt=start_tilt, tilt=tilt, delta_tilt=tilt - start_tilt,
                root_z_mm=float(s.data.qpos[2] * 1000),
                yaw_deg=float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))),
                head_peak_N=peak['head'], tail_peak_N=peak['tail'],
                feet_mm=[float(v * 1000) for v in s.data.geom_xpos[s.foot_geoms, 2]],
                knee_deg=float(q[s.names.index('left_knee')]),
                hip_deg=float(q[s.names.index('left_hip_pitch')]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plant', default=None)
    p.add_argument('--direction', default='back', choices=list(DIRECTIONS))
    p.add_argument('--seed', type=int, default=941)
    p.add_argument('--out', default='authority')
    a = p.parse_args()
    selected = json.loads((CONTACT / 'selected_contact.json').read_text())
    plant = CONTACT / (a.plant or selected.get('plant_directory', 'normal_contact_plant'))
    case = HardwareCase(motor_curve=True, voltage=12., physics_dt=.00125, command_ms=10,
                        foot_torsional_friction_m=selected['friction'][1])
    folder = CONTACT / a.out
    folder.mkdir(parents=True, exist_ok=True)
    TESTS = [
        ('baseline(hold fallen)', {}),
        ('hip_pitch +60', {'left_hip_pitch': 60., 'right_hip_pitch': 60.}),
        ('hip_pitch -60', {'left_hip_pitch': -60., 'right_hip_pitch': -60.}),
        ('knee -80 (fold)', {'left_knee': -80., 'right_knee': -80.}),
        ('knee +60 (extend)', {'left_knee': 60., 'right_knee': 60.}),
        ('neck_pitch +60 (head down/back)', {'neck_pitch': 60.}),
        ('neck_pitch -60 (head tuck)', {'neck_pitch': -60.}),
        ('tail_pitch +60', {'tail_pitch': 60.}),
        ('tail_pitch -45 (tail press)', {'tail_pitch': -45.}),
        ('legs to HOME', {'left_hip_pitch': -26., 'right_hip_pitch': 26.,
                          'left_knee': 0., 'right_knee': 0.,
                          'left_ankle': 26., 'right_ankle': -26.}),
    ]
    rows = []
    for name, off in TESTS:
        r = measure(plant, a.direction, a.seed, case, off)
        r['test'] = name
        r['offsets'] = off
        rows.append(r)
        print(f"{name:34s} Δtilt {r['delta_tilt']:+7.1f}°  终tilt {r['tilt']:6.1f}°  "
              f"根高 {r['root_z_mm']:6.1f}mm  头峰值 {r['head_peak_N']:5.2f}N  "
              f"尾峰值 {r['tail_peak_N']:5.2f}N  脚z {np.round(r['feet_mm'],0)}", flush=True)
    (folder / f'authority_{a.direction}.json').write_text(
        json.dumps(dict(plant_sha256=sha(plant / 'nominal.mjb'), hardware=asdict(case),
                        direction=a.direction, rows=rows), indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\n-> {folder / f"authority_{a.direction}.json"}')


if __name__ == '__main__':
    main()
