"""仰卧逃逸的动力学测试(第 0 步诊断二,纯 CPU)。

权限图证明仰卧是准静态陷阱:单独关节 ±60° 的**持续**命令只让躯干转 ≤9°。
鲤鱼打挺靠的是甩腿的角动量,不是静态力矩 —— 所以这一轮测**振荡/甩动**。

对每个模式测: 躯干最大转角 |Δtilt|、是否把脚甩到地面、根高抬升。有模式能显著转动躯干
(比如 >25°),说明"甩腿过支点"这条路成立,关键帧要用快甩而不是慢插值;
全都不行则说明仰卧需要换策略(先侧滚到趴姿再起)。
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
from evaluate_policy import sha
from mjlab_microduck.tasks import recovery_bounds

ROOT = Path(__file__).resolve().parent
CONTACT = ROOT / '20260914_contact_motion'
DIRECTIONS = {'left': [0, 1, 0], 'right': [0, -1, 0], 'front': [-1, 0, 0], 'back': [1, 0, 0]}


def trial(plant, direction, seed, case, pattern, amp_deg, freq_hz, seconds=3.0):
    e = configure_owned(MotionExperiment(CALIBRATOR, 'run', 0., case, plant=plant,
                                         operating_envelope=True), (1,))
    constrain_recovery_head(e)
    previous = e.sim.transform_target
    limits = e.sim.model.jnt_range[e.sim.jids]
    s = e.sim
    e.reset(seed)
    prepare_fall(e, DIRECTIONS[direction])
    fallen = s.data.qpos[s.jadr].copy()
    home = np.asarray(s.home, dtype=float).copy()
    holder = {'goal': fallen.copy(), 't0': None}

    def scripted(target, obs):
        g = holder['goal'].copy()
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
    for _ in range(round(.3 / s.dt)):
        e.poll(); s.step(np.zeros(18))
    holder['t0'] = s.data.time
    amp = np.deg2rad(amp_deg)
    tilts, feet_z, rootz = [], [], []
    for _ in range(round(seconds / s.dt)):
        t = s.data.time - holder['t0']
        w = 2 * np.pi * freq_hz
        osc = amp * np.sin(w * t)
        g = home.copy()
        for jn, sign in pattern.items():
            g[s.names.index(jn)] = home[s.names.index(jn)] + sign * osc
        holder['goal'] = g
        e.poll(); s.step(np.zeros(18))
        rot = s.data.xmat[s.body].reshape(3, 3)
        tilts.append(np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1))))
        feet_z.append(float(min(s.data.geom_xpos[g0, 2] for g0 in s.foot_geoms) * 1000))
        rootz.append(float(s.data.qpos[2] * 1000))
        if not np.isfinite(s.data.qpos).all():
            break
    tilts = np.asarray(tilts)
    return dict(start_tilt=float(tilts[0]), tilt_max=float(tilts.max()), tilt_min=float(tilts.min()),
                excursion=float(tilts.max() - tilts.min()), tilt_end=float(tilts[-1]),
                feet_z_min_mm=float(np.min(feet_z)), root_z_max_mm=float(np.max(rootz)))


PATTERNS = {
    'hip': {'left_hip_pitch': 1., 'right_hip_pitch': 1.},
    'hip_knee_anti': {'left_hip_pitch': 1., 'right_hip_pitch': 1., 'left_knee': -1., 'right_knee': -1.},
    'neck': {'neck_pitch': -1.},
    'tail': {'tail_pitch': 1.},
    'hip_tail': {'left_hip_pitch': 1., 'right_hip_pitch': 1., 'tail_pitch': -1.},
    'hip_neck_tail': {'left_hip_pitch': 1., 'right_hip_pitch': 1., 'neck_pitch': -1., 'tail_pitch': -1.},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plant', default=None)
    p.add_argument('--direction', default='back', choices=list(DIRECTIONS))
    p.add_argument('--seed', type=int, default=941)
    p.add_argument('--out', default='dynamic')
    p.add_argument('--seconds', type=float, default=3.0)
    a = p.parse_args()
    selected = json.loads((CONTACT / 'selected_contact.json').read_text())
    plant = CONTACT / (a.plant or selected.get('plant_directory', 'normal_contact_plant'))
    case = HardwareCase(motor_curve=True, voltage=12., physics_dt=.00125, command_ms=10,
                        foot_torsional_friction_m=selected['friction'][1])
    folder = CONTACT / a.out
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"方向 {a.direction}:甩动模式 × 幅值 × 频率  -> 躯干转角(excursion)")
    for name, pat in PATTERNS.items():
        for amp in (40., 70.):
            for f in (1.5, 3.0):
                r = trial(plant, a.direction, a.seed, case, pat, amp, f, a.seconds)
                r.update(pattern=name, amp_deg=amp, freq_hz=f)
                rows.append(r)
                print(f"  {name:14s} {amp:4.0f}° {f:3.1f}Hz  起点 {r['start_tilt']:6.1f}° "
                      f"摆幅 {r['excursion']:5.1f}°  末 {r['tilt_end']:6.1f}°  "
                      f"脚最低 {r['feet_z_min_mm']:6.1f}mm  根高最大 {r['root_z_max_mm']:6.1f}mm", flush=True)
    rows.sort(key=lambda r: -r['excursion'])
    (folder / f'dynamic_{a.direction}.json').write_text(
        json.dumps(dict(plant_sha256=sha(plant / 'nominal.mjb'), hardware=asdict(case),
                        direction=a.direction, rows=rows), indent=2, ensure_ascii=False), encoding='utf-8')
    b = rows[0]
    print(f"\n最大摆幅: {b['pattern']} {b['amp_deg']:.0f}° {b['freq_hz']:.1f}Hz -> "
          f"{b['excursion']:.1f}°(脚最低 {b['feet_z_min_mm']:.1f}mm)")


if __name__ == '__main__':
    main()
