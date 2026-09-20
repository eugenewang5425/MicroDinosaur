"""Interactive skill console on the SPECIALISTS' own plants (Tk window).

Why this exists (2026-09-15, user: "做"): the demo window in
D:/microduck_rl/scripts/microdinosaur_play.py runs the walk policy on a
hand-built 5 ms plant, which proved unfaithful (delay-sensitive behaviour).
The specialists were trained and validated on THIS research stack: 1.25 ms
physics, motor curve, 10/20/20 ms delays, 6 s stationary calibration, body-IMU
heading outer loop and head-IMU attitude compensation.  This console drives
that full stack interactively instead of approximating it.

Per policy-step loop for the head-stack policies (identical to
heading_sim.HeadingExperiment.run): poll -> estimators -> the heading
controller rewrites the user's wz -> sim.step (16 physics substeps inside).
The run expert uses its final-evaluation plant (normal_contact_plant) with
plain fixed commands.

Squat/getup stay on their batch videos for now: the squat skill is a scripted
reference trajectory plus a bounded residual network, and getup needs its reset
bank -- neither is a plain 18-command policy, so they are not interactive here.
"""
import argparse
import json
import sys
import threading
import time
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

POLICIES = {
    'v7': dict(
        plant=HERE / '20260913_handoff/native_v07',
        onnx=HERE / '20260913_handoff/v7_reference.onnx',
        head=True, label='v7 行走(v07 植物+头部IMU栈)'),
    'candidate': dict(
        plant=HERE / '20260913_handoff/native_v07',
        onnx=Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_v07_calibration/'
                  '20260913_calibration_4096x200/candidate.onnx'),
        head=True, label='校准候选(v07 植物+头部IMU栈)'),
    'run': dict(
        plant=HERE / '20260914_contact_motion/normal_contact_plant',
        onnx=HERE / '20260914_contact_motion/run_16700.onnx',
        head=False, label='跑步专家(normal_contact 植物, 10ms 指令延迟)'),
    'squat': dict(
        plant=HERE / '20260914_squat_specialist/plant',
        onnx=HERE / '20260914_squat_specialist/imitation/candidate.onnx',
        head='squat', label='蹲专项(参考轨迹+有界残差, 滑块控蹲深 0~-25mm)'),
    'crouch': dict(
        plant=HERE / '20260913_handoff/native_v07',
        onnx=Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_deep_crouch/'
                  '20260914_train_512x301/candidate.onnx'),
        head='crouch', label='深蹲 RL(0~-40mm, TerrainSkill 植物+头部IMU栈)'),
}


class DeepCrouchStepper:
    """Trained deep-crouch policy via TerrainSkillExperiment, interactive.

    Their evaluation wrapper slews the body-z reference toward a FIXED posture
    target but only inside the 2..7 s evaluation window; for interactive use
    the same slew rate (+-0.4 mm per policy step) tracks the SLIDER instead,
    with no time gate.  Everything else is their stack: frozen flat plant,
    head-IMU operating profile, CALIBRATOR warm start, 10 ms command delay.
    """
    kind = 'crouch'

    def __init__(self, plant, onnx):
        from hardware_sim import HardwareCase
        from terrain_skill_eval import TerrainSkillExperiment
        self.exp = TerrainSkillExperiment(
            onnx, terrain='flat', posture='crouch40',
            hardware_case=HardwareCase(physics_dt=.00125, command_ms=10))
        self.sim = self.exp.sim
        self.dt = self.sim.dt
        # Recover the plant's own step (the TerrainSkill wrapper captured it in
        # its closure); we replace the wrapper so the crouch tracks the slider
        # continuously instead of only inside their 2..7 s eval window.
        inner = None
        for cell in (self.sim.step.__closure__ or ()):
            c = cell.cell_contents
            if callable(c) and not isinstance(c, (float, int)):
                inner = c
        if inner is None:
            raise SystemExit('未能解开 TerrainSkill 的 step 包装')
        self._inner_step = inner
        self.height_ref = 0.0
        self.target = 0.0
        self.calibration = {}
        self.age = 0.0
        self.applied_wz = 0.0

    def reset(self, seed):
        last = None
        for attempt in range(5):
            try:
                self.calibration = self.exp.reset(seed + attempt)
                break
            except ValueError as exc:
                if not str(exc).startswith('Calibration '):
                    raise
                last = exc
        else:
            raise SystemExit(f'连续 5 个种子校准被拒: {last}')
        self.calibration['accepted_seed'] = seed + attempt
        self.height_ref = 0.0

    def step(self, user, amplitude=1.0):
        self.exp.poll()
        self.target = float(user.get('depth', 0.0))
        self.height_ref += float(np.clip(
            self.target - self.height_ref, -.0004, .0004))
        cmd = np.zeros(18, dtype=np.float32)
        cmd[0] = float(user.get('vx', 0.0))
        cmd[2] = float(user.get('wz', 0.0))
        cmd[9] = self.height_ref
        self._inner_step(cmd)
        self.age = max(0., self.sim.data.time - self.exp.body_est.last_time)
        self.applied_wz = float(cmd[2])
        return cmd


class SquatStepper:
    """Validated squat skill, interactive via the body-z command.

    Reproduces try_squat_reference.run(residual=True): the 6 s calibration
    keeps the v7 calibrator session and the jaw-warmup adapter; afterwards the
    transform_target adapter builds the leg reference from the 25 mm pose,
    scaled by fraction = clip(-obs[72]/0.025, 0, 1) -- obs[72] IS the body-z
    command, so the user's slider drives the squat live -- plus the bounded
    0.015*tanh residual from the specialist ONNX, ankle clamp +-51 deg,
    jaw +0.04, and the head-IMU compensation on top.
    """
    kind = 'squat'

    def __init__(self, plant, onnx):
        import onnxruntime as ort
        from hardware_sim import HardwareCase
        from head_attitude_sim import HeadExperiment
        from head_attitude import HeadConfig
        from imu_owned_head import configure_owned
        from run_heading_stable_start import CALIBRATOR
        self.residual_session = ort.InferenceSession(str(onnx),
                                                     providers=['CPUExecutionProvider'])
        case = HardwareCase(physics_dt=.00125, motor_curve=True, voltage=12.,
                            command_ms=10, position_ms=20, velocity_ms=20,
                            ground_friction=1.)
        self.exp = configure_owned(
            HeadExperiment(plant, CALIBRATOR, case,
                           head_config=HeadConfig(max_measurement_age_s=.04)), (1,))
        self.sim = self.exp.sim
        self.dt = self.sim.dt
        self.calibration = {}
        self.age = 0.0
        self.applied_wz = 0.0
        self.body_z = 0.0
        s = self.sim
        jaw = s.names.index('jaw_hinge')

        def warmup(target, obs):
            t = target.copy()
            t[jaw] = .04
            return self.exp.transform_target(t, obs)
        s.transform_target = warmup

    def reset(self, seed):
        s = self.sim
        # The strict stationary-calibration gate rejects some seeds (their own
        # fixed plans kept rejections in the denominator: 6/13).  Interactive
        # use retries the next seeds and reports the accepted one.
        last = None
        for attempt in range(5):
            try:
                self.calibration = self.exp.reset(seed + attempt)
                break
            except ValueError as exc:
                if not str(exc).startswith('Calibration '):
                    raise
                last = exc
            else:
                break
        else:
            raise SystemExit(f'连续 5 个种子校准被拒: {last}')
        self.calibration['accepted_seed'] = seed + attempt
        ref = next(r for r in json.loads(
            (HERE / '20260914_squat_specialist/pose_references.json').read_text())
            if r['depth_mm'] == 25)
        self.initial = s.applied.copy()
        self.ref25 = np.asarray(ref['target'])
        self.legs = [i for i, n in enumerate(s.names) if n.startswith(('left_', 'right_'))]
        s.session = self.residual_session
        s.raw = np.zeros(19, np.float32)

        def trajectory(target, obs):
            goal = self.initial.copy()
            fraction = np.clip(-float(obs[72]) / .025, 0, 1)
            goal[self.legs] += fraction * (self.ref25 - s.home)[self.legs]
            raw = (target - s.home) / s.scale
            goal[self.legs] += .015 * np.tanh(raw[self.legs])
            for name in ('left_ankle', 'right_ankle'):
                idx = s.names.index(name)
                goal[idx] = np.clip(goal[idx], -np.deg2rad(51), np.deg2rad(51))
            goal[s.names.index('jaw_hinge')] = .04
            return self.exp.transform_target(goal, obs)
        s.transform_target = trajectory

    def step(self, user, amplitude=1.0):
        self.exp.poll()
        cmd = np.zeros(18, dtype=np.float32)
        cmd[9] = user.get('body_z', 0.0)   # negative = squat (validated to -25 mm)
        cmd[17] = .04
        self.sim.step(cmd)
        self.body_z = float(cmd[9])
        return cmd


class HeadStepper:
    """Full stack: heading outer loop + head-IMU compensation, per policy step."""
    kind = 'head'

    def __init__(self, plant, onnx):
        from head_attitude_sim import HeadExperiment
        from head_attitude import HeadConfig, HeadController
        from hardware_sim import HardwareCase
        self.exp = HeadExperiment(plant, onnx, HardwareCase(physics_dt=.00125),
                                  head_enabled=True, head_config=HeadConfig())
        # Operating profile (head_attitude_runtime): drop head correction when
        # the IMU source is stale; inlined so custom onnx paths work.
        self.exp.head_config = dc_replace(self.exp.head_config, max_measurement_age_s=.04)
        self.exp.head_controller = HeadController(self.exp.kinematics, self.exp.head_config)
        self.sim = self.exp.sim
        self.dt = self.sim.dt
        self.calibration = {}
        self.age = 0.0
        self.applied_wz = 0.0

    def reset(self, seed):
        last = None
        for attempt in range(5):
            try:
                self.calibration = self.exp.reset(seed + attempt)
                break
            except ValueError as exc:
                if not str(exc).startswith('Calibration '):
                    raise
                last = exc
        else:
            raise SystemExit(f'连续 5 个种子校准被拒: {last}')
        self.calibration['accepted_seed'] = seed + attempt

    def step(self, user, amplitude=1.0):
        self.exp.poll()
        heading = self.exp.body_est.yaw
        self.age = max(0., self.sim.data.time - self.exp.body_est.last_time)
        corrected = self.exp.controller.update(
            heading, user[2], self.dt, moving=abs(user[0]) > .05,
            measurement_age=self.age)
        cmd = user.copy()
        cmd[2] = corrected
        self.sim.step(cmd, amplitude)
        self.applied_wz = float(cmd[2])
        return cmd


class PlainStepper:
    """Fixed-delay plant, user wz passes straight through (run expert's final
    evaluation drove plain fixed commands)."""
    kind = 'plain'

    def __init__(self, plant, onnx):
        from hardware_sim import HardwareSim, HardwareCase
        self.sim = HardwareSim(plant, onnx,
                               HardwareCase(physics_dt=.00125, command_ms=10))
        self.dt = self.sim.dt
        self.calibration = {'mode': 'plain(无校准环节)'}
        self.age = 0.0
        self.applied_wz = 0.0

    def reset(self, seed):
        self.sim.reset(seed, seed > 0)

    def step(self, user, amplitude=1.0):
        self.sim.step(user, amplitude)
        self.applied_wz = float(user[2])
        return user


def build_stepper(name):
    cfg = POLICIES[name]
    if not cfg['onnx'].exists():
        raise SystemExit(f'缺策略文件: {cfg["onnx"]}')
    if not (cfg['plant'] / 'nominal.mjb').exists():
        raise SystemExit(f'缺植物: {cfg["plant"]}/nominal.mjb')
    cls = {'head': HeadStepper, 'plain': PlainStepper,
           'squat': SquatStepper, 'crouch': DeepCrouchStepper}[cfg['head']]
    return cls(cfg['plant'], cfg['onnx']), cfg


def metrics(stepper):
    d = stepper.sim.data
    R = d.xmat[stepper.sim.body].reshape(3, 3)
    v_local = R.T @ d.qvel[:3]
    tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
    row = {'vx': float(v_local[0]), 'vy': float(v_local[1]), 'tilt': tilt}
    if stepper.kind == 'head':
        row['ref_deg'] = float(np.degrees(stepper.exp.controller.reference))
        row['age_ms'] = stepper.age * 1000
        row['stale'] = bool(stepper.exp.controller.stale)
    return row


def run_smoke(name, seed, seconds):
    stepper, cfg = build_stepper(name)
    t0 = time.perf_counter()
    stepper.reset(seed)
    print(f'校准: {time.perf_counter()-t0:.1f}s')
    if stepper.kind == 'squat':
        user = {'body_z': 0.0}
        for _ in range(round(1.5 / stepper.dt)):
            stepper.step(user)
        base = stepper.sim.data.qpos[2]
        user = {'body_z': -0.020}
        zmin = base
        for _ in range(round(seconds / stepper.dt)):
            stepper.step(user)
            zmin = min(zmin, stepper.sim.data.qpos[2])
        print(f'蹲冒烟: 基准 {base*1000:.1f}mm → 最低 {zmin*1000:.1f}mm '
              f'(实际降 {(base-zmin)*1000:.1f}mm, 指令 -20mm)')
        return
    if stepper.kind == 'crouch':
        user = {'depth': 0.0, 'vx': 0.0, 'wz': 0.0}
        for _ in range(round(1.5 / stepper.dt)):
            stepper.step(user)
        base = stepper.sim.data.qpos[2]
        user['depth'] = -0.040
        zmin = base
        for _ in range(round(seconds / stepper.dt)):
            stepper.step(user)
            zmin = min(zmin, stepper.sim.data.qpos[2])
        print(f'深蹲冒烟: 基准 {base*1000:.1f}mm → 最低 {zmin*1000:.1f}mm '
              f'(实际降 {(base-zmin)*1000:.1f}mm, 指令 -40mm)')
        return
    cmd = np.zeros(18, dtype=np.float32)
    cmd[0] = 0.45
    vx_all, fell = [], False
    for _ in range(round(seconds / stepper.dt)):
        stepper.step(cmd)
        m = metrics(stepper)
        vx_all.append(m['vx'])
        if m['tilt'] > 60 or stepper.sim.data.qpos[2] < 0.055:
            fell = True
            break
    print(f'前进 {seconds:g}s: vx均值={np.mean(vx_all):+.3f} m/s  '
          f'末vx={vx_all[-1]:+.3f}  跌倒={fell}')


def run_window(name, seed):
    import tkinter as tk
    from PIL import Image, ImageDraw, ImageFont, ImageTk
    import mujoco

    stepper, cfg = build_stepper(name)
    W, H = 720, 520
    root = tk.Tk()
    root.title(f'MicroDinosaur 专项交互台 - {cfg["label"]}')
    root.minsize(520, 420)
    img = ImageTk.PhotoImage(Image.new('RGB', (W, H), (24, 24, 28)))
    label = tk.Label(root, image=img, bd=0)
    label.pack(fill='both', expand=True)
    holder = {'img': img}

    panel = tk.Frame(root)
    panel.pack(fill='x')
    st = dict(vx=0.0, wz=0.0, tail=0.0, amplitude=1.0)

    def slider(parent, key, lo, hi, res, text, fmt='{:.2f}'):
        f = tk.Frame(parent)
        f.pack(side='left', padx=6)
        tk.Label(f, text=text, font=('Microsoft YaHei', 12)).pack(anchor='w')
        val = tk.Label(f, text=fmt.format(st[key]),
                       font=('Microsoft YaHei', 13, 'bold'), fg='#0a6')
        val.pack(anchor='w')

        def on_change(v, k=key, l=val, fm=fmt):
            st[k] = float(v)
            l.configure(text=fm.format(float(v)))
        sc = tk.Scale(f, from_=lo, to=hi, resolution=res, orient='horizontal',
                      length=190, showvalue=False, command=on_change)
        sc.set(st[key])
        sc.pack()
    if stepper.kind == 'squat':
        st['body_z'] = 0.0
        slider(panel, 'body_z', -0.025, 0.0, 0.001, '蹲深指令 (0 ~ -25mm)')
        tk.Label(panel, text='蹲专项: 蹲深滑块即指令; vx/wz 在该技能里不参与(与其验收条件一致)',
                 font=('Microsoft YaHei', 11), fg='#666').pack(anchor='w', padx=6)
    elif stepper.kind == 'crouch':
        st['depth'] = 0.0
        slider(panel, 'depth', -0.04, 0.0, 0.001, '蹲深指令 (0 ~ -40mm, 内部限速 8mm/s)')
        slider(panel, 'vx', -0.1, 0.55, 0.05, '前进 vx (训练域 -0.1~0.55)')
        slider(panel, 'wz', -0.35, 0.35, 0.05, '转向 wz (训练域 ±0.35)')
        tk.Label(panel, text='深蹲 RL: 蹲/走可同时; 尾巴手臂嘴在训练里固定为 0, 不提供滑块',
                 font=('Microsoft YaHei', 11), fg='#666').pack(anchor='w', padx=6)
    else:
        slider(panel, 'vx', -0.3, 0.8, 0.05, '前进 vx (m/s)')
        slider(panel, 'wz', -0.45, 0.45, 0.05, '转向 wz (经航向外环)')
        slider(panel, 'tail', 0.0, 1.75, 0.05, '尾巴抬起 (rad)')
        slider(panel, 'amplitude', 0.8, 1.2, 0.05, '动作幅度')

    status = tk.Label(panel, text='就绪', font=('Consolas', 12), anchor='w',
                      justify='left')
    status.pack(fill='x', padx=4)
    btnf = tk.Frame(panel)
    btnf.pack(fill='x')

    state = {'calibrating': True, 'fell': False, 'steps': 0, 'seed': seed}

    def calibration_work():
        stepper.reset(state['seed'])
        state['seed'] += 1
        state['calibrating'] = False
        state['fell'] = False
        state['steps'] = 0

    def do_reset():
        if state['calibrating']:
            return
        state['calibrating'] = True
        status.configure(text='校准中(6 s 静止标定, 几秒)...', fg='#c60')
        threading.Thread(target=calibration_work, daemon=True).start()

    tk.Button(btnf, text='重置+校准', font=('Microsoft YaHei', 12),
              command=do_reset).pack(side='left', padx=2)
    tk.Button(btnf, text='停', font=('Microsoft YaHei', 12),
              command=lambda: st.update(vx=0.0, wz=0.0)).pack(side='left', padx=2)
    tk.Button(btnf, text='退出', font=('Microsoft YaHei', 12),
              command=root.quit).pack(side='left', padx=2)

    ren = {'r': None, 'size': None}

    def tick():
        if state['calibrating']:
            root.after(50, tick)
            return
        t_target = time.perf_counter() + stepper.dt
        if stepper.kind in ('squat', 'crouch'):
            stepper.step(st)          # st carries body_z / depth+vx+wz
        else:
            cmd = np.zeros(18, dtype=np.float32)
            cmd[0], cmd[2], cmd[16] = st['vx'], st['wz'], st['tail']
            stepper.step(cmd, st['amplitude'])
        state['steps'] += 1
        m = metrics(stepper)
        if m['tilt'] > 60 or stepper.sim.data.qpos[2] < 0.055:
            state['fell'] = True
        info = (f'vx={m["vx"]:+.3f}  tilt={m["tilt"]:.1f}deg  '
                f'steps={state["steps"]}'
                + (f'  [跌倒→重置+校准]' if state['fell'] else ''))
        if stepper.kind == 'squat':
            frac = max(0., min(1., -st['body_z'] / .025))
            info += (f'  蹲深指令={-st["body_z"]*1000:.0f}mm(比例{frac*100:.0f}%)'
                     f'  根高={stepper.sim.data.qpos[2]*1000:.1f}mm')
        if stepper.kind == 'crouch':
            info += (f'  蹲深指令={-st["depth"]*1000:.0f}mm'
                     f'  参考={stepper.height_ref*1000:+.1f}mm'
                     f'  根高={stepper.sim.data.qpos[2]*1000:.1f}mm')
        if stepper.kind == 'head':
            info += (f'  航向参考={m["ref_deg"]:+.1f}deg'
                     f'  IMU龄={m["age_ms"]:.0f}ms'
                     f'{"  [保持]" if m["stale"] else ""}')
        status.configure(text=info, fg='#c00' if state['fell'] else '#060')
        # Render at the CURRENT window size: drag-resize works like the walk
        # window.  Recreating a Renderer per resize event is expensive, so the
        # size is quantised to 32 px buckets, which also stops jitter mid-drag.
        w = max(320, (label.winfo_width() // 32) * 32)
        h = max(240, (label.winfo_height() // 32) * 32)
        if ren['size'] != (w, h):
            if ren['r'] is not None:
                ren['r'].close()
            ren['r'] = mujoco.Renderer(stepper.sim.model, height=h, width=w)
            ren['size'] = (w, h)
        cam = mujoco.MjvCamera()
        p = stepper.sim.data.xpos[stepper.sim.body]
        cam.lookat[:] = [p[0], p[1], max(p[2], 0.09)]
        cam.azimuth, cam.elevation, cam.distance = 135, -18, 0.9
        ren['r'].update_scene(stepper.sim.data, camera=cam)
        frame = ren['r'].render()
        im = Image.fromarray(frame)
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc', 17)
        draw.rectangle([0, 0, im.width, 26], fill=(0, 0, 0))
        draw.text((8, 4), f'{name}  vx={m["vx"]:+.2f}  tilt={m["tilt"]:.0f}',
                  fill=(255, 220, 120), font=font)
        holder['img'] = ImageTk.PhotoImage(im)
        label.configure(image=holder['img'])
        delay = max(1, round((t_target + stepper.dt - time.perf_counter()) * 1000))
        root.after(delay, tick)

    status.configure(text='校准中(6 s 静止标定, 几秒)...', fg='#c60')
    threading.Thread(target=calibration_work, daemon=True).start()
    root.after(50, tick)
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', choices=tuple(POLICIES), default='v7')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--smoke', type=float, default=0.,
                    help='无窗口冒烟: 校准后以 vx=0.45 走 N 秒并打印速度')
    args = ap.parse_args()
    if args.smoke:
        run_smoke(args.policy, args.seed, args.smoke)
        return
    root = run_window(args.policy, args.seed)
    root.mainloop()


if __name__ == '__main__':
    main()
