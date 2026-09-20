"""命令式动作模组演示(零训练):站姿上浮现不同幅度的深蹲与折叠蹲 + 尾/脖/臂偏移。

三个幅度通道(全部是指令,没有任何新训练):
  1. 深蹲幅度 = body-z 命令(obs[72]),SquatStepper 已验证 0~-25mm;
  2. 折叠蹲幅度 = 向折腿姿态(±86°,起身模组验证过的同一条)插值的比例 0..1;
  3. 尾巴/脖子/双臂 = 目标数组上的附加偏移(慢斜坡)。
头部俯仰/偏航由 IMU 稳定环托管(用户设计意图 yaw_follows_trunk),演示里不叠加。
步态(跑模块 vx/wz)按用户决定暂不并入,可调性已由绕圈演示(0.45/0.6 m/s)证实。
"""
from pathlib import Path
import sys
import json
import math

import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from skill_console import SquatStepper  # noqa: E402
from mjlab_microduck.tasks import recovery_bounds  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / '20260914_contact_motion'
W, H = 760, 760
FONT = 'C:/Windows/Fonts/msyh.ttc'
SMOOTH = lambda u: float(np.clip(u, 0., 1.)) ** 2 * (3 - 2 * np.clip(u, 0., 1.))


def smoothstep(t, t0, t1):
    return SMOOTH((t - t0) / max(t1 - t0, 1e-9))


def pulse(t, t0, t1, t2, t3):
    """梯形: t0-t1 升到 1,t2-t3 降回 0。"""
    return smoothstep(t, t0, t1) * (1 - smoothstep(t, t2, t3))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    squat = json.loads((HERE / '20260914_squat_specialist' / 'pose_references.json')
                       .read_text())
    stepper = SquatStepper(HERE / '20260914_squat_specialist' / 'plant',
                           HERE / '20260914_squat_specialist' / 'imitation/candidate.onnx')
    stepper.reset(941)
    s = stepper.sim
    names = s.names
    home = np.asarray(s.home, dtype=float)
    legs = stepper.legs
    ix = {n: names.index(n) for n in
          ('tail_pitch', 'tail_yaw', 'arm_l', 'arm_r', 'neck_pitch', 'jaw_hinge')}
    # 折叠目标(单一真源:起身模组验证过的紧凑折腿)
    fold = home.copy()
    for n, deg in recovery_bounds.SITFOLD_FOLD_DEG.items():
        fold[names.index(n)] = np.deg2rad(deg)
    limits = s.model.jnt_range[s.jids]
    fold_delta = (fold - home)[legs]

    state = dict(fold=0., tail_pitch=0., tail_yaw=0., arm=0., neck=0.)

    def trajectory(target, obs):
        """SquatStepper 原版 + 折叠比例通道 + 尾/脖/臂偏移通道。"""
        fraction = np.clip(-float(obs[72]) / .025, 0, 1)
        goal = stepper.initial.copy()
        goal[legs] += fraction * (stepper.ref25 - home)[legs]
        goal[legs] += state['fold'] * fold_delta
        raw = (target - s.home) / s.scale
        goal[legs] += .015 * np.tanh(raw[legs])
        goal[ix['tail_pitch']] += state['tail_pitch']
        goal[ix['tail_yaw']] += state['tail_yaw']
        goal[ix['arm_l']] += state['arm']
        goal[ix['arm_r']] += state['arm']
        goal[ix['neck_pitch']] += state['neck']
        goal[legs] = np.clip(goal[legs], limits[legs, 0] + .07, limits[legs, 1] - .07)
        goal[ix['jaw_hinge']] = .04
        return stepper.exp.transform_target(goal, obs)

    s.transform_target = trajectory

    # ---------------- 时间表(秒) ----------------
    def schedule(t):
        """返回 (body_z 命令 m, 折叠比例, 偏移 dict, 阶段名)。"""
        # 深蹲幅度阶梯 2-10s: 0→-12mm→0→-25mm→0
        z1 = -.012 * pulse(t, 2., 3.5, 5., 6.)
        z2 = -.025 * pulse(t, 6., 7.5, 9., 10.)
        # 折叠幅度阶梯 12-24s
        f1 = pulse(t, 12., 13.5, 15., 16.5) * .35
        f2 = pulse(t, 17., 18.5, 20., 21.5) * .70
        # 全折叠 24-33s: 下蹲 3s → 保持 → 弹起 3s(与起身验证脚本同时序)
        full = smoothstep(t, 24., 27.) * (1 - smoothstep(t, 30., 33.))
        fold = float(np.clip(f1 + f2 + full, 0., 1.))
        body_z = float(z1 + z2)
        # 尾/脖/臂偏移:站立段(10-14)扫摆;全折叠保持段(27-30)收姿
        wave = pulse(t, 10., 11., 13., 14.)
        tuck = smoothstep(t, 27., 28.) * (1 - smoothstep(t, 30., 31.))
        tail_pitch = (-np.deg2rad(22.) * wave) + (-np.deg2rad(30.) * tuck)
        tail_yaw = np.deg2rad(25.) * math.sin(math.pi * max(0., min(1., (t - 10.) / 4.))) * wave
        arm = np.deg2rad(28.) * wave + (np.deg2rad(10.) * tuck)
        neck = (np.deg2rad(14.) * smoothstep(t, 11., 12.5)
                * (1 - smoothstep(t, 12.5, 14.))
                - np.deg2rad(10.) * smoothstep(t, 13., 14.) * (1 - smoothstep(t, 14., 15.)))
        phase = ('站立' if t < 2 else
                 '深蹲幅度阶梯' if t < 10 else
                 '尾巴/脖子/手臂扫摆' if t < 12 else
                 '折叠蹲幅度阶梯' if t < 24 or t < 24 else
                 '折叠蹲幅度阶梯' if t < 24 else
                 '全折叠(坐进折叠蹲)' if t < 33 else '回到站立 + 姿势扫摆')
        return body_z, fold, dict(tail_pitch=tail_pitch, tail_yaw=tail_yaw,
                                  arm=arm, neck=neck), phase

    TOTAL = 38.
    frames = []
    prev_cb = s.substep_callback

    def record():
        prev_cb()
        if len(frames) == 0 or s.data.time - frames[-1][0] >= .04 - 1e-9:
            frames.append((s.data.time, s.data.qpos.copy()))

    s.substep_callback = record
    labels = []
    n_steps = int(TOTAL / s.dt)
    for k in range(n_steps):
        t = k * s.dt
        body_z, fold, off, phase = schedule(t)
        state.update(fold=fold, **off)
        stepper.step({'body_z': body_z})
        labels.append((t, phase, body_z, fold, dict(off)))
        if not np.isfinite(s.data.qpos).all():
            print(f'[abort] t={t:.2f}s NONFINITE')
            break
    s.substep_callback = prev_cb
    print(f'[sim] {n_steps} 步,采帧 {len(frames)}', flush=True)

    # ---------------- 渲染 ----------------
    m = s.model
    renderer = mujoco.Renderer(m, height=H, width=W)
    data = mujoco.MjData(m)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = .9, 135., -14
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    big = ImageFont.truetype(FONT, 26)
    mid = ImageFont.truetype(FONT, 21)
    small = ImageFont.truetype(FONT, 17)

    def bar(dr, x, y, w, h, frac, color, text):
        dr.rectangle((x, y, x + w, y + h), outline=(90, 100, 110), width=2)
        dr.rectangle((x, y, x + int(w * np.clip(frac, 0, 1)), y + h), fill=color)
        dr.text((x, y + h + 4), text, font=small, fill=(190, 200, 210))

    writer = imageio.get_writer(str(OUT / 'pose_modules_demo.mp4'), fps=25,
                                codec='libx264', quality=7)
    for k, (t, qpos) in enumerate(frames):
        data.qpos[:] = qpos
        mujoco.mj_forward(m, data)
        cam.lookat[:] = [data.qpos[0], data.qpos[1], data.qpos[2] + .02]
        renderer.update_scene(data, camera=cam, scene_option=opt)
        left = Image.fromarray(renderer.render())
        dl = ImageDraw.Draw(left)
        dl.rectangle((0, 0, W, 40), fill=(20, 24, 28))
        li = min(len(labels) - 1, int(t / s.dt))
        _, phase, body_z, fold, off = labels[li]
        dl.text((12, 7), f'命令式动作模组演示  t={t:5.1f}s  {phase}', font=mid, fill='white')

        img = Image.new('RGB', (2 * W, H), (14, 17, 20))
        img.paste(left, (0, 0))
        dr = ImageDraw.Draw(img)
        dr.rectangle((W, 0, 2 * W, H), fill=(20, 24, 28))
        dr.text((W + 20, 24), '零训练 · 全部为指令', font=big, fill=(255, 220, 90))
        bar(dr, W + 20, 90, 560, 22, -body_z / .025, (90, 170, 255),
            f'深蹲幅度  body-z {body_z * 1000:+6.1f} mm  (通道 obs[72], 已验证 0~-25)')
        bar(dr, W + 20, 160, 560, 22, fold, (255, 160, 60),
            f'折叠蹲幅度  折腿比例 {fold * 100:5.1f} %  (起身模组同一条 ±86° 折腿)')
        dr.text((W + 20, 240), '尾巴/脖子/手臂偏移 (rad):', font=small, fill=(190, 200, 210))
        for j, (nm, key) in enumerate([('tail_pitch', 'tail_pitch'), ('tail_yaw', 'tail_yaw'),
                                       ('arm_l=arm_r', 'arm'), ('neck_pitch', 'neck')]):
            dr.text((W + 40, 268 + j * 26),
                    f'{nm:12s} {off[key]:+.3f}', font=small, fill=(160, 220, 160)
                    if abs(off[key]) > 1e-4 else (120, 130, 140))
        dr.text((W + 20, H - 120), '深蹲 = 蹲深命令通道(蹲专项残差网络出平衡)',
                font=small, fill=(190, 200, 210))
        dr.text((W + 20, H - 92), '折叠蹲 = 向已验证折腿姿态插值(起身模组同一条)',
                font=small, fill=(190, 200, 210))
        dr.text((W + 20, H - 64), '步态(跑模块 vx/wz)独立可调: 0.3-0.75 m/s 已验证',
                font=small, fill=(190, 200, 210))
        dr.text((W + 20, H - 36), '头部俯仰/偏航 = IMU 稳定环托管(yaw_follows_trunk)',
                font=small, fill=(150, 160, 170))
        writer.append_data(np.asarray(img))
    writer.close()
    renderer.close()
    print(f'[write] {OUT / "pose_modules_demo.mp4"}  {len(frames)}帧 @25fps')


if __name__ == '__main__':
    main()
