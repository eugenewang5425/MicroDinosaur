"""幅度扫描视频:姿态栈每个命令通道 × 4 档幅度,顺序扫一遍并标注。

段表(每档: 斜坡→保持→回零;深蹲斜坡由内部 0.02m/s 转速决定时长):
  深蹲 -6/-12/-18/-25mm → 折叠蹲 25/50/75/100% → 尾俯仰 ∓25°
  → 尾偏航 ±30° → 左臂 ±30° → 右臂 ∓30° → 脖子 +14/-10°
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
from pose_stack import PoseStack  # noqa: E402

OUT = ROOT / '20260914_contact_motion'
W, H = 760, 760
FONT = 'C:/Windows/Fonts/msyh.ttc'
D2R = math.pi / 180.

# (通道, 幅度, 斜坡s, 保持s) — 依次执行,每档结束自动回零
SWEEP = [
    ('body_z', -.006, 1.2, 1.6), ('body_z', -.012, 1.2, 1.6),
    ('body_z', -.018, 1.2, 1.6), ('body_z', -.025, 1.2, 2.0),
    ('fold', .25, 1.5, 1.5), ('fold', .50, 1.5, 1.5),
    ('fold', .75, 2.0, 1.5), ('fold', 1.0, 2.5, 3.0),
    ('REST', 0., 0., 3.5),
    ('tail_pitch', -25 * D2R, .8, 1.4), ('tail_pitch', +25 * D2R, .8, 1.4),
    ('tail_yaw', +30 * D2R, .8, 1.4), ('tail_yaw', -30 * D2R, .8, 1.4),
    ('arm_l', +30 * D2R, .8, 1.4), ('arm_l', -30 * D2R, .8, 1.4),
    ('arm_r', -30 * D2R, .8, 1.4), ('arm_r', +30 * D2R, .8, 1.4),
    ('neck_pitch', +14 * D2R, .8, 1.4), ('neck_pitch', -10 * D2R, .8, 1.4),
]
GAP = .6  # 每档之间回零停顿


def build_schedule():
    """生成 (t, 通道值 dict, 段标签) 的时间函数表。"""
    t = 1.0  # 开头站立 1s
    plan = []  # (t0, t1, t2, t3, channel, amp, label)
    for ch, amp, ramp, hold in SWEEP:
        if ch == 'REST':
            plan.append((t, t, t, t + hold, 'fold', 0., '起身回站立(全折叠→0 自动站起)'))
            t += hold + GAP
            continue
        label = {'body_z': f'深蹲 {abs(amp) * 1000:.0f} mm',
                 'fold': f'折叠蹲 {amp * 100:.0f} %'}.get(
            ch, f'{ch} {amp / D2R:+.0f}°' if ch != 'neck_pitch'
            else f'脖子 {amp / D2R:+.0f}°')
        if ch == 'neck_pitch':
            label = f'脖子 {amp / D2R:+.0f}°'
        plan.append((t, t + ramp, t + ramp + hold, t + ramp + hold + ramp,
                     ch, amp, label))
        t += ramp + hold + ramp + GAP
    total = t + 1.5
    return plan, total


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    stack = PoseStack(941)
    plan, total = build_schedule()
    s = stack.s
    dt = s.dt
    n_steps = int(total / dt)

    def sched(t):
        value = {ch: 0. for ch in
                 ('body_z', 'fold', 'tail_pitch', 'tail_yaw', 'arm_l', 'arm_r',
                  'neck_pitch')}
        label = '站立'
        S = lambda u: float(np.clip(u, 0., 1.)) ** 2 * (3 - 2 * float(np.clip(u, 0., 1.)))
        for t0, t1, t2, t3, ch, amp, lab in plan:
            if t < t0 or t > t3:
                continue
            u = (S((t - t0) / (t1 - t0)) if t < t1
                 else 1. if t < t2 else 1. - S((t - t2) / (t3 - t2)))
            value[ch] = amp * u
            label = lab
        # 深折叠自动收姿(抬尾+收臂):伸腿弹起的配重。实测(扫描首轮)没有它,
        # fold 100→0 后机器人卡在 80mm/22.9° 的支撑姿态站不起来 —— 与演示版
        # (带收姿,成功站起)的唯一差异。依据 = pose_modules_demo 的对照。
        tuck = S((value['fold'] - .95) / .05)
        value['tail_pitch'] -= .52 * tuck
        value['arm_l'] += .17 * tuck
        value['arm_r'] += .17 * tuck
        return value, label

    frames = []
    labels = []
    prev_cb = s.substep_callback

    def record():
        prev_cb()
        if not frames or s.data.time - frames[-1][0] >= .04 - 1e-9:
            frames.append((s.data.time, s.data.qpos.copy()))
            _, lab = sched(s.data.time)
            # 读数必须录制期采样:渲染期 s.data 已是终态,三帧会同数
            labels.append((s.data.time, lab, stack.readout()))

    s.substep_callback = record
    finite = True
    for k in range(n_steps):
        value, _ = sched(k * dt)
        stack.apply(value)
        stack.step()
        if not np.isfinite(s.data.qpos).all():
            print(f'[abort] t={k * dt:.2f}s NONFINITE')
            finite = False
            break
    s.substep_callback = prev_cb
    print(f'[sim] {n_steps} 步,帧 {len(frames)},finite={finite}', flush=True)

    m = s.model
    renderer = mujoco.Renderer(m, height=H, width=W)
    data = mujoco.MjData(m)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = .95, 135., -16
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    big = ImageFont.truetype(FONT, 30)
    mid = ImageFont.truetype(FONT, 21)
    small = ImageFont.truetype(FONT, 17)

    lo, hi, _ = __import__('pose_stack').CHANNELS['body_z']

    writer = imageio.get_writer(str(OUT / 'pose_amplitude_sweep.mp4'), fps=25,
                                codec='libx264', quality=7)
    for i, (t, qpos) in enumerate(frames):
        data.qpos[:] = qpos
        mujoco.mj_forward(m, data)
        cam.lookat[:] = [data.qpos[0], data.qpos[1], data.qpos[2] + .02]
        renderer.update_scene(data, camera=cam, scene_option=opt)
        left = Image.fromarray(renderer.render())
        dl = ImageDraw.Draw(left)
        dl.rectangle((0, 0, W, 40), fill=(20, 24, 28))
        dl.text((12, 6), '姿态栈幅度扫描 · 每个指令 4 档', font=mid, fill='white')
        li = min(len(labels) - 1, i)
        t_i, lab, ro = labels[li]

        img = Image.new('RGB', (2 * W, H), (14, 17, 20))
        img.paste(left, (0, 0))
        dr = ImageDraw.Draw(img)
        dr.rectangle((W, 0, 2 * W, H), fill=(20, 24, 28))
        dr.text((W + 20, 20), lab, font=big, fill=(255, 220, 90))
        # 当前段的所有非零通道读数
        value, _ = sched(t_i)
        y = 100
        names = {'body_z': '深蹲 mm', 'fold': '折叠 %', 'tail_pitch': '尾俯仰°',
                 'tail_yaw': '尾偏航°', 'arm_l': '左臂°', 'arm_r': '右臂°',
                 'neck_pitch': '脖子°'}
        for ch, nm in names.items():
            v = value[ch]
            txt = f'{v * 1000:+6.1f} mm' if ch == 'body_z' else (
                f'{v * 100:5.1f} %' if ch == 'fold' else f'{v / D2R:+6.1f}°')
            active = abs(v) > 1e-6
            dr.text((W + 24, y), f'{nm:8s} {txt}', font=mid,
                    fill=(160, 220, 160) if active else (90, 100, 110))
            y += 34
        dr.rectangle((W, H - 92, 2 * W, H), fill=(20, 24, 28))
        dr.text((W + 24, H - 84), f'根高 {ro["root_z_mm"]:6.1f} mm   倾角 '
                                  f'{ro["tilt_deg"]:5.2f}°   双脚 {ro["foot_n"]:.2f} N',
                font=small, fill=(190, 200, 210))
        dr.text((W + 24, H - 56), '深蹲=蹲深命令通道  折叠蹲=折腿比例(起身模组同源)',
                font=small, fill=(190, 200, 210))
        dr.text((W + 24, H - 28), '尾/脖/臂=直接偏移  零训练', font=small,
                fill=(190, 200, 210))
        writer.append_data(np.asarray(img))
    writer.close()
    renderer.close()
    print(f'[write] {OUT / "pose_amplitude_sweep.mp4"}  {len(frames)}帧 @25fps = '
          f'{len(frames) * .04:.1f}s')


if __name__ == '__main__':
    main()
