"""录制"出生姿态 → 策略把腿放下"的对比视频。

用户观察到的"两种抬脚形态"实测真相:
  出生瞬间是 Flamingo 前伸姿态(髋+69°/膝-46°/踝+46°),
  策略在 **0.24 秒内**把腿放下来(髋 -10°),之后稳定在"垂下"形态(髋+5°);
  每 6 秒 episode 重置一次,所以长时看像"两种形态来回切换"。

视频分两段:
  段1 慢动作 6 倍:出生后 1.5 秒,看清腿怎么被放下的
  段2 正常速度 12 秒:两个完整 episode,看清"垂下 → 重置回前伸"的循环
两段都叠加 髋/膝/踝 实时角度 + 姿态参考值 + episode 编号。
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import mujoco
import onnxruntime as ort
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from flamingo_cfg import build_config, FLAMINGO_POSE_BY_NAME  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

W, H = 960, 720
FONT = 'C:/Windows/Fonts/msyh.ttc'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--slow-sec', type=float, default=1.5)
    ap.add_argument('--slow-factor', type=int, default=6)
    ap.add_argument('--normal-sec', type=float, default=12.)
    a = ap.parse_args()

    task, cfg = build_config(envs=1, seed=47, standing_prob=0.)
    env = ManagerBasedRlEnv(cfg.env, device='cuda')
    sess = ort.InferenceSession(a.checkpoint, providers=['CPUExecutionProvider'])
    robot = env.scene['robot']
    jid = {k: 7 + robot.find_joints([k])[0][0] for k in
           ('left_hip_pitch', 'left_knee', 'left_ankle')}
    ref = FLAMINGO_POSE_BY_NAME

    from terrain_skill_eval import freeze_plant
    plant = freeze_plant('flat') / 'nominal.mjb'
    model = mujoco.MjModel.from_binary_path(
        'nominal.mjb', assets={'nominal.mjb': plant.read_bytes()})
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 105., -6., .62
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0

    fb = ImageFont.truetype(FONT, 26)
    fm = ImageFont.truetype(FONT, 21)
    fs = ImageFont.truetype(FONT, 18)

    obs, _ = env.reset()
    epi = [0]

    def step_and_draw(ep_tag):
        nonlocal obs
        o = obs['actor'] if isinstance(obs, dict) else obs
        o = o.flatten(1) if o.dim() > 2 else o
        act = sess.run(None, {'obs': o.detach().cpu().numpy().astype(np.float32)})[0]
        obs, *_ = env.step(torch.as_tensor(act, device=env.device))
        d = env.sim.data
        data.qpos[:] = d.qpos[0].cpu().numpy()
        mujoco.mj_forward(model, data)
        cam.lookat[:] = [float(d.qpos[0, 0].cpu()), float(d.qpos[0, 1].cpu()), .10]
        renderer.update_scene(data, camera=cam, scene_option=opt)
        img = Image.fromarray(renderer.render()).copy()
        q = d.qpos[0]
        hip = math.degrees(float(q[jid['left_hip_pitch']].cpu()))
        knee = math.degrees(float(q[jid['left_knee']].cpu()))
        ank = math.degrees(float(q[jid['left_ankle']].cpu()))
        dr = ImageDraw.Draw(img)
        dr.rectangle((0, 0, W, 62), fill=(16, 19, 23))
        dr.text((14, 6), f'{ep_tag}   episode {epi[0]}', font=fb, fill=(255, 220, 90))
        dr.text((14, 36), f't = {float(d.time.cpu()):5.2f}s', font=fm, fill=(200, 210, 220))
        # 参考线:姿态前伸 = 髋+69/膝-46/踝+46;放下 = 髋+5/膝+23/踝-19
        dr.rectangle((0, H - 118, W, H), fill=(16, 19, 23))
        def bar(y, name, val, ref_val, lo, hi):
            x0, x1 = 190, W - 130
            dr.text((14, y + 2), name, font=fm, fill=(200, 210, 220))
            dr.rectangle((x0, y + 6, x1, y + 24), outline=(80, 90, 100))
            zx = x0 + int((0 - lo) / (hi - lo) * (x1 - x0))
            dr.line((zx, y + 2, zx, y + 28), fill=(110, 120, 130), width=2)
            for rv, col, lab in ((ref_val, (255, 220, 90), '姿态'),
                                 (0., (150, 160, 170), '')):
                rx = x0 + int((rv - lo) / (hi - lo) * (x1 - x0))
                dr.line((rx, y, rx, y + 30), fill=col, width=3)
            vx = x0 + int((np.clip(val, lo, hi) - lo) / (hi - lo) * (x1 - x0))
            dr.ellipse((vx - 7, y + 8, vx + 7, y + 22), fill=(90, 200, 120))
            dr.text((x1 + 8, y + 2), f'{val:+6.1f}°', font=fm, fill=(150, 220, 140))
        bar(H - 112, '左髋俯仰', hip, math.degrees(ref['left_hip_pitch']), -90, 90)
        bar(H - 78, '左膝    ', knee, math.degrees(ref['left_knee']), -90, 90)
        bar(H - 44, '左踝    ', ank, math.degrees(ref['left_ankle']), -90, 90)
        return np.asarray(img)

    frames = []
    # ── 段1:慢动作(每个物理帧重复 slow_factor 次) ──
    n_slow = int(a.slow_sec / .02)
    for k in range(n_slow):
        f = step_and_draw(f'段1 慢动作 {a.slow_factor}× · 出生后 {k * .02:.2f}s')
        frames.extend([f] * a.slow_factor)
    # ── 段2:正常速度,覆盖两次 episode 重置 ──
    n_norm = int(a.normal_sec / .02)
    for k in range(n_norm):
        t_before = float(env.sim.data.time.cpu())
        f = step_and_draw('段2 正常速度 · 注意 episode 边界处姿态跳变')
        if float(env.sim.data.time.cpu()) < t_before:   # 发生了 reset
            epi[0] += 1
        frames.append(f)

    writer = imageio.get_writer(a.out, fps=25, codec='libx264', quality=7)
    for f in frames:
        writer.append_data(f)
    writer.close()
    renderer.close()
    total_s = len(frames) / 25
    print(f'[write] {a.out}  {len(frames)}帧 = {total_s:.1f}s '
          f'(段1 慢动作 {a.slow_sec}s×{a.slow_factor} = {n_slow * a.slow_factor / 25:.1f}s, '
          f'段2 正常 {a.normal_sec}s)')


if __name__ == '__main__':
    main()
