"""Flamingo 单脚站立多角度高清截图。

8 个方位角 + 俯视,分辨率 1280x960,拼成一张联系表(contact sheet)便于逐角度检查。
每个角度下策略实际运行,读数叠加在图上(摆动脚高度/质心距/倾角/头尾偏航)。
"""
from pathlib import Path
import sys
import argparse
import math

import numpy as np
import torch
import mujoco
import onnxruntime as ort
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from flamingo_cfg import build_config  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

W, H = 1280, 960
FONT = 'C:/Windows/Fonts/msyh.ttc'
AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--warm-s', type=float, default=2.0,
                    help='每个角度先跑这么久再截图(让姿态稳定)')
    ap.add_argument('--elevation', type=float, default=-10.)
    a = ap.parse_args()

    task, cfg = build_config(envs=1, seed=47, standing_prob=0.)
    env = ManagerBasedRlEnv(cfg.env, device='cpu')
    sess = ort.InferenceSession(a.checkpoint, providers=['CPUExecutionProvider'])
    robot = env.scene['robot']

    from terrain_skill_eval import freeze_plant
    plant = freeze_plant('flat') / 'nominal.mjb'
    model = mujoco.MjModel.from_binary_path(
        'nominal.mjb', assets={'nominal.mjb': plant.read_bytes()})
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)

    names = [n.split('/')[-1] for n in robot.joint_names]
    jadr = {k: 7 + robot.find_joints([k])[0][0]
            for k in ('head_yaw', 'tail_yaw', 'tail_pitch')}
    sites = [n.split('/')[-1] for n in robot.site_names]
    sw, st = sites.index('left_foot'), sites.index('right_foot')
    root_bid = int(robot.indexing.root_body_id)

    font_big = ImageFont.truetype(FONT, 40)
    font_mid = ImageFont.truetype(FONT, 28)

    obs, _ = env.reset()

    def step_once():
        nonlocal obs
        o = obs['actor'] if isinstance(obs, dict) else obs
        if o.dim() > 2:
            o = o.flatten(1)
        onx = o.detach().numpy().astype(np.float32)
        act = np.stack([sess.run(None, {'obs': onx[b:b + 1]})[0][0]
                        for b in range(onx.shape[0])])
        obs, *_ = env.step(torch.as_tensor(act))

    tiles = []
    for az in AZIMUTHS:
        obs, _ = env.reset()                    # 每个角度独立出生(可复现)
        for _ in range(int(a.warm_s / .04)):    # 稳定 2 秒
            step_once()
        d = env.sim.data
        data.qpos[:] = d.qpos[0].numpy()
        data.qvel[:] = d.qvel[0].numpy()
        mujoco.mj_forward(model, data)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth, cam.elevation, cam.distance = float(az), a.elevation, .52
        cam.lookat[:] = [float(d.qpos[0, 0]), float(d.qpos[0, 1]), .11]
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = 0
        renderer.update_scene(data, camera=cam, scene_option=opt)
        img = Image.fromarray(renderer.render()).copy()
        sz = float(robot.data.site_pos_w[0, sw, 2]) * 1000
        com = np.asarray(d.subtree_com[0, root_bid, :2])
        foot = robot.data.site_pos_w[0, st, :2].numpy()
        cd = float(np.linalg.norm(com - foot)) * 1000
        gz = robot.data.projected_gravity_b[0].numpy()
        tilt = math.degrees(math.acos(min(1., abs(gz[2]))))
        hy = math.degrees(float(d.qpos[0, jadr['head_yaw']]))
        ty = math.degrees(float(d.qpos[0, jadr['tail_yaw']]))
        tp = math.degrees(float(d.qpos[0, jadr['tail_pitch']]))
        dr = ImageDraw.Draw(img)
        dr.rectangle((0, 0, W, 60), fill=(16, 19, 23))
        dr.text((16, 10), f'方位角 {az}°   摆动脚 {sz:5.1f}mm   质心距 {cd:5.1f}mm   '
                          f'倾角 {tilt:4.1f}°', font=font_big, fill=(255, 220, 90))
        dr.rectangle((0, H - 52, W, H), fill=(16, 19, 23))
        dr.text((16, H - 44), f'头偏航 {hy:+6.1f}°   尾偏航 {ty:+6.1f}°   '
                              f'尾俯仰 {tp:+6.1f}°', font=font_mid, fill=(170, 200, 230))
        img.save(f'{a.out}_az{az:03d}.png')
        tiles.append(img.resize((W // 2, H // 2)))
        print(f'  方位角 {az:3d}°: 摆动脚 {sz:5.1f}mm 质心距 {cd:5.1f}mm '
              f'倾角 {tilt:4.1f}° 头 {hy:+5.0f}° 尾偏 {ty:+5.0f}° 尾俯 {tp:+5.0f}°',
              flush=True)
    sheet = Image.new('RGB', (W, H * 2), (10, 12, 16))
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % 2) * (W // 2), (k // 2) * (H // 2)))
    sheet.save(f'{a.out}_sheet.png')
    renderer.close()
    print(f'[write] {a.out}_az*.png 和 {a.out}_sheet.png')


if __name__ == '__main__':
    main()
