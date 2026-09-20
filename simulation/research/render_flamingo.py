"""Flamingo 单脚站立视频:侧视 + 读数(摆动脚高/质心距/倾角/头尾配平观察)。"""
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

W, H = 640, 400
FONT = 'C:/Windows/Fonts/msyh.ttc'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', default='Flamingo 单脚站立')
    a = ap.parse_args()
    task, cfg = build_config(envs=1, seed=47, standing_prob=0.)
    env = ManagerBasedRlEnv(cfg.env, device='cpu')
    session = ort.InferenceSession(a.checkpoint, providers=['CPUExecutionProvider'])
    robot = env.scene['robot']

    from terrain_skill_eval import freeze_plant
    plant_dir = freeze_plant('flat')
    raw_model = mujoco.MjModel.from_binary_path(
        'nominal.mjb', assets={'nominal.mjb': (plant_dir / 'nominal.mjb').read_bytes()})
    renderer = mujoco.Renderer(raw_model, height=H, width=W)
    _rd = mujoco.MjData(raw_model)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    cam.azimuth, cam.elevation, cam.distance = 90., -8., .75

    names = [n.split('/')[-1] for n in robot.joint_names]
    jid = {k: 7 + robot.find_joints([k])[0][0] for k in ('head_yaw', 'tail_yaw')}
    sites = [n.split('/')[-1] for n in robot.site_names]
    swing_site = sites.index('left_foot')
    stance_site = sites.index('right_foot')
    root_bid = int(robot.indexing.root_body_id)

    big = ImageFont.truetype(FONT, 24)
    mid = ImageFont.truetype(FONT, 20)
    small = ImageFont.truetype(FONT, 17)
    mid_f = big
    obs, info = env.reset()
    frames = []
    for step in range(150):
        with torch.no_grad():
            o = obs['actor'] if isinstance(obs, dict) else obs
            if o.dim() > 2:
                o = o.flatten(1)
            onx = o.detach().numpy().astype(np.float32)
            act = np.stack([session.run(None, {'obs': onx[b:b + 1]})[0][0]
                            for b in range(onx.shape[0])])
        obs, rew, term, trunc, extras = env.step(torch.as_tensor(act))
        d = env.sim.data
        sz = float(robot.data.site_pos_w[0, swing_site, 2]) * 1000
        com = np.asarray(d.subtree_com[0, root_bid, :2])
        foot = robot.data.site_pos_w[0, stance_site, :2].numpy()
        cd = float(np.linalg.norm(com - foot)) * 1000
        gz = robot.data.projected_gravity_b[0].numpy()
        tilt = math.degrees(math.acos(min(1., abs(gz[2]))))
        hy = math.degrees(d.qpos[0, jid['head_yaw']])
        ty = math.degrees(d.qpos[0, jid['tail_yaw']])

        cam.lookat[:] = [d.qpos[0, 0], d.qpos[0, 1], .09]
        _rd.qpos[:] = d.qpos
        _rd.qvel[:] = d.qvel
        mujoco.mj_forward(raw_model, _rd)
        renderer.update_scene(_rd, camera=cam, scene_option=opt)
        img = Image.fromarray(renderer.render()).copy()
        dr = ImageDraw.Draw(img)
        dr.rectangle((0, 0, W, 34), fill=(18, 21, 25))
        dr.text((10, 5), ('%s   t=%5.1fs' % (a.label, step * .04)), font=mid_f, fill='white')
        dr.rectangle((0, H - 122, W, H), fill=(18, 21, 25))
        col = (120, 220, 120) if sz > 40 else (150, 200, 255)
        dr.text((10, H - 116), ('摆动脚(左)离地 %6.1f mm   目标 50 mm' % sz), font=mid, fill=col)
        dr.text((10, H - 88), ('质心-支撑脚水平距 %5.1f mm   倾角 %4.1f°' % (cd, tilt)), font=small, fill=(180, 195, 210))
        dr.text((10, H - 60), ('配平观察: 头偏航 %+6.1f°   尾偏航 %+6.1f°' % (hy, ty)), font=small, fill=(180, 195, 210))
        dr.text((10, H - 32), '官方 flamingo 配方 - 姿态内出生 - 保持 6s', font=small, fill=(150, 160, 170))
        frames.append(np.asarray(img))
    renderer.close()

    writer = imageio.get_writer(a.out, fps=25, codec='libx264', quality=7)
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f'[write] {a.out}  {len(frames)}帧')


if __name__ == '__main__':
    main()
