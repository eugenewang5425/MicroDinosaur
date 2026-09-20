"""单脚站立的效果视频:侧视图 + 抬脚高度读数。

相机用正侧面(azimuth=90),抬脚高度在侧视图里最直观;右栏实时显示双脚高度、
支撑脚接触力、倾角——数值来自物理,不是旁白。
"""
from pathlib import Path
import sys
import argparse

import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from corrective_common import OUT  # noqa: E402
from evaluate_owned_terrain import experiment, evaluate  # noqa: E402
from evaluate_policy import sha  # noqa: E402

W, H = 640, 400
FONT = 'C:/Windows/Fonts/msyh.ttc'


def foot_low(model, data, gid):
    mid = model.geom_dataid[gid]
    a = model.mesh_vertadr[mid]
    n = model.mesh_vertnum[mid]
    v = model.mesh_vert[a:a + n] @ data.geom_xmat[gid].reshape(3, 3).T \
        + data.geom_xpos[gid]
    return float(v[:, 2].min())


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--label', default='单脚站立')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--seconds', type=float, default=16.)
    ap.add_argument('--cmd-vy', type=float, default=1.,
                    help='抬脚指令:+1 抬左脚,-1 抬右脚(必须给:random 训练的模型没见过 0)')
    a = ap.parse_args()
    case = dict(terrain='flat', scenario='stand', program='legacy', depth=0,
                delay=10, speed=None, seconds=a.seconds)
    e = experiment(str(a.policy), case)

    def fixed_cmd(t, scenario, _v=a.cmd_vy):
        u = np.zeros(18)
        u[1] = _v
        return u
    e.user_command = fixed_cmd
    s = e.sim
    m = s.model
    fl = m.geom('robot/left_foot_collision').id
    fr = m.geom('robot/right_foot_collision').id
    floor = m.geom('terrain').id
    m.vis.global_.offwidth = W
    m.vis.global_.offheight = H
    renderer = mujoco.Renderer(m, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 90., -8., .75
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    mid_f = ImageFont.truetype(FONT, 24)
    mid = ImageFont.truetype(FONT, 20)
    small = ImageFont.truetype(FONT, 17)
    frames = []
    prev = s.substep_callback

    def cap():
        prev()
        if not e.active:
            return
        t = s.data.time - 6.
        if t <= 1e-8 or (frames and t - frames[-1][0] < .04 - 1e-9):
            return
        d = s.data
        w = np.zeros(6)
        load = [0., 0.]
        for ci, c in enumerate(d.contact):
            if floor not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == floor else c.geom1
            mujoco.mj_contactForce(m, d, ci, w)
            if other == fl:
                load[0] += max(0., w[0])
            elif other == fr:
                load[1] += max(0., w[0])
        rot = d.xmat[s.body].reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1))))
        cam.lookat[:] = [d.qpos[0], d.qpos[1], .09]   # 跟随(抬脚会蹒跚移动)
        renderer.update_scene(d, camera=cam, scene_option=opt)
        img = Image.fromarray(renderer.render()).copy()
        dr = ImageDraw.Draw(img)
        hl, hr = foot_low(m, d, fl) * 1000, foot_low(m, d, fr) * 1000
        dr.rectangle((0, 0, W, 34), fill=(18, 21, 25))
        dr.text((10, 5), f'{a.label}  t={t:5.1f}s', font=mid_f, fill='white')
        dr.rectangle((0, H - 96, W, H), fill=(18, 21, 25))
        dr.text((10, H - 90), f'左脚高度 {hl:6.1f} mm   右脚高度 {hr:6.1f} mm',
                font=mid, fill=(150, 220, 150) if max(hl, hr) > 8 else (200, 200, 120))
        dr.text((10, H - 62), f'接触力  左 {load[0]:5.2f} N   右 {load[1]:5.2f} N',
                font=small, fill=(180, 195, 210))
        cmd_txt = '抬左脚' if a.cmd_vy > 0 else '抬右脚'
        dr.text((10, H - 36), f'倾角 {tilt:5.1f}°   指令: {cmd_txt}',
                font=small, fill=(180, 195, 210))
        frames.append((t, img))

    s.substep_callback = cap
    metrics, _ = evaluate(e, case, a.seed)
    s.substep_callback = prev
    renderer.close()
    for k in range(50):                     # 末态冻结 2s
        frames.append((frames[-1][0] + .04, frames[-1][1]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(a.out), fps=25, codec='libx264', quality=7)
    for _, img in frames:
        writer.append_data(np.asarray(img))
    writer.close()
    hl = [foot_low(m, e.sim.data, fl) for _ in (0,)]
    print(f'[write] {a.out}  {len(frames)}帧  策略sha={sha(a.policy)[:12]}  '
          f'跌倒={metrics.get("fell")}')


if __name__ == '__main__':
    main()
