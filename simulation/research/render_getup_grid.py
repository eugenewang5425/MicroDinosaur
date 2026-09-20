"""四方位起身视频:侧视图 + 俯视偏航罗盘,一行一个方位。

数据来自 evaluate_contact_motion.py 已产出的 npz(qpos @ 0.04s),不再重跑仿真。
为什么必须带偏航罗盘:三个方位的窗口判据已 97.6-99.4%,唯一挡路的是"起身最后
0.6 秒绕竖轴甩 59-115°"(见 GETUP_TASK.md 10.3)—— 侧视图里这个甩几乎看不出来,
俯视罗盘 + 数值才把它钉在画面上。
"""
from pathlib import Path
import sys
import json

import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from build_contact_motion_bank import experiment  # noqa: E402

CASE = ROOT / '20260914_contact_motion' / (sys.argv[1] if len(sys.argv) > 1 else 'getup_video_36549')
ORDER = ['left', 'right', 'front', 'back']
SIDE_W, SIDE_H, TOP_W = 760, 428, 428
FONT = 'C:/Windows/Fonts/msyh.ttc'


def quat_yaw(q):
    qw, qx, qy, qz = q[3], q[4], q[5], q[6]
    import math
    return math.degrees(math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    e, _ = experiment(941)
    model, body = e.sim.model, e.sim.body
    renderer = mujoco.Renderer(model, height=SIDE_H, width=SIDE_W)
    data = mujoco.MjData(model)

    cam_side = mujoco.MjvCamera()
    cam_side.distance, cam_side.azimuth, cam_side.elevation = .8, 35., -12
    cam_top = mujoco.MjvCamera()
    cam_top.distance, cam_top.azimuth, cam_top.elevation = .62, 90., -84
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0

    big = ImageFont.truetype(FONT, 30)
    mid = ImageFont.truetype(FONT, 24)
    small = ImageFont.truetype(FONT, 19)

    rows = json.loads((CASE / 'summary.json').read_text(encoding='utf-8'))
    meta = {r['key']: r for r in rows}

    writers = {}
    frames_buf = {d: [] for d in ORDER}
    yaws_all = {}
    for d in ORDER:
        npz = CASE / f'{d}_c10_s941.npz'
        q = np.load(npz, allow_pickle=True)['qpos']
        xy = q[:, :2]
        cam_side.lookat[:] = [xy[:, 0].mean(), xy[:, 1].mean(), .10]
        cam_top.lookat[:] = [xy[:, 0].mean(), xy[:, 1].mean(), .05]
        yaws = np.rad2deg(np.unwrap(np.deg2rad([quat_yaw(f) for f in q])))
        yaws_all[d] = yaws
        for i, frame in enumerate(q):
            data.qpos[:] = frame
            mujoco.mj_forward(model, data)
            # 侧视
            renderer.update_scene(data, camera=cam_side, scene_option=opt)
            side = Image.fromarray(renderer.render())
            # 俯视
            renderer.update_scene(data, camera=cam_top, scene_option=opt)
            top = Image.fromarray(renderer.render()).resize((TOP_W, TOP_W))
            c = data.subtree_com[body][:2]
            px = TOP_W / 2 + (c[0] - cam_top.lookat[0]) / (2 * cam_top.distance) * TOP_W
            py = TOP_W / 2 - (c[1] - cam_top.lookat[1]) / (2 * cam_top.distance) * TOP_W
            dr = ImageDraw.Draw(top)
            y = np.deg2rad(yaws[i])
            L = 78
            dr.line((px, py, px + L * np.cos(y), py - L * np.sin(y)), fill=(255, 70, 70), width=5)
            dr.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(255, 70, 70))
            dr.rectangle((0, 0, TOP_W, 34), fill=(20, 24, 28))
            dr.text((8, 5), f'俯视 yaw={yaws[i]:+8.1f}°', font=small, fill='white')

            W = SIDE_W + TOP_W
            img = Image.new('RGB', (W, SIDE_H), (12, 15, 18))
            img.paste(side, (0, 0))
            img.paste(top, (SIDE_W, 0))
            dd = ImageDraw.Draw(img)
            dd.rectangle((0, 0, 330, 40), fill=(20, 24, 28))
            dd.text((10, 5), f'{d}  t={i * .04:5.2f}s', font=big, fill=(255, 220, 90))
            dd.rectangle((0, SIDE_H - 36, W, SIDE_H), fill=(20, 24, 28))
            tilt = np.degrees(np.arccos(np.clip(data.xmat[body].reshape(3, 3)[2, 2], -1, 1)))
            dd.text((10, SIDE_H - 32),
                    f'倾角 {tilt:5.1f}°   偏航 {yaws[i]:+8.1f}°   站起需 1.9s,偏航全丢在最后 0.6s',
                    font=small, fill=(200, 210, 220))
            frames_buf[d].append(np.asarray(img))

        m = meta[f'{d}_c10_s941']
        print(f"{d:6s} 落地 {m['fallen_yaw_deg']:+7.1f}° -> 末态 {m['final_yaw_deg']:+7.1f}°"
              f"  漂移 {m['yaw_drift_deg']:6.1f}°(门30°)  帧数 {len(frames_buf[d])}")

    out = ROOT / '20260914_contact_motion' / 'getup_yaw_grid.mp4'
    H = SIDE_H * len(ORDER)
    W = SIDE_W + TOP_W
    writer = imageio.get_writer(str(out), fps=25, codec='libx264', quality=8)
    n = len(frames_buf['left'])
    for i in range(n):
        sheet = Image.new('RGB', (W, H), (12, 15, 18))
        dd = ImageDraw.Draw(sheet)
        for k, d in enumerate(ORDER):
            sheet.paste(Image.fromarray(frames_buf[d][i]), (0, k * SIDE_H))
            if k:
                dd.line((0, k * SIDE_H, W, k * SIDE_H), fill=(60, 70, 80), width=2)
        # 行内标签:落地->末态偏航
        for k, d in enumerate(ORDER):
            m = meta[f'{d}_c10_s941']
            dd.text((SIDE_W + 12, k * SIDE_H + SIDE_H - 70),
                    f"漂移 {m['yaw_drift_deg']:5.1f}° / 门 30°",
                    font=mid, fill=(255, 160, 60) if m['yaw_drift_deg'] > 30 else (120, 220, 120))
        writer.append_data(np.asarray(sheet))
    writer.close()
    renderer.close()
    print(f'[write] {out}  {W}x{H}  {n}帧 @25fps = {n / 25:.1f}s')


if __name__ == '__main__':
    main()
