"""直道障碍赛视频的离线渲染:读 course_frames.npz,不重跑仿真。

相机约定(实测选定,2026-09-17):
- 显式 cam.type=mjCAMERA_FREE(不设的话 follow 行为依版本而异,台阶段会贴地穿模);
- 台阶段(3.8-6.6m)用 az=90 正侧面 + 仰角 -18°,看爬阶轮廓;
- 其余路段 az=135 后侧追拍 + 仰角 -14°。
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
OUT = ROOT / '20260914_contact_motion'
W, H = 760, 760
FONT = 'C:/Windows/Fonts/msyh.ttc'
SECTIONS = [('平地', 0., 4.), ('台阶', 4., 6.4), ('平地', 6.4, 8.),
            ('凹凸路', 8., 12.), ('平地', 12., 13.5), ('迷宫', 13.5, 17.5),
            ('终点直道', 17.5, 20.)]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    z = np.load(OUT / 'course_frames.npz')
    frames = z['qpos']
    rows = z['rows']
    stats = json.loads((OUT / 'course_stats.json').read_text(encoding='utf-8'))
    m = mujoco.MjModel.from_binary_path('x', assets={
        'x': (OUT / 'obstacle_plant' / 'nominal.mjb').read_bytes()})
    data = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=H, width=W)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    mid_f = ImageFont.truetype(FONT, 26)
    mid = ImageFont.truetype(FONT, 21)
    small = ImageFont.truetype(FONT, 17)

    sc = 660. / 20.
    ox, oy = 50., H / 2.

    def mp(p):
        # 右栏小地图:x 加 W 偏移 + 50 边距;y 放大 3.2 倍便于看清迷宫走向
        return (W + ox + p[0] * sc, oy - p[1] * sc * 3.2)

    dt_f = 32 * .00125
    xy = frames[:, :2]
    tilt_max = float(rows[:, 8].max())
    finish_t = stats.get('finish_time_s', float('nan'))

    writer = imageio.get_writer(str(OUT / 'course_video.mp4'), fps=25,
                                codec='libx264', quality=7)
    n = len(frames)
    for i in range(n):
        data.qpos[:] = frames[i]
        mujoco.mj_forward(m, data)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        if 3.8 <= xy[i, 0] < 6.6:
            cam.distance, cam.azimuth, cam.elevation = 1.1, 90., -18
        else:
            cam.distance, cam.azimuth, cam.elevation = .8, 135., -14
        cam.lookat[:] = [xy[i, 0], xy[i, 1], frames[i][2] + .02]
        renderer.update_scene(data, camera=cam, scene_option=opt)
        left = Image.fromarray(renderer.render())
        dl = ImageDraw.Draw(left)
        dl.rectangle((0, 0, W, 40), fill=(20, 24, 28))
        dl.text((12, 7), 'MicroDinosaur 直道障碍赛 20 m: 台阶→凹凸路→迷宫',
                font=mid_f, fill='white')

        img = Image.new('RGB', (2 * W, H), (14, 17, 20))
        img.paste(left, (0, 0))
        dr = ImageDraw.Draw(img)
        for name, x0, x1 in SECTIONS:
            color = {'台阶': (60, 90, 130), '凹凸路': (95, 80, 55),
                     '迷宫': (80, 60, 100)}.get(name, (40, 46, 52))
            dr.rectangle([mp([x0, 1])[0], oy - 150, mp([x1, 0])[0], oy + 150],
                         fill=color)
        for o in json.loads((OUT / 'obstacle_plant' / 'course.json').read_text(
                encoding='utf-8'))['obstacles']:
            color = (150, 190, 255) if o['label'].startswith(('stair', 'bump')) \
                else (220, 140, 230)
            dr.rectangle([mp([o['x'] - o['hx'], o['y'] + o['hy']])[0],
                          mp([0, o['y'] + o['hy']])[1],
                          mp([o['x'] + o['hx'], 0])[0],
                          mp([0, o['y'] - o['hy']])[1]], fill=color)
        wp = json.loads((OUT / 'obstacle_plant' / 'course.json').read_text(
            encoding='utf-8'))['waypoints']
        dr.line([mp(p) for p in np.asarray(wp)], fill=(140, 200, 140), width=3)
        pts = [mp(p) for p in xy[:max(i, 1)]]
        if len(pts) > 1:
            dr.line(pts, fill=(90, 170, 255), width=3)
        q = frames[i]
        yaw = math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))
        px, py = mp(xy[i])
        dr.ellipse((px - 6, py - 6, px + 6, py + 6), fill=(255, 90, 90))
        dr.line((px, py, px + 26 * math.cos(yaw), py - 26 * math.sin(yaw)),
                fill=(255, 90, 90), width=4)
        for name, x0, x1 in SECTIONS:
            if name == '平地' and x1 - x0 < 1.5:
                continue
            dr.text((mp([x0, 0])[0] + 4, oy + 156), name, font=small,
                    fill=(170, 180, 190))
        dr.text((mp([19.7, 0])[0] - 20, oy - 176), '终点', font=small,
                fill=(255, 210, 70))
        dr.rectangle((W, 0, 2 * W, 44), fill=(20, 24, 28))
        sec_now = next((nm for nm, x0, x1 in SECTIONS if x0 <= xy[i, 0] < x1), '—')
        dr.text((W + 14, 8), f'当前路段 {sec_now}   x = {xy[i, 0]:5.2f} / 20 m',
                font=mid, fill='white')
        dr.rectangle((W, H - 92, 2 * W, H), fill=(20, 24, 28))
        dr.text((W + 14, H - 84), '指令: 前速 0.45 m/s(台阶保持速度,动量即爬阶)  '
                                  '迷宫段 0.26 m/s 短前瞻', font=small, fill=(190, 200, 210))
        dr.text((W + 14, H - 56), f'最大倾角 {tilt_max:.1f}°   最大前进 x '
                                  f'{xy[:i + 1, 0].max():.2f} m', font=small,
                fill=(190, 200, 210))
        fin = f'{finish_t:.1f} s' if stats.get('finished') and i >= finish_t * 25 \
            else '未完赛'
        dr.text((W + 14, H - 30), f'完赛: {fin}', font=small, fill=(255, 210, 70))
        writer.append_data(np.asarray(img))
    writer.close()
    renderer.close()
    print(f'[write] {OUT / "course_video.mp4"}  {n}帧 @25fps = {n * dt_f:.1f}s')


if __name__ == '__main__':
    main()
