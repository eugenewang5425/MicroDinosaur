"""起身偏航诊断:俯视图 + 躯干朝向箭头,看偏航是在哪一帧、以什么方式丢的。

背景(实测):三个方向的窗口判据已经 97.6-99.4%,站姿 8.4°/脚底 0.6°/倾角 5.7° 全部正确,
唯一挡路的是 `朝向保持 <= 30°`。把 npz 的 qpos 逐帧喂回模型重算 trunk 偏航:
    首帧 == fallen_yaw、末帧 == final_yaw(逐位吻合),
    0-6s 变化 -114.9/-113.8/+59.2°,6-12s 变化 +-0.6°。
=> 偏航 100% 丢在起身动作里,不是站定后的漂移。所以判据量的不是"站不稳",
   而是"站起来之后脸朝哪"。这张图就是回答"它到底怎么转的"。

输出 2x4 网格:上行俯视图(带朝向箭头),下行偏航/倾角时间曲线。
"""
from pathlib import Path
import sys

import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from build_contact_motion_bank import experiment  # noqa: E402

CASE = ROOT / '20260914_contact_motion' / 'getup_video_36549'
FONT = 'C:/Windows/Fonts/msyh.ttc'
GRID = (2, 2)                      # 2x2 个方位,每个方位一行(俯视 + 曲线)
SAMPLE_S = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 11.6)


def trunk_yaw(model, data, body, qpos):
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    r = data.xmat[body].reshape(3, 3)
    return float(np.degrees(np.arctan2(r[1, 0], r[0, 0]))), r


def render_row(model, body, frames, title):
    """一个方位一行:8 个俯视快照拼横排,每个格子里画朝向箭头和数值。"""
    renderer = mujoco.Renderer(model, height=340, width=340)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = .62, 90., -84.
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    data = mujoco.MjData(model)
    xy = frames[:, :2]
    cam.lookat[:] = [xy[:, 0].mean(), xy[:, 1].mean(), .05]
    yaws = [trunk_yaw(model, data, body, q)[0] for q in frames]
    yaws = np.rad2deg(np.unwrap(np.deg2rad(yaws)))
    dt = 12. / len(frames)

    font = ImageFont.truetype(FONT, 19)
    cells = []
    for t in SAMPLE_S:
        i = min(len(frames) - 1, int(round(t / dt)))
        data.qpos[:] = frames[i]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam, scene_option=opt)
        img = Image.fromarray(renderer.render())
        dr = ImageDraw.Draw(img)
        c = data.subtree_com[body][:2]
        px = 170 + (c[0] - cam.lookat[0]) / (2 * cam.distance) * 340
        py = 170 - (c[1] - cam.lookat[1]) / (2 * cam.distance) * 340
        y = np.deg2rad(yaws[i])
        L = 62
        dr.line((px, py, px + L * np.cos(y), py - L * np.sin(y)), fill=(255, 70, 70), width=4)
        dr.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(255, 70, 70))
        dr.rectangle((0, 0, 340, 30), fill=(20, 24, 28))
        dr.text((7, 4), f't={t:4.1f}s  yaw={yaws[i]:+7.1f}°', font=font, fill='white')
        cells.append(img)
    renderer.close()
    return cells, yaws


def curve_row(yaws, tilt, title):
    """偏航(红)与倾角(蓝)时间曲线,标出 30° 门与 fallen 起点。"""
    W, H = 340 * len(SAMPLE_S), 340
    img = Image.new('RGB', (W, H), (20, 24, 28))
    dr = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 20)
    small = ImageFont.truetype(FONT, 17)
    n = len(yaws)
    t = np.arange(n)

    def X(i):
        return 60 + i / (n - 1) * (W - 90)

    def Y(v):
        return H - 55 - (v + 200) / 400 * (H - 100)

    for v in range(-160, 161, 80):
        dr.line((50, Y(v), W - 25, Y(v)), fill=(48, 56, 64), width=1)
        dr.text((4, Y(v) - 10), f'{v:+4d}°', font=small, fill=(140, 150, 160))
    for v in (30, -30):
        dr.line((50, Y(v), W - 25, Y(v)), fill=(200, 160, 40), width=2)
    dr.text((W - 130, Y(30) - 24), '朝向门 ±30°', font=small, fill=(220, 180, 60))
    dr.line(list(zip([X(i) for i in t], [Y(y) for y in yaws])), fill=(255, 90, 90), width=4)
    dr.line(list(zip([X(i) for i in t], [Y(v) for v in tilt])), fill=(90, 170, 255), width=3)
    for i in (0, 150, 225, n - 1):
        dr.text((X(i) - 22, H - 40), f'{i * 12. / n:.0f}s', font=small, fill=(160, 170, 180))
    dr.text((62, H - 40), f'红=躯干偏航(共 {yaws[-1] - yaws[0]:+.0f}°)  蓝=倾角  '
                          f'灰格 80° 间隔', font=small, fill=(190, 200, 210))
    return img


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    names = [d.name for d in sorted(CASE.glob('*_c10_s941.mp4'))]
    order = ['left', 'right', 'front', 'back']
    e, _ = experiment(941)
    model, body = e.sim.model, e.sim.body
    tile = 340
    W, H = tile * len(SAMPLE_S), tile * 2 * len(order)
    sheet = Image.new('RGB', (W, H), (12, 15, 18))
    big = ImageFont.truetype(FONT, 40)
    dr = ImageDraw.Draw(sheet)
    for k, name in enumerate(order):
        npz = CASE / f'{name}_c10_s941.npz'
        if not npz.exists():
            continue
        frames = np.load(npz, allow_pickle=True)['qpos']
        cells, yaws = render_row(model, body, frames, name)
        data = mujoco.MjData(model)
        tilt = []
        for q in frames:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            tilt.append(np.degrees(np.arccos(np.clip(data.xmat[body].reshape(3, 3)[2, 2], -1, 1))))
        for j, c in enumerate(cells):
            sheet.paste(c, (j * tile, k * 2 * tile))
        sheet.paste(curve_row(yaws, tilt, name), (0, (k * 2 + 1) * tile))
        dr.text((12, k * 2 * tile + 4), f'{name}  yaw {yaws[0]:+.0f}° -> {yaws[-1]:+.0f}°',
                font=big, fill=(255, 220, 90))
    out = ROOT / '20260914_contact_motion' / 'yaw_diagnosis.png'
    sheet.save(out)
    print(f'[write] {out}  {sheet.size[0]}x{sheet.size[1]}')


if __name__ == '__main__':
    main()
