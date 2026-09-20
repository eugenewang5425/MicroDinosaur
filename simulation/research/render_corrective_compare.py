"""v2 纠正轮对比视频:台阶速度档 + 刹车延迟档并排渲染。

两个视频(都带每行标签与实测读数):
  compare_steps.mp4  台阶 0.2 / 0.35 / 0.55 m/s 三档 —— 看慢速档卡在哪
  compare_stop.mp4   刹车 5ms / 15ms 两档 —— 看最快档为什么反而停不住
标签与读数来自 evaluate() 的实测指标,不是旁白。
"""
from pathlib import Path
import sys
import json
import math
import argparse

import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from corrective_common import OUT  # noqa: E402
from evaluate_owned_terrain import cases, key, experiment, evaluate  # noqa: E402
from terrain_skill_eval import ground_height  # noqa: E402
from evaluate_policy import sha  # noqa: E402

W, H = 624, 384            # 均为 16 的倍数,避免 ffmpeg 补边
FONT = 'C:/Windows/Fonts/msyh.ttc'
POLICY = Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_corrective_v2'
              '/20260917_train_512x201/candidate.onnx')
SEED = 201
HOLD_S = 2.0


def pick(kind, delay=None, speed=None):
    if kind == 'steps':
        c = next(c for c in cases() if c['terrain'] == 'steps_10' and c['speed'] == speed)
        return dict(c, delay=delay or 10)
    c = next(c for c in cases() if c['program'] == 'resume' and c['delay'] == 5)
    return dict(c, delay=delay)


def render_row(case, seed, label, result_holder):
    """跑一个案例并把每 0.04s 渲染成一行帧。"""
    e = experiment(str(POLICY), case)
    s = e.sim
    s.model.vis.global_.offwidth = W
    s.model.vis.global_.offheight = H
    renderer = mujoco.Renderer(s.model, height=H, width=W)
    cam = mujoco.MjvCamera()
    # 固定机位(关键):跟随相机会让"卡住"和"通过"都居中,看不出距离差。
    # 台阶对准台阶中段(4.0-6.4m),刹车对准制动区(0-1.6m)。
    if case['terrain'] == 'steps_10':
        # steps_10 = 台阶高 10mm,共 3 级(地面 >0.30m 起每 0.18m 升一级到 0.66m),
        # 不是"10 级"。对准台阶口,通过的会走出画面右缘。
        cam.azimuth, cam.elevation, cam.distance = 112., -16., 1.15
        cam.lookat[:] = [.72, 0., .055]
    else:
        cam.azimuth, cam.elevation, cam.distance = 108., -15., 2.2
        cam.lookat[:] = [1.05, 0., .06]
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    frames = []

    prev = s.substep_callback

    def cap():
        prev()
        if not e.active:
            return
        t = s.data.time - 6.0            # 去预热
        if t <= 1e-8:
            return
        if frames and t - frames[-1][0] < .04 - 1e-9:
            return
        v = s.view(s.data.time)
        renderer.update_scene(s.data, camera=cam, scene_option=opt)
        frames.append((t, Image.fromarray(renderer.render()).copy(), float(v[1]), float(v[13])))

    s.substep_callback = cap
    m, t = evaluate(e, case, seed)
    s.substep_callback = prev
    renderer.close()
    # 末态冻结 HOLD_S 秒
    if frames:
        for k in range(int(HOLD_S / .04)):
            frames.append((frames[-1][0] + .04, frames[-1][1], frames[-1][2], frames[-1][3]))
    result_holder.append((label, case, m, frames))
    print(f'[{label}] fell={m["fell"]} '
          f'前进={m.get("forward_displacement_m", float("nan")):.2f}m '
          f'停稳={(m.get("post_stop") or {}).get("planar_speed_mean_mm_s", float("nan")):.1f}mm/s '
          f'越限={np.degrees(m.get("joint_limit_violation_max_rad", 0)):.2f}° '
          f'帧={len(frames)}', flush=True)


def compose(rows, dest, title):
    big = ImageFont.truetype(FONT, 24)
    mid = ImageFont.truetype(FONT, 19)
    small = ImageFont.truetype(FONT, 16)
    n = max(len(f) for _, _, _, f in rows)
    HEAD = 28
    grid = Image.new('RGB', (W, HEAD + H * len(rows)), (12, 15, 18))
    writer = imageio.get_writer(str(dest), fps=25, codec='libx264', quality=7)
    for i in range(n):
        for r, (label, case, m, frames) in enumerate(rows):
            fr = frames[min(i, len(frames) - 1)][1]
            grid.paste(fr, (0, HEAD + r * H))
        dr = ImageDraw.Draw(grid)
        for r, (label, case, m, frames) in enumerate(rows):
            y = HEAD + r * H
            hd, sub = label.split('|')
            dr.rectangle((0, y, W, y + 54), fill=(18, 21, 25))
            dr.text((10, y + 4), hd, font=big, fill=(255, 220, 90))
            t = frames[min(i, len(frames) - 1)][0]
            dr.text((10, y + 31), f'{sub}   t = {t:5.2f}s', font=small,
                    fill=(170, 180, 190))
            # 右侧读数(与左侧标签分栏,不重叠)
            if 'resume' in str(case.get('program')):
                sp = (m.get('post_stop') or {}).get('planar_speed_mean_mm_s', float('nan'))
                txt = f'停稳残速 {sp:5.1f} mm/s'
                note = '门 10'
                col = (120, 220, 120) if sp <= 10 else (255, 150, 90)
            else:
                fwd = m.get('forward_displacement_m', 0.)
                txt = f'前进 {fwd:4.2f} m'
                note = '门 0.80'
                col = (120, 220, 120) if fwd >= .8 else (255, 150, 90)
            dr.text((W - 210, y + 4), txt, font=mid, fill=col)
            ex = np.degrees(m.get('joint_limit_violation_max_rad', 0.))
            ex_txt = f'关节越限 {ex:.2f}° (门 1.15°)' if ex > 0 else '关节越限 0.00°'
            dr.text((W - 210, y + 31), ex_txt, font=small,
                    fill=(255, 150, 90) if ex > np.degrees(.02) else (150, 160, 170))
            dr.text((W - 62, y + 31), note, font=small, fill=(120, 130, 140))
            # 底部条:台阶=位置(通过的会走出画面),刹车=速度(核心是停没停)
            bx0, bx1, by = 10, W - 10, y + H - 18
            dr.rectangle((bx0, by, bx1, by + 8), fill=(38, 44, 52))
            fr_now = frames[min(i, len(frames) - 1)]
            if case['terrain'] == 'steps_10':
                span, xnow = 2.0, fr_now[2]
                gx = bx0 + int((bx1 - bx0) * .8 / span)
                dr.line((gx, by - 3, gx, by + 11), fill=(255, 210, 70), width=2)
                dr.text((gx + 3, by - 16), '门 0.8m', font=small, fill=(255, 210, 70))
                px = bx0 + int((bx1 - bx0) * min(max(xnow, 0.), span) / span)
                dr.rectangle((bx0, by, px, by + 8), fill=(90, 170, 255))
                dr.text((bx0 + 2, by - 16), f'x = {xnow:4.2f} m', font=small,
                        fill=(160, 200, 255))
            else:
                span, vnow = .32, abs(fr_now[3])
                gx = bx0 + int((bx1 - bx0) * .01 / span)   # 10 mm/s 门槛
                dr.line((gx, by - 3, gx, by + 11), fill=(255, 210, 70), width=2)
                dr.text((gx + 3, by - 16), '门 10mm/s', font=small, fill=(255, 210, 70))
                px = bx0 + int((bx1 - bx0) * min(vnow, span) / span)
                dr.rectangle((bx0, by, px, by + 8),
                             fill=(255, 120, 90) if vnow > .01 else (120, 220, 120))
                dr.text((bx0 + 2, by - 16), f'速度 {vnow * 1000:5.1f} mm/s', font=small,
                        fill=(255, 180, 150) if vnow > .01 else (150, 220, 150))
            dr.line((0, y, W, y), fill=(60, 70, 80), width=2)
        dr.rectangle((0, 0, W, HEAD), fill=(24, 28, 33))
        dr.text((8, 4), title, font=small, fill='white')
        writer.append_data(np.asarray(grid))
    writer.close()
    print(f'[write] {dest}')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', choices=('steps', 'stop', 'all'), default='all')
    a = ap.parse_args()
    print(f'策略 {POLICY.name} sha={sha(POLICY)[:12]} seed={SEED}', flush=True)

    if a.which in ('steps', 'all'):
        rows = []
        for v, note in ((.2, '慢速档:预期卡住'), (.35, '中速档:预期通过'), (.55, '快速档:预期通过')):
            render_row(pick('steps', speed=v), SEED,
                       f'台阶 3 级×10mm(总高 30mm) · 指令 {v} m/s|{note}', rows)
        compose(rows, OUT / 'videos' / 'compare_steps.mp4',
                '台阶对比 · v2 模型 · 同一初态/延迟 · 固定机位')

    if a.which in ('stop', 'all'):
        rows = []
        for d, note in ((5, '最快响应:预期停不住'), (15, '慢响应:预期停稳')):
            render_row(pick('stop', delay=d), SEED,
                       f'平地急停(0.32→0 m/s) · 控制延迟 {d} ms|{note}', rows)
        compose(rows, OUT / 'videos' / 'compare_stop.mp4',
                '急停对比 · v2 模型 · 同一初态/速度 · 固定机位')


if __name__ == '__main__':
    main()
