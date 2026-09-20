"""不同力度冲击下的前伸单脚站立对比视频。

四段递增: 0.15 / 0.25 / 0.4 / 0.6 m/s(超限展示真实摔倒), 每段 12 s。
模型用 0.4 边界级(抗扰最强)。冲击来自 env 的 push_robot 事件(速度突变 Δv,
侧向为主, 每 1-3 s 随机), 画面上以红色闪光+方向箭头标注冲击时刻。
"""
from pathlib import Path
import sys
import argparse
import math
from collections import deque

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

W, H = 1280, 720
RW, RH = 720, 720
FONT = 'C:/Windows/Fonts/msyh.ttc'
SEG_STEPS = 300            # 12 s @ 25 fps(策略 0.04 s/步)
LEVELS = [(.15, 0), (.25, 0), (.40, 0), (.60, 1), (.80, 2)]


def render_titlecard(draw_fn, lines):
    img = Image.new('RGB', (W, H), (12, 14, 17))
    dr = ImageDraw.Draw(img)
    draw_fn(dr, lines)
    return np.asarray(img)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    # ONNX 中文路径问题: 拷到 ASCII 临时路径
    import shutil, tempfile
    tmp_onnx = Path(tempfile.gettempdir()) / 'push_demo.onnx'
    shutil.copy2(a.checkpoint, tmp_onnx)
    session = ort.InferenceSession(str(tmp_onnx),
                                   providers=['CPUExecutionProvider'])

    big = ImageFont.truetype(FONT, 44)
    mid = ImageFont.truetype(FONT, 26)
    small = ImageFont.truetype(FONT, 20)
    tiny = ImageFont.truetype(FONT, 17)

    from terrain_skill_eval import freeze_plant
    plant_dir = freeze_plant('flat')
    raw_model = mujoco.MjModel.from_binary_path(
        'nominal.mjb', assets={'nominal.mjb': (plant_dir / 'nominal.mjb').read_bytes()})
    renderer = mujoco.Renderer(raw_model, height=RH, width=RW)
    _rd = mujoco.MjData(raw_model)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    cam.azimuth, cam.elevation, cam.distance = 90., -8., .78

    all_frames = []
    for lv, ov in LEVELS:
        # 标题卡 1 s
        lv_txt = f'{lv:.2f} m/s'
        tc = Image.new('RGB', (W, H), (12, 14, 17))
        td = ImageDraw.Draw(tc)
        td.text((W // 2 - 230, H // 2 - 90), f'侧向冲击 {lv_txt}', font=big,
                fill=(240, 240, 245) if not ov else (255, 120, 110))
        td.text((W // 2 - 300, H // 2 + 10),
                '速度突变 Δv · 每 1-3 秒随机一次 · 前后 0.6 倍', font=mid,
                fill=(160, 170, 180))
        td.text((W // 2 - 300, H // 2 + 55),
                ('超过训练上限, 展示真实边界' if ov == 1 else
                 ('极限测试' if ov == 2 else
                  f'整机 1.10 kg · 冲量 {1.098*lv:.2f} N·s')), font=small,
                fill=(130, 140, 150))
        all_frames.append(np.asarray(tc))

        task, cfg = build_config(envs=1, seed=47, fix_side=1.0, fix_pose=1.0,
                                 push=lv)
        env = ManagerBasedRlEnv(cfg.env, device='cpu')
        obs, _ = env.reset()
        robot = env.scene['robot']
        sites = [n.split('/')[-1] for n in robot.site_names]
        swing_site = sites.index('left_foot')
        stance_site = sites.index('right_foot')
        root_bid = int(robot.indexing.root_body_id)

        tilt_hist = deque(maxlen=150)
        prev_v = None
        flash = 0
        flash_dir = 1.
        pushes = 0
        fallen_at = None
        lookat = None

        for step in range(SEG_STEPS):
            with torch.no_grad():
                o = obs['actor'] if isinstance(obs, dict) else obs
                if o.dim() > 2:
                    o = o.flatten(1)
                onx = o.detach().numpy().astype(np.float32)
                act = np.stack([session.run(None, {'obs': onx[b:b + 1]})[0][0]
                                for b in range(onx.shape[0])])
            obs, rew, term, trunc, extras = env.step(torch.as_tensor(act))
            d = env.sim.data
            v = d.qvel[0, :2].clone()
            if prev_v is not None:
                dv = float(np.linalg.norm(v - prev_v))
                if dv > 0.12:                     # 速度突变 = 被推了
                    pushes += 1
                    flash = 4
                    flash_dir = 1. if (v - prev_v)[1] >= 0 else -1.
            prev_v = v

            sz = float(robot.data.site_pos_w[0, swing_site, 2]) * 1000
            com = np.asarray(d.subtree_com[0, root_bid, :2])
            foot = robot.data.site_pos_w[0, stance_site, :2].numpy()
            cd = float(np.linalg.norm(com - foot)) * 1000
            gz = robot.data.projected_gravity_b[0].numpy()
            tilt = math.degrees(math.acos(min(1., abs(gz[2]))))
            tilt_hist.append(tilt)
            if fallen_at is None and tilt > 70.:
                fallen_at = step * .04
            if flash > 0:
                flash -= 1

            # 渲染
            if lookat is None:
                lookat = np.array([d.qpos[0, 0], d.qpos[0, 1], .09])
            else:
                lookat = lookat * .9 + np.array(
                    [d.qpos[0, 0], d.qpos[0, 1], .09]) * .1
            cam.lookat[:] = lookat
            _rd.qpos[:] = d.qpos
            _rd.qvel[:] = d.qvel
            mujoco.mj_forward(raw_model, _rd)
            renderer.update_scene(_rd, camera=cam, scene_option=opt)
            img = Image.new('RGB', (W, H), (16, 19, 23))
            img.paste(Image.fromarray(renderer.render()).resize((RW, RH)), (0, 0))
            dr = ImageDraw.Draw(img)

            # 冲击闪光: 边框红光 + 方向箭头
            if flash > 0:
                wdt = 6 + flash * 2
                dr.rectangle([0, 0, RW - 1, wdt], fill=(255, 70, 60))
                dr.rectangle([0, RH - 1 - wdt, RW - 1, RH - 1], fill=(255, 70, 60))
                dr.rectangle([0, 0, wdt, RH - 1], fill=(255, 70, 60))
                dr.rectangle([RW - 1 - wdt, 0, RW - 1, RH - 1], fill=(255, 70, 60))
                ax = RW // 2 + int(flash_dir * 200)
                dr.polygon([(ax, 96), (ax - int(18 * flash_dir), 76),
                            (ax - int(18 * flash_dir), 116)], fill=(255, 90, 80))
                dr.text((RW // 2 - 46, 68), '受击!', font=mid, fill=(255, 90, 80))

            if fallen_at is not None:
                dr.text((RW // 2 - 120, RH // 2 - 40), '摔 倒', font=big,
                        fill=(255, 80, 70))

            # 右侧信息面板(画布已是 1280x720, 渲染只占左侧 720x540)
            px = RW + 12
            dr2 = dr
            dr2.text((px, 16), f'侧向冲击 {lv_txt}' +
                     (' (超限)' if ov == 1 else (' (极限)' if ov == 2 else '')),
                     font=mid, fill=(240, 240, 245))
            dr2.text((px, 58), f'已受击 {pushes} 次', font=small,
                     fill=(255, 150, 140) if pushes else (150, 160, 170))
            tc_col = (120, 220, 120) if tilt < 45 else ((255, 200, 90) if tilt < 68
                                                        else (255, 90, 80))
            dr2.text((px, 104), f'躯干倾角 {tilt:5.1f}°', font=big, fill=tc_col)
            # 倾角曲线(最近 6 s)
            ch_y, ch_h, ch_w = 190, 160, 480
            dr2.rectangle([px, ch_y, px + ch_w, ch_y + ch_h], fill=(24, 28, 33))
            for gl in (30, 50):
                gy = ch_y + ch_h - int(gl / 75 * ch_h)
                dr2.line([px, gy, px + ch_w, gy], fill=(45, 52, 60))
            gy70 = ch_y + ch_h - int(70. / 75 * ch_h)
            dr2.line([px, gy70, px + ch_w, gy70], fill=(200, 60, 55))
            dr2.text((px + ch_w - 60, gy70 - 24), '70° 失败线', font=tiny,
                     fill=(230, 100, 90))
            if len(tilt_hist) > 1:
                pts = [(px + int(i / (150 - 1) * ch_w),
                        ch_y + ch_h - min(1., t / 75.) * ch_h)
                       for i, t in enumerate(tilt_hist)]
                dr2.line(pts, fill=(110, 190, 250), width=2)
            dr2.text((px, ch_y + ch_h + 12), f'质心-支撑脚 {cd:5.1f} mm (线 40)',
                     font=small,
                     fill=(120, 220, 120) if cd < 40 else (255, 160, 80))
            dr2.text((px, ch_y + ch_h + 46), f'摆动脚高 {sz:6.1f} mm (目标 50)',
                     font=small,
                     fill=(120, 220, 120) if 10 < sz < 90 else (255, 160, 80))
            dr2.text((px, ch_y + ch_h + 86),
                     ('t = %5.1f s' % (step * .04)) +
                     (f'   于 {fallen_at:.1f}s 摔倒' if fallen_at is not None else ''),
                     font=small, fill=(170, 180, 190))
            dr2.text((px, H - 40), '前伸单脚站立 · 0.4 边界级模型 · 右脚支撑',
                     font=tiny, fill=(120, 130, 140))
            all_frames.append(np.asarray(img))
        env.close()
        print(f'  段 {lv_txt}: 受击 {pushes} 次, '
              f'{"于 %.1fs 摔倒" % fallen_at if fallen_at is not None else "未摔倒"}',
              flush=True)
    renderer.close()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out, fps=25, codec='libx264', quality=7)
    for f in all_frames:
        writer.append_data(f)
    writer.close()
    print(f'[write] {out}  {len(all_frames)}帧  {len(all_frames)/25:.0f}s')


if __name__ == '__main__':
    main()
