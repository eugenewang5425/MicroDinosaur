"""40 m 标准环形跑道绕圈 × 三种地面材质,标准 IMU 航向栈。

跑道定义(用户令:"40米长的标准环形跑道绕一圈"):
- 圆周长 40 m → 半径 R=40/2π=6.3662 m;圆心 (0,R),机器人从圆环正底部 (0,0) 出发。
- 纯追踪引导:瞄准圆周上前方 LOOKAHEAD 米的点,ω=(2v/L)·sin(α);只改偏航率参考,
  标准 IMU 栈与植物物理逐字节不动。

三种材质(用户令:"三种材质,粗糙度和平整度要有大差异,三种外观"):
- 物理侧 = 摩擦三参数 (slide, torsion, roll)。MuJoCo 接触取两 geom 的逐元素 max,
  所以足底和地面**都**设成目标值。标称口径 (1.0, 0.01, 1e-6),训练随机化 slide 0.7-1.3。
- 外观侧 = 程序纹理直接写进 model.tex_rgb(地面材质自带 300×300 纹理),
  mat_texrepeat 控制颗粒尺度 —— 三种材质三种外观,不是换滤镜。
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
from evaluate_run_jump import MotionExperiment  # noqa: E402
from imu_owned_head import configure_owned  # noqa: E402
from hardware_sim import HardwareCase  # noqa: E402

OUT = ROOT / '20260914_contact_motion'
POLICY = OUT / 'run_16700.onnx'
CIRC = 40.0
R = CIRC / (2 * math.pi)
CENTER = np.array([0.0, R])
V = 0.6
OMEGA = V / R
SEED = 941
SECONDS = 130.0
PURSUIT = True
LOOKAHEAD = 1.2
W, H = 760, 760          # 每栏尺寸
FONT = 'C:/Windows/Fonts/msyh.ttc'


# ---------------------------------------------------------------------------
# 材质定义
# ---------------------------------------------------------------------------
def _paint_tile(tex):
    """光滑瓷砖:2×2 大块亮面、浅缝、零颗粒 —— 视觉上最平整。"""
    t = tex.reshape(300, 300, 3)
    t[:] = (205, 210, 218)
    for by in (0, 150):
        for bx in (0, 150):
            shade = ((bx + by) // 150) * np.array([3, 1, -3])
            t[by:by + 148, bx:bx + 148] = np.clip(
                t[by:by + 148, bx:bx + 148] + shade, 0, 255)
    for k in range(0, 300, 150):
        t[k:k + 2, :] = (150, 158, 170)
        t[:, k:k + 2] = (150, 158, 170)


def _paint_rubber(tex):
    """标准橡胶跑道:赭红底 + 细密棋盘格(跑道面胶的标准观感)。"""
    t = tex.reshape(300, 300, 3)
    yy, xx = np.mgrid[0:300, 0:300]
    chess = (((xx // 15) + (yy // 15)) % 2).astype(float) * 14 - 7
    t[..., 0] = np.clip(155 + chess, 0, 255)
    t[..., 1] = np.clip(78 + chess * .5, 0, 255)
    t[..., 2] = np.clip(48 + chess * .32, 0, 255)


def _paint_grit(tex):
    """粗糙砂面/沥青:深灰底 + 逐像素噪声 + 亮砂粒斑 —— 视觉上最糙。"""
    rng = np.random.default_rng(7)
    t = tex.reshape(300, 300, 3)
    base = np.full((300, 300, 3), (58, 60, 64), np.int16)
    base += rng.normal(0, 14, base.shape).astype(np.int16)
    for _ in range(320):
        y, x = int(rng.integers(0, 296)), int(rng.integers(0, 296))
        base[y:y + 3, x:x + 3] = (122, 122, 124)
    np.clip(base, 0, 255, out=base)
    t[:] = base


MATERIALS = {
    'tile': dict(label='光滑瓷砖', slide=.45, torsion=.004, roll=2e-5,
                 painter=_paint_tile, texrepeat=(2., 2.),
                 note='μ=0.45,低于训练随机化下限 0.7(挑战滑移)'),
    'rubber': dict(label='橡胶跑道', slide=1.0, torsion=.01, roll=1e-6,
                   painter=_paint_rubber, texrepeat=(6., 6.),
                   note='标称口径,训练分布中值'),
    'grit': dict(label='粗糙砂面', slide=1.6, torsion=.03, roll=5e-5,
                 painter=_paint_grit, texrepeat=(10., 10.),
                 note='μ=1.6,高于训练上限 1.3(高抓地)'),
}


def apply_material(sim, mat):
    """把材质写进编译好的 mjModel:足底+地面摩擦、地面纹理、texrepeat。

    纹理数据在 mujoco 3.10 里是 `m.tex_data`(扁平 uint8,h·w·nchan),不是旧名的
    tex_rgb;nchannel=3(RGB),按 300×300×3 写入。"""
    m = sim.model
    tid = m.geom('terrain').id
    for gid in tuple(sim.foot_geoms) + (tid,):
        m.geom_friction[gid] = (mat['slide'], mat['torsion'], mat['roll'])
    if m.ntex > 0:
        h, w, nc = int(m.tex_height[0]), int(m.tex_width[0]), int(m.tex_nchannel[0])
        canvas = np.zeros((h, w, 3), np.uint8)
        mat['painter'](canvas)
        flat = canvas.reshape(-1) if nc == 3 else np.repeat(canvas, nc // 3, axis=2).reshape(-1)
        m.tex_data[:] = flat
    mid = int(m.geom_matid[tid])
    if mid >= 0:
        m.mat_texrepeat[mid][:2] = mat['texrepeat']


def run_one(key):
    mat = MATERIALS[key]
    sys.stdout.reconfigure(encoding='utf-8')
    selected = json.loads((OUT / 'selected_contact.json').read_text())
    plant = OUT / selected.get('plant_directory', 'normal_contact_plant')
    case = HardwareCase(motor_curve=True, voltage=12., physics_dt=.00125, command_ms=10,
                        foot_torsional_friction_m=selected['friction'][1])
    e = configure_owned(MotionExperiment(POLICY, 'run', V, case, plant=plant,
                                         operating_envelope=True), (1,))
    apply_material(e.sim, mat)
    print(f"[lap:{key}] {mat['label']}  μ=({mat['slide']},{mat['torsion']},{mat['roll']})  "
          f"{mat['note']}", flush=True)
    user = np.zeros(18)
    user[0], user[2] = V, OMEGA

    wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi

    def lap_command(t, s):
        """纯追踪引导:瞄准圆周上前方 LOOKAHEAD 米的点,ω=(2v/L)·sin(α)。

        依据(2026-09-17 实测):恒定 ω=v/R 的开环引导径向误差均值 1.47m ——
        航向闭环 RMS 9.6°,恒速转不够贴线。引导只改 user[2](偏航率参考)。"""
        d = e.sim.data
        q = d.qpos
        pos = np.array([q[0], q[1]])
        psi = math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))
        th_i = math.atan2(pos[1] - CENTER[1], pos[0] - CENTER[0])
        aim = CENTER + R * np.array([math.cos(th_i + LA), math.sin(th_i + LA)])
        phi_d = math.atan2(aim[1] - pos[1], aim[0] - pos[0])
        alpha = wrap(phi_d - psi)
        dist = float(np.hypot(aim[0] - pos[0], aim[1] - pos[1]))
        u = user.copy()
        u[2] = float(np.clip((2 * V / max(dist, .3)) * math.sin(alpha), -.7, .7))
        return u

    LA = LOOKAHEAD / R
    e.user_command = (lambda t, s: user) if not PURSUIT else lap_command
    metrics, trace = e.run('lap', 'imu', SEED, SECONDS, True)

    frames = np.asarray(e.qpos_frames)
    dt_f = 32 * case.physics_dt          # 0.04 s → 25 fps
    xy = frames[:, :2]
    rel = xy - CENTER
    th = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    th_acc = th - th[0]
    path = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    done = np.where(th_acc >= 2 * np.pi)[0]
    lap_i = int(done[0]) if len(done) else -1
    lap_t = lap_i * dt_f if lap_i >= 0 else float('nan')
    radial = np.abs(np.hypot(rel[:, 0], rel[:, 1]) - R)
    rows = np.asarray(e.rows)
    tilt_max = float(rows[:, 8].max()) if len(rows) else -1.
    stats = dict(material=key, label=mat['label'],
                 friction=[mat['slide'], mat['torsion'], mat['roll']],
                 policy='run_16700.onnx', seed=SEED, v_cmd=V, lookahead_m=LOOKAHEAD,
                 track_radius_m=R, frames=len(frames),
                 lap_completed=bool(lap_i >= 0), lap_time_s=float(lap_t),
                 lap_path_m=float(path[lap_i]) if lap_i >= 0 else float(path[-1]),
                 total_path_m=float(path[-1]),
                 radial_error_mean_m=float(radial.mean()),
                 radial_error_max_m=float(radial.max()),
                 tilt_max_deg=tilt_max,
                 heading_error_rms_deg=metrics.get('heading_error_rms_deg'))
    (OUT / f'lap_stats_{key}.json').write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)

    # ---------------- 渲染 ----------------
    renderer = mujoco.Renderer(e.sim.model, height=H, width=W)
    data = mujoco.MjData(e.sim.model)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = .8, 135., -14
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    mid_f = ImageFont.truetype(FONT, 26)
    mid = ImageFont.truetype(FONT, 21)
    small = ImageFont.truetype(FONT, 17)

    sc = (W - 128) / (2 * (R + 1.6))

    def mp(p):
        # 右栏小地图:x 加一栏偏移 W;世界 y 向上 → 像素 y 向下
        return (W + W / 2 + p[0] * sc, H / 2 - (p[1] - R) * sc)

    writer = imageio.get_writer(str(OUT / f'lap_{key}.mp4'), fps=25,
                                codec='libx264', quality=7)
    n = len(frames)
    for i in range(n):
        data.qpos[:] = frames[i]
        mujoco.mj_forward(e.sim.model, data)
        cam.lookat[:] = [xy[i, 0], xy[i, 1], .10]
        renderer.update_scene(data, camera=cam, scene_option=opt)
        left = Image.fromarray(renderer.render())
        dl = ImageDraw.Draw(left)
        dl.rectangle((0, 0, W, 40), fill=(20, 24, 28))
        dl.text((12, 7), f'MicroDinosaur 40 m 环形跑道 · {mat["label"]}({key})',
                font=mid_f, fill='white')

        img = Image.new('RGB', (2 * W, H), (14, 17, 20))
        img.paste(left, (0, 0))
        dr = ImageDraw.Draw(img)
        # 跑道双圈线 + 中线
        dr.ellipse([mp([-R - .45, 0])[0], mp([0, 2 * R + .45])[1],
                    mp([R + .45, 0])[0], mp([0, -.45])[1]],
                   outline=(70, 78, 88), width=2)
        dr.ellipse([mp([-R + .45, 0])[0], mp([0, 2 * R - .45])[1],
                    mp([R - .45, 0])[0], mp([0, .45])[1]],
                   outline=(70, 78, 88), width=2)
        dr.ellipse([mp([-R, 0])[0], mp([0, 2 * R])[1], mp([R, 0])[0], mp([0, 0])[1]],
                   outline=(120, 130, 142), width=2)
        a, b = mp([-.35, 0]), mp([.35, 0])
        dr.line([a, b], fill=(255, 210, 70), width=5)
        dr.text((a[0] - 34, a[1] + 8), '起点/终点', font=small, fill=(255, 210, 70))
        pts = [mp(p) for p in xy[:max(i, 1)]]
        if len(pts) > 1:
            dr.line(pts, fill=(90, 170, 255), width=3)
        q = frames[i]
        yaw = math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))
        px, py = mp(xy[i])
        dr.ellipse((px - 7, py - 7, px + 7, py + 7), fill=(255, 90, 90))
        dr.line((px, py, px + 34 * math.cos(yaw), py - 34 * math.sin(yaw)),
                fill=(255, 90, 90), width=5)
        th_i = th[0] + (OMEGA * i * dt_f if i * dt_f <= CIRC / V else 2 * np.pi)
        rp = CENTER + R * np.array([math.cos(th_i), math.sin(th_i)])
        rx, ry = mp(rp)
        dr.ellipse((rx - 4, ry - 4, rx + 4, ry + 4), outline=(140, 200, 140), width=2)
        dr.rectangle((W, 0, 2 * W, 44), fill=(20, 24, 28))
        prog = min(1.0, th_acc[i] / (2 * np.pi))
        dr.text((W + 14, 8), f'圈进度 {prog * 100:5.1f}%   已跑 {path[i]:5.1f} m / 40 m',
                font=mid, fill='white')
        dr.rectangle((W, H - 120, 2 * W, H), fill=(20, 24, 28))
        dr.text((W + 14, H - 112), f'材质 {mat["label"]}:  摩擦 μ={mat["slide"]}'
                                   f'  扭转 {mat["torsion"]}  {mat["note"]}',
                font=small, fill=(255, 210, 70))
        dr.text((W + 14, H - 84), f'指令: 前速 {V:.2f} m/s   纯追踪前瞻 {LOOKAHEAD:.1f} m',
                font=small, fill=(190, 200, 210))
        dr.text((W + 14, H - 56), f'径向误差 均值 {radial[:i + 1].mean():.3f} m  '
                                  f'最大 {radial[:i + 1].max():.3f} m',
                font=small, fill=(190, 200, 210))
        lap_text = (f'{lap_t:.1f} s' if lap_i >= 0 and i >= lap_i
                    else (f'预计 {CIRC / V:.0f} s' if lap_i < 0 else '—'))
        dr.text((W + 14, H - 30), f'圈时 {lap_text}(理论 {CIRC / V:.0f} s)',
                font=small, fill=(255, 210, 70))
        writer.append_data(np.asarray(img))
    writer.close()
    renderer.close()
    print(f'[write] {OUT / f"lap_{key}.mp4"}  {n}帧', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--material', choices=[*MATERIALS, 'all'], default='all')
    a = ap.parse_args()
    keys = list(MATERIALS) if a.material == 'all' else [a.material]
    for k in keys:
        run_one(k)


if __name__ == '__main__':
    main()
