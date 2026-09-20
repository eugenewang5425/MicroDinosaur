"""20 m 直道障碍赛:平地 → 10 级台阶 → 凹凸路 → 迷宫 → 终点。

植物 = obstacle_plant(preferred 同源 spec + 35 个静态障碍 geom,机器人侧字段
逐字节一致,见 build_obstacle_plant.py)。引导 = 沿穿行路径折线的纯追踪
(迷宫段路径走缺口中心);速度 0.45 m/s 给障碍留控制裕度。
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
sys.path.insert(0, str(ROOT))
from evaluate_run_jump import MotionExperiment  # noqa: E402
from imu_owned_head import configure_owned  # noqa: E402
from hardware_sim import HardwareCase  # noqa: E402
from build_obstacle_plant import WAYPOINTS, OBSTACLES  # noqa: E402

OUT = ROOT / '20260914_contact_motion'
PLANT = OUT / 'obstacle_plant'
POLICY = OUT / 'run_16700.onnx'
V = 0.45
LOOKAHEAD = 0.9
SEED = 941
SECONDS = 120.0
FINISH_X = 19.6
W, H = 760, 760
FONT = 'C:/Windows/Fonts/msyh.ttc'
SECTIONS = [('平地', 0., 4.), ('台阶', 4., 6.4), ('平地', 6.4, 8.),
            ('凹凸路', 8., 12.), ('平地', 12., 13.5), ('迷宫', 13.5, 17.5),
            ('终点直道', 17.5, 20.)]


def path_point(s):
    """折线上弧长 s 处的点。"""
    seg = np.asarray(WAYPOINTS, float)
    d = np.hypot(*np.diff(seg, axis=0).T)
    cum = np.r_[0., np.cumsum(d)]
    s = float(np.clip(s, 0., cum[-1]))
    k = int(np.searchsorted(cum, s, 'right') - 1)
    k = min(k, len(d) - 1)
    f = (s - cum[k]) / max(d[k], 1e-9)
    a, b = seg[k], seg[k + 1]
    return a + f * (b - a)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    case = HardwareCase(motor_curve=True, voltage=12., physics_dt=.00125, command_ms=10,
                        foot_torsional_friction_m=.01)
    e = configure_owned(MotionExperiment(POLICY, 'run', V, case, plant=PLANT,
                                         operating_envelope=True), (1,))
    wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
    seg = np.asarray(WAYPOINTS, float)
    seglen = np.hypot(*np.diff(seg, axis=0).T)
    cum = np.r_[0., np.cumsum(seglen)]
    total_path = cum[-1]

    def course_command(t, scenario):
        d = e.sim.data
        q = d.qpos
        pos = np.array([q[0], q[1]])
        x_now = pos[0]
        # 分段限速/前瞻。教训(实测两次):台阶**不能减速** —— 慢速没动量,0.30 m/s
        # 在第 3 级摔倒,0.45 反而两次通过;凹凸路同理。只有迷宫 S 弯需要慢+短前瞻。
        if 13.3 <= x_now < 17.6:      # 迷宫
            v, look = .26, .45
        elif 3.8 <= x_now < 6.6:      # 台阶:保持速度,动量是爬阶的一部分
            v, look = .45, .70
        elif 7.8 <= x_now < 12.2:     # 凹凸路
            v, look = .45, .70
        else:                          # 平地
            v, look = .45, .90
        # 按弧长投影:折线 x 单调不减,直接用 x 找所在线段(避免卡在弯外)
        k = int(np.clip(np.searchsorted(seg[:, 0], pos[0], 'right') - 1, 0, len(seglen) - 1))
        f = float(np.clip((pos[0] - seg[k][0]) / max(seg[k + 1][0] - seg[k][0], 1e-9), 0., 1.))
        s_proj = cum[k] + f * seglen[k]
        aim = path_point(s_proj + look)
        psi = math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))
        phi_d = math.atan2(aim[1] - pos[1], aim[0] - pos[0])
        alpha = wrap(phi_d - psi)
        dist = float(np.hypot(aim[0] - pos[0], aim[1] - pos[1]))
        u = np.zeros(18)
        u[0] = v
        u[2] = float(np.clip((2 * v / max(dist, .3)) * math.sin(alpha), -.9, .9))
        return u

    e.user_command = course_command
    metrics, trace = e.run('course', 'imu', SEED, SECONDS, True)

    frames = np.asarray(e.qpos_frames)
    dt_f = 32 * case.physics_dt
    xy = frames[:, :2]
    rows = np.asarray(e.rows)
    np.savez_compressed(OUT / 'course_frames.npz', qpos=frames, rows=rows.astype(float))
    tilt_max = float(rows[:, 8].max()) if len(rows) else -1.
    x = xy[:, 0]
    finish_i = int(np.where(x >= FINISH_X)[0][0]) if (x >= FINISH_X).any() else -1
    finish_t = finish_i * dt_f if finish_i >= 0 else float('nan')
    # 分段用时
    section_times = {}
    for name, x0, x1 in SECTIONS:
        hit = np.where(x >= x1 - .05)[0]
        section_times[name] = float(hit[0] * dt_f) if len(hit) else float('nan')
    fell = int(np.sum((rows[:, 8] > 45.)[:, None] if rows.ndim == 1 else rows[:, 8] > 45.)) \
        if len(rows) else 0
    fell_t = float(np.where(rows[:, 8] > 45.)[0][0] * case.physics_dt) \
        if len(rows) and (rows[:, 8] > 45.).any() else float('nan')
    stats = dict(policy='run_16700.onnx', seed=SEED, v_cmd=V, lookahead_m=LOOKAHEAD,
                 course='直道 20m: 台阶10级+凹凸路+迷宫',
                 finished=bool(finish_i >= 0), finish_time_s=float(finish_t),
                 max_x_m=float(x.max()), tilt_max_deg=tilt_max,
                 fell_samples_over45deg=int(fell), first_fall_s=fell_t,
                 section_finish_times_s=section_times,
                 heading_error_rms_deg=metrics.get('heading_error_rms_deg'))
    (OUT / 'course_stats.json').write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)

    # ---------------- 渲染 ----------------
    renderer = mujoco.Renderer(e.sim.model, height=H, width=W)
    data = mujoco.MjData(e.sim.model)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 1.05, 135., -30
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    mid_f = ImageFont.truetype(FONT, 26)
    mid = ImageFont.truetype(FONT, 21)
    small = ImageFont.truetype(FONT, 17)

    # 小地图:整条赛道俯视,20m → 660px,垂直居中;y 上为正
    sc = 660. / 20.
    ox, oy = 50., H / 2.

    def mp(p):
        return (ox + p[0] * sc, oy - p[1] * sc * 3.2)   # y 放大 3.2 倍便于看清迷宫

    writer = imageio.get_writer(str(OUT / 'course_video.mp4'), fps=25,
                                codec='libx264', quality=7)
    n = len(frames)
    for i in range(n):
        data.qpos[:] = frames[i]
        mujoco.mj_forward(e.sim.model, data)
        cam.lookat[:] = [xy[i, 0], xy[i, 1], frames[i][2] + .02]  # 跟随根高
        if 3.8 <= xy[i, 0] < 6.6:
            # 台阶段:正侧面机位,从旁边看爬阶(视线越过 12cm 阶体,不遮挡)
            cam.azimuth, cam.elevation, cam.distance = 90., -8., 1.0
        else:
            cam.azimuth, cam.elevation, cam.distance = 135., -14., .8
        renderer.update_scene(data, camera=cam, scene_option=opt)
        left = Image.fromarray(renderer.render())
        dl = ImageDraw.Draw(left)
        dl.rectangle((0, 0, W, 40), fill=(20, 24, 28))
        dl.text((12, 7), 'MicroDinosaur 直道障碍赛 20 m: 台阶→凹凸路→迷宫',
                font=mid_f, fill='white')

        img = Image.new('RGB', (2 * W, H), (14, 17, 20))
        img.paste(left, (0, 0))
        dr = ImageDraw.Draw(img)
        # 赛道条带底色 + 分段
        y0, y1 = oy - 150, oy + 150
        for name, x0, x1 in SECTIONS:
            color = {'台阶': (60, 90, 130), '凹凸路': (95, 80, 55),
                     '迷宫': (80, 60, 100)}.get(name, (40, 46, 52))
            dr.rectangle([mp([x0, 1])[0], y0, mp([x1, 0])[0], y1], fill=color)
        # 障碍几何(俯视)
        for o in OBSTACLES:
            color = (150, 190, 255) if o['label'].startswith(('stair', 'bump')) \
                else (220, 140, 230)
            dr.rectangle([mp([o['x'] - o['hx'], o['y'] + o['hy']])[0],
                          mp([0, o['y'] + o['hy']])[1],
                          mp([o['x'] + o['hx'], 0])[0],
                          mp([0, o['y'] - o['hy']])[1]], fill=color)
        # 路径与轨迹
        dr.line([mp(p) for p in np.asarray(WAYPOINTS)], fill=(140, 200, 140), width=3)
        pts = [mp(p) for p in xy[:max(i, 1)]]
        if len(pts) > 1:
            dr.line(pts, fill=(90, 170, 255), width=3)
        q = frames[i]
        yaw = math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))
        px, py = mp(xy[i])
        dr.ellipse((px - 6, py - 6, px + 6, py + 6), fill=(255, 90, 90))
        dr.line((px, py, px + 26 * math.cos(yaw), py - 26 * math.sin(yaw)),
                fill=(255, 90, 90), width=4)
        # 分段标签
        for name, x0, x1 in SECTIONS:
            if name == '平地' and x1 - x0 < 1.5:
                continue
            dr.text((mp([x0, 0])[0] + 4, y1 + 6), name, font=small, fill=(170, 180, 190))
        dr.text((mp([19.7, 0])[0] - 20, y0 - 26), '终点', font=small, fill=(255, 210, 70))
        # 统计
        dr.rectangle((W, 0, 2 * W, 44), fill=(20, 24, 28))
        sec_now = next((nm for nm, x0, x1 in SECTIONS if x0 <= x[i] < x1), '—')
        dr.text((W + 14, 8), f'当前路段 {sec_now}   x = {x[i]:5.2f} / 20 m',
                font=mid, fill='white')
        dr.rectangle((W, H - 92, 2 * W, H), fill=(20, 24, 28))
        dr.text((W + 14, H - 84), f'指令: 前速 {V:.2f} m/s   沿折线纯追踪(前瞻 {LOOKAHEAD} m)',
                font=small, fill=(190, 200, 210))
        dr.text((W + 14, H - 56), f'最大倾角 {tilt_max:.1f}°   最大前进 x {x[:i + 1].max():.2f} m',
                font=small, fill=(190, 200, 210))
        fin = f'{finish_t:.1f} s' if finish_i >= 0 and i >= finish_i else '未完赛'
        dr.text((W + 14, H - 30), f'完赛: {fin}', font=small, fill=(255, 210, 70))
        writer.append_data(np.asarray(img))
    writer.close()
    renderer.close()
    print(f'[write] {OUT / "course_video.mp4"}  {n}帧', flush=True)


if __name__ == '__main__':
    main()
