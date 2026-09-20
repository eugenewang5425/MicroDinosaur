"""直道障碍赛的植物:preferred 同源 spec + 障碍 geom,编译成独立 mjb。

为什么走 mjlab Scene 而不是改 mjb:mjb 是编译产物,不能加 geom;
Scene(cfg.env.scene) 是研究线植物的**同一条**构建路径(surface_contacts 用
selected['friction'],robot_spec 同源),编译后与 preferred_plant 逐字段校验
(质量/惯量/关节/执行器不变,只有 ngeom 因障碍增加),物理口径不漂。

赛道(用户令:"直道,中间放 10 级台阶、一片坑坑洼洼的路、一段迷宫"),沿 +x:
  0-4m   平地热身
  4.0-6.4m  台阶:上 10 级(每级高 1.2cm/深 9cm,共 12cm)+ 0.6m 平台 + 下 10 级
  6.4-8.0m  平地恢复
  8.0-12.0m 凹凸路:10 块随机高低凸块(0.8-1.4cm 高,密集排布)
  12.0-13.5m 平地
  13.5-17.5m 迷宫:两侧导轨 + 4 道梳齿墙,左右交替留 30cm 缺口,S 形穿行
  17.5-20m  平地终点
"""
from pathlib import Path
import json
import shutil
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / '20260914_contact_motion'
RNG = np.random.default_rng(11)

# 半尺寸 (hx,hy,hz) @ 中心 (x,y,z);单位 m。材质统一 = 橡胶标称摩擦。
OBSTACLES = []


def box(x, y, z, hx, hy, hz, label):
    OBSTACLES.append(dict(x=x, y=y, z=z, hx=hx, hy=hy, hz=hz, label=label))


# --- 台阶(上 10 + 平台 + 下 10) ---
# RISE 可用环境变量调:实测 1.2cm 平地策略爬不过(第 3 级摔倒),用来扫描阈值。
import os
RISE = float(os.environ.get('OBST_RISE', .012))
TREAD, STAIR_W = .09, .70
for k in range(1, 11):
    box(4.0 + (k - .5) * TREAD, 0, k * RISE / 2, TREAD / 2, STAIR_W, k * RISE / 2,
        f'stair_up_{k:02d}')
box(5.2, 0, 10 * RISE / 2, .30, STAIR_W, 10 * RISE / 2, 'stair_plateau')
for j in range(1, 11):
    box(5.5 + (j - .5) * TREAD, 0, (11 - j) * RISE / 2, TREAD / 2, STAIR_W,
        (11 - j) * RISE / 2, f'stair_down_{j:02d}')

# --- 凹凸路 8-12m:凸块不可穿越即止(足长 7cm,凸块高可调) ---
x = 8.0
bi = 0
BUMP_MAX = float(os.environ.get('OBST_BUMP_MAX', .014))
while x < 11.7:
    depth = float(RNG.uniform(.15, .30))
    h = float(RNG.uniform(BUMP_MAX * .6, BUMP_MAX))
    y = float(RNG.uniform(-.35, .35))
    box(x + depth / 2, y, h / 2, depth / 2, .30, h / 2, f'bump_{bi:02d}')
    x += depth + float(RNG.uniform(.15, .35))
    bi += 1

# --- 迷宫 13.5-17.5m:两侧导轨 + 4 道梳齿墙(缺口 30cm 左右交替) ---
box(15.5, -.47, .05, 2.0, .02, .05, 'maze_rail_left')
box(15.5, +.47, .05, 2.0, .02, .05, 'maze_rail_right')
MAZE_WALLS = [(13.9, -.15), (14.9, +.15), (15.9, -.15), (16.9, +.15)]
for i, (wx, wy) in enumerate(MAZE_WALLS):
    box(wx, wy, .05, .02, .30, .05, f'maze_wall_{i + 1}')

# 穿行路径(引导用):缺口中心 ±0.30,前后加过渡点让 S 弯更缓(实测第一版
# 直角折线在第二道墙甩倒)
WAYPOINTS = [(0., 0.), (13.3, 0.),
             (13.55, +.30), (14.15, +.30), (14.4, 0.),
             (14.65, -.30), (15.15, -.30), (15.4, 0.),
             (15.65, +.30), (16.15, +.30), (16.4, 0.),
             (16.65, -.30), (17.15, -.30), (17.5, 0.),
             (18.5, 0.), (20., 0.)]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    from contact_motion_cfg import build_config
    from mjlab.scene import Scene

    old = mujoco.MjModel.from_binary_path(
        'x', assets={'x': (OUT / 'preferred_plant' / 'nominal.mjb').read_bytes()})
    task, cfg = build_config('run', 1, 914)
    scene = Scene(cfg.env.scene, device='cpu')
    for key in ('timestep', 'integrator', 'solver', 'iterations', 'ls_iterations',
                'cone', 'impratio', 'tolerance', 'ls_tolerance'):
        setattr(scene.spec.option, key, getattr(old.opt, key))
    world = scene.spec.worldbody
    for o in OBSTACLES:
        world.add_geom(name=o['label'], type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=[o['x'], o['y'], o['z']],
                       size=[o['hx'], o['hy'], o['hz']],
                       rgba=[.25, .45, .85, 1.])
        # 摩擦与橡胶标称一致(接触取 max,与地面同口径)
    m = scene.compile()
    for gid in range(m.ngeom):
        name = m.geom(gid).name
        if name.startswith(('stair_', 'bump_', 'maze_')):
            m.geom_friction[gid] = (1.0, .01, 1e-6)

    # --- 与 preferred_plant 逐字段校验:机器人侧必须逐字节相同 ---
    for field in ('body_mass', 'body_inertia', 'body_ipos', 'body_pos', 'body_quat',
                  'jnt_range', 'dof_damping', 'dof_armature', 'dof_frictionloss',
                  'actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange'):
        np.testing.assert_array_equal(getattr(m, field), getattr(old, field)),
        print(f'[invariant] {field} == preferred ✓')
    added = [m.geom(g).name for g in range(old.ngeom, m.ngeom)]
    assert len(added) == len(OBSTACLES), (len(added), len(OBSTACLES))
    feet = [m.geom('robot/' + s + '_foot_collision').id for s in ('left', 'right')]
    for gid in feet:
        assert tuple(m.geom_friction[gid]) == (1., .01, 1e-6)
    print(f'[invariant] 足底摩擦 (1,.01,1e-6) ✓   新增 geom {len(added)} 个 ✓')

    folder = OUT / 'obstacle_plant'
    folder.mkdir(exist_ok=True)
    buffer = np.empty(mujoco.mj_sizeModel(m), np.uint8)
    mujoco.mj_saveModel(m, buffer=buffer)
    (folder / 'nominal.mjb').write_bytes(buffer.tobytes())
    shutil.copy2(OUT / 'preferred_plant' / 'contract.json', folder / 'contract.json')
    (folder / 'course.json').write_text(json.dumps(dict(
        obstacles=OBSTACLES, waypoints=WAYPOINTS,
        stair_rise_m=RISE, bump_max_m=BUMP_MAX,
        n_geom_added=len(added), base='preferred_plant',
        note='机器人侧字段与 preferred_plant 逐字节一致;只加了静态障碍 geom'), indent=2),
        encoding='utf-8')
    print(f'[write] {folder / "nominal.mjb"}  (+course.json)')


if __name__ == '__main__':
    main()
