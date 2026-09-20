"""Flamingo 单脚站立 · 中文网页演示台(viser)。

为什么不用 MuJoCo 原生窗口:它的菜单/提示文字编译在 C++ 二进制里,无法汉化;
viser 是网页界面,中文没有限制,而且能把按钮/滑块/读数都做成中文。

功能:
  - 画面上鼠标拖动 = 转视角(viser 原生),也可用左侧滑块精确调相机
  - 「重置到单脚站姿态」按钮 = 重新采样出生扰动;**重置后自动暂停**,
    让你看清出生态,按「继续」才让策略接管(修掉了原生窗口"只闪一下"的问题)
  - 「按摆动腿切换」= 左脚 / 右脚支撑互换(镜像出生姿态)
  - 实时中文读数:摆动脚高度 / 质心-支撑脚距离 / 躯干倾角 / 头尾偏航俯仰
启动后浏览器打开 http://127.0.0.1:8080
"""
import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import mujoco
import onnxruntime as ort
import viser

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from flamingo_cfg import build_config, FLAMINGO_POSE_BY_NAME, BASE_ROLL, BASE_PITCH  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

DEFAULT_CKPT = 'C:/Users/eugen/AppData/Local/Temp/flamingo_v2.onnx'
RW, RH = 1600, 900   # 16:9 全宽;实测渲染 1280x720 仅 3.5ms,不是瓶颈


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT)
    ap.add_argument('--port', type=int, default=8080)
    a = ap.parse_args()

    task, cfg = build_config(envs=1, seed=47, standing_prob=0.)
    # 物理必须用 GPU:实测单次物理步 CPU 3.70ms vs GPU 0.12ms(31 倍),
    # 策略一步 = 16 次物理步 -> CPU 上限 16.9fps(页面卡顿的根因)、GPU 520fps。
    env = ManagerBasedRlEnv(cfg.env, device='cuda')
    sess = ort.InferenceSession(a.checkpoint, providers=['CPUExecutionProvider'])
    robot = env.scene['robot']

    from terrain_skill_eval import freeze_plant
    plant = freeze_plant('flat') / 'nominal.mjb'
    model = mujoco.MjModel.from_binary_path(
        'nominal.mjb', assets={'nominal.mjb': plant.read_bytes()})
    model.vis.global_.offwidth, model.vis.global_.offheight = RW, RH
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=RH, width=RW)

    names = [n.split('/')[-1] for n in robot.joint_names]
    jadr = {k: 7 + robot.find_joints([k])[0][0]
            for k in ('head_yaw', 'tail_yaw', 'tail_pitch')}
    sites = [n.split('/')[-1] for n in robot.site_names]
    sw, st = sites.index('left_foot'), sites.index('right_foot')
    root_bid = int(robot.indexing.root_body_id)

    # ── 网页界面(全中文) ─────────────────────────────────────────────────
    server = viser.ViserServer(port=a.port, label='MicroDinosaur 单脚站立演示台')
    state = dict(paused=True, reset=False, stopped=False)

    with server.gui.add_folder('操作', expand_by_default=True):
        b_reset = server.gui.add_button('重置到单脚站姿态(并暂停)')
        b_play = server.gui.add_button('继续 / 暂停')
        b_noise = server.gui.add_slider('出生扰动: 关节噪声 (rad)', 0., .12, .005, .05)

    @b_reset.on_click
    def _(_):
        state['reset'] = True

    @b_play.on_click
    def _(_):
        state['paused'] = not state['paused']

    with server.gui.add_folder('相机', expand_by_default=True):
        s_az = server.gui.add_slider('方位角 °', -180., 180., 1., 135.)
        s_el = server.gui.add_slider('俯仰角 °', -89., 20., 1., -10.)
        s_di = server.gui.add_slider('距离 m', .25, 1.5, .01, .52)
        s_th = server.gui.add_slider('绕躯干环视 °', -90., 90., 1., 0.)

    readout = server.gui.add_markdown('_等待启动…_')
    # **把机器人放进网页的 3D 场景**(而不是只推一张本地渲染的小图):
    # mjviser 把 mujoco 的 mesh 一次性加载给浏览器,每帧只更新 body 位姿,
    # 渲染由浏览器 GPU(WebGL)完成 —— 背景空间就是真 3D,可以任意转。
    import mjviser
    scene = mjviser.ViserMujocoScene(server, model, num_envs=1)
    img = None   # 保留小窗口作为"跟拍特写"开关
    with server.gui.add_folder('视角', expand_by_default=True):
        cb_follow = server.gui.add_checkbox('跟拍特写窗口(小图)', True)

    def render():
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = float(s_az.value)
        cam.elevation = float(s_el.value)
        cam.distance = float(s_di.value)
        yaw = math.radians(float(s_th.value))
        # 环视:在躯干上方绕一个半径 0 的点转(纯旋转相机方位角就够,这里做水平偏移)
        cam.lookat[:] = [float(data.qpos[0]) + .0 * math.cos(yaw),
                         float(data.qpos[1]) + .0 * math.sin(yaw), .11]
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = 0
        renderer.update_scene(data, camera=cam, scene_option=opt)
        return renderer.render()

    def do_reset():
        obs, _ = env.reset()
        state['obs'] = obs['actor'] if isinstance(obs, dict) else obs
        d = env.sim.data
        data.qpos[:] = d.qpos[0].cpu().numpy()
        data.qvel[:] = d.qvel[0].cpu().numpy()
        mujoco.mj_forward(model, data)
        state['paused'] = True

    do_reset()
    frames = [0]
    t0 = time.time()
    while True:
        if state['reset']:
            state['reset'] = False
            do_reset()
            print('[web] 已重置到单脚站出生态并暂停', flush=True)
        if not state['paused']:
            obs = env.__dict__.get('_obs_cache')
            o = env.obs_buf if hasattr(env, 'obs_buf') else None
            # 用 env 的观测缓存(每次 step 后更新)
            o = state.get('obs')
            if o is not None:
                onx = torch.as_tensor(o).reshape(1, -1).numpy().astype(np.float32) \
                    if not torch.is_tensor(o) else o.reshape(1, -1).numpy().astype(np.float32)
                act = sess.run(None, {'obs': onx})[0]
                obs, *_ = env.step(torch.as_tensor(act, device=env.device))
                state['obs'] = (obs['actor'] if isinstance(obs, dict) else obs)
            d = env.sim.data
            data.qpos[:] = d.qpos[0].cpu().numpy()
            data.qvel[:] = d.qvel[0].cpu().numpy()
            mujoco.mj_forward(model, data)
        d = env.sim.data
        sz = float(robot.data.site_pos_w[0, sw, 2].cpu()) * 1000
        com = np.asarray(d.subtree_com[0, root_bid, :2].cpu())
        foot = robot.data.site_pos_w[0, st, :2].cpu().numpy()
        cd = float(np.linalg.norm(com - foot)) * 1000
        gz = robot.data.projected_gravity_b[0].cpu().numpy()
        tilt = math.degrees(math.acos(min(1., abs(gz[2]))))
        hy = math.degrees(float(d.qpos[0, jadr['head_yaw']].cpu()))
        ty = math.degrees(float(d.qpos[0, jadr['tail_yaw']].cpu()))
        tp = math.degrees(float(d.qpos[0, jadr['tail_pitch']].cpu()))
        readout.content = (
            f'### 实时读数\n'
            f'| 项 | 值 |\n|---|---|\n'
            f'| 摆动脚(左)离地 | **{sz:.1f} mm** |\n'
            f'| 质心-支撑脚水平距 | **{cd:.1f} mm** |\n'
            f'| 躯干倾角 | {tilt:.1f}° (姿态 24.4°) |\n'
            f'| 头偏航(配重) | {hy:+.1f}° (姿态 −86°) |\n'
            f'| 尾偏航 / 尾俯仰 | {ty:+.1f}° / {tp:+.1f}° |\n'
            f'| 状态 | {"暂停(显示出生态)" if state["paused"] else "运行中"} |'
        )
        try:
            scene.update_from_mjdata(data)      # 只更新位姿,渲染在浏览器端
        except Exception as exc:
            if not state.get('scene_err'):
                state['scene_err'] = True
                print('[web] 3D 场景更新异常:', repr(exc)[:200], flush=True)
        if img is not None and cb_follow.value:
            try:
                img.image = render()
            except Exception:
                pass
        time.sleep(.008)


if __name__ == '__main__':
    main()
