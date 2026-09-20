"""Flamingo 单脚站立交互演示窗口。

用法: python flamingo_viewer.py [--checkpoint xxx.onnx]
  - 鼠标左键拖动 = 转视角;右键拖动 = 平移;滚轮 = 缩放(mujoco viewer 原生)
  - 按 R = 重置回单脚站出生状态(重采样倾斜/关节噪声)
  - 空格 = 暂停/继续
  - 窗口标题实时显示: 摆动脚高度 / 质心距 / 倾角 / 头尾偏航

实现: mjlab env 负责物理与策略观测(与训练同源), 每步把 qpos 同步到原生
mjModel 供 mujoco.viewer 渲染 —— 两边的物理参数来自同一株植物。
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import mujoco
import mujoco.viewer
import onnxruntime as ort

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from flamingo_cfg import build_config  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

DEFAULT_CKPT = 'C:/Users/eugen/AppData/Local/Temp/flamingo_v2.onnx'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT)
    ap.add_argument('--standing-prob', type=float, default=0.)
    a = ap.parse_args()

    task, cfg = build_config(envs=1, seed=47, standing_prob=a.standing_prob)
    env = ManagerBasedRlEnv(cfg.env, device='cpu')
    sess = ort.InferenceSession(a.checkpoint, providers=['CPUExecutionProvider'])
    robot = env.scene['robot']

    from terrain_skill_eval import freeze_plant
    plant = freeze_plant('flat') / 'nominal.mjb'
    model = mujoco.MjModel.from_binary_path(
        'nominal.mjb', assets={'nominal.mjb': plant.read_bytes()})
    data = mujoco.MjData(model)
    data.qpos[:] = env.sim.data.qpos[0].numpy()
    mujoco.mj_forward(model, data)

    names = [n.split('/')[-1] for n in robot.joint_names]
    jadr = {k: 7 + robot.find_joints([k])[0][0]
            for k in ('head_yaw', 'tail_yaw', 'tail_pitch')}
    sites = [n.split('/')[-1] for n in robot.site_names]
    sw = sites.index('left_foot')
    st = sites.index('right_foot')
    root_bid = int(robot.indexing.root_body_id)

    paused = [False]

    def key_cb(k):
        if k in (ord('R'), ord('r')):
            # 重置后**自动暂停**:否则主循环立刻 step 一步, 单脚站出生姿态只闪一下
            # 就被策略改掉了(用户实测反馈"按 reset 只显示一瞬间")。
            env.reset()
            paused[0] = True
            print('[viewer] 已重置到单脚站出生态并暂停;看清后按空格开始', flush=True)
        elif k == 32:
            paused[0] = not paused[0]
            print(f'[viewer] {"暂停" if paused[0] else "继续"}', flush=True)

    obs, _ = env.reset()
    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as h:
        h.cam.distance, h.cam.azimuth, h.cam.elevation = .55, 135., -12
        h.cam.lookat[:] = [0., 0., .11]
        while h.is_running():
            if not paused[0]:
                o = obs['actor'] if isinstance(obs, dict) else obs
                if o.dim() > 2:
                    o = o.flatten(1)
                onx = o.detach().numpy().astype(np.float32)
                act = np.stack([sess.run(None, {'obs': onx[b:b + 1]})[0][0]
                                for b in range(onx.shape[0])])
                obs, rew, term, trunc, _ = env.step(torch.as_tensor(act))
            d = env.sim.data
            data.qpos[:] = d.qpos[0].numpy()
            data.qvel[:] = d.qvel[0].numpy()
            mujoco.mj_forward(model, data)
            # 标注
            sz = float(robot.data.site_pos_w[0, sw, 2]) * 1000
            com = np.asarray(d.subtree_com[0, root_bid, :2])
            foot = robot.data.site_pos_w[0, st, :2].numpy()
            cd = float(np.linalg.norm(com - foot)) * 1000
            gz = robot.data.projected_gravity_b[0].numpy()
            tilt = math.degrees(math.acos(min(1., abs(gz[2]))))
            hy = math.degrees(float(d.qpos[0, jadr['head_yaw']]))
            ty = math.degrees(float(d.qpos[0, jadr['tail_yaw']]))
            tp = math.degrees(float(d.qpos[0, jadr['tail_pitch']]))
            # set_texts 参数: (font, gridpos, text1, text2)
            h.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150.value,
                         mujoco.mjtGridPos.mjGRID_TOPLEFT,
                         f'摆动脚 {sz:5.1f}mm  质心距 {cd:5.1f}mm  倾角 {tilt:4.1f}°',
                         f'头偏航 {hy:+5.0f}°  尾偏航 {ty:+5.0f}°  尾俯仰 {tp:+5.0f}°'))
            h.sync()
            time.sleep(.01)


if __name__ == '__main__':
    main()
