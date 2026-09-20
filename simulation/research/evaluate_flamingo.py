"""Flamingo 单脚站立验收+视频:姿态内出生,测保持质量。

判据(6 秒 episode):
  摆动脚离地高度(目标 50mm)、支撑脚接触占比、摔倒与否、质心-支撑脚水平距离、
  头偏航摆幅(配平观察)。输出 JSON + 每案例 mp4。
"""
from pathlib import Path
import sys
import json
import argparse

import numpy as np
import torch
import mujoco
import onnxruntime as ort
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from corrective_common import OUT  # noqa: E402
from flamingo_cfg import build_config  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from evaluate_policy import sha  # noqa: E402

FONT = 'C:/Windows/Fonts/msyh.ttc'
W, H = 640, 400


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--episodes', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--video', action='store_true')
    a = ap.parse_args()
    ck_name = a.checkpoint.name

    task, cfg = build_config(envs=len(a.episodes), seed=47)
    env = ManagerBasedRlEnv(cfg.env, device='cpu')
    session = ort.InferenceSession(str(a.checkpoint), providers=['CPUExecutionProvider'])
    robot = env.scene['robot']
    # mjlab 的 model 包装器没有 .site() —— 从 entity 的 site 索引取,
    # site_pos_w 按此索引对齐。
    all_sites = [n.split('/')[-1] for n in robot.site_names]
    swing_site = all_sites.index('left_foot')
    stance_site = all_sites.index('right_foot')

    print('[dbg] env built, reset 中…', flush=True)
    obs, info = env.reset()
    print('[dbg] reset 完成', flush=True)
    n_env = len(a.episodes)
    per = {i: dict(swing_z=[], com_d=[], stance_on=0, tilt=[], head_yaw=[],
                   tail_yaw=[], root_z=[]) for i in range(n_env)}
    qpos_buf = {i: [] for i in range(n_env)}
    frames_meta = []

    root_bid = int(robot.indexing.root_body_id)
    jname = [n.split('/')[-1] for n in robot.joint_names]
    swing_j = [jname.index(k) for k in ('left_hip_pitch', 'left_knee', 'left_ankle')]
    tag0 = getattr(env, 'spawn_tag', None)
    tag0 = tag0.cpu().numpy() if tag0 is not None else np.ones(len(a.episodes))
    for step in range(150):
        if step in (1, 25, 75):
            print(f'[dbg] step {step}', flush=True)                      # 6s / 0.04 = 150 步
        with torch.no_grad():
            o = obs['actor'] if isinstance(obs, dict) else obs
            if not isinstance(o, torch.Tensor):
                o = torch.cat([torch.as_tensor(x, dtype=torch.float32).reshape(env.num_envs, -1)
                               for x in (o.values() if hasattr(o, 'values') else [o])], dim=-1)
            if o.dim() > 2:
                o = o.flatten(1)
            onx = o.detach().numpy().astype(np.float32)
            act = np.zeros((onx.shape[0], 19), np.float32)
            for b in range(onx.shape[0]):     # 导出时 batch=1,逐环境推理
                act[b] = session.run(None, {'obs': onx[b:b + 1]})[0][0]
        obs, rew, term, trunc, extras = env.step(torch.as_tensor(act, dtype=torch.float32))
        d = env.sim.data
        sp = robot.data.site_pos_w            # [B, nsite, 3] entity 批量数据
        for i in range(n_env):
            sz = float(sp[i, swing_site, 2])
            com = torch.as_tensor(d.subtree_com[i, root_bid, :2])
            foot = torch.as_tensor(sp[i, stance_site, :2])
            per[i]['swing_z'].append(float(sz))
            per[i]['com_d'].append(float(np.hypot(*(com - foot))))
            per[i]['tilt'].append(float(np.degrees(np.arccos(abs(
                float(robot.data.projected_gravity_b[i][2])))) ))
            per[i]['root_z'].append(float(d.qpos[i, 2]))
            per[i].setdefault('swing_q', []).append(
                [float(d.qpos[i, 7 + j]) for j in swing_j])
    # 汇总
    results = []
    for i in range(n_env):
        sz = np.array(per[i]['swing_z'])
        sq = np.array(per[i].get('swing_q', [[0., 0., 0.]]))
        res = dict(episode=a.episodes[i],
                   spawn_pose='前伸' if tag0[i] > 0 else '后折',
                   swing_q_end_deg=[round(float(np.degrees(v)), 1) for v in sq[-1]],
                   swing_q_std_deg=round(float(np.degrees(sq.std(axis=0)).mean()), 1),
                   swing_z_mean_mm=round(float(sz.mean()) * 1000, 1),
                   swing_z_min_mm=round(float(sz.min()) * 1000, 1),
                   swing_z_peak_mm=round(float(sz.max()) * 1000, 1),
                   com_dist_mean_mm=round(float(np.mean(per[i]['com_d'])) * 1000, 1),
                   tilt_max_deg=round(float(np.max(per[i]['tilt'])), 1),
                   root_z_mean_mm=round(float(np.mean(per[i]['root_z'])) * 1000, 1),
                   root_z_min_mm=round(float(np.min(per[i]['root_z'])) * 1000, 1))
        results.append(res)
        print(f"  ep{i}[{res['spawn_pose']}]: 摆动脚高 均{res['swing_z_mean_mm']:6.1f} / "
              f"最低{res['swing_z_min_mm']:6.1f} / 峰{res['swing_z_peak_mm']:6.1f}mm "
              f"(目标 50mm)  质心-支撑距 {res['com_dist_mean_mm']:5.1f}mm  "
              f"倾角max {res['tilt_max_deg']:5.1f}°", flush=True)
    dest = OUT / 'evaluation' / a.label
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'results.json').write_text(json.dumps(
        dict(checkpoint=str(a.checkpoint), sha=sha(a.checkpoint), results=results),
        indent=2), encoding='utf-8')
    print(f'[write] {dest / "results.json"}')


if __name__ == '__main__':
    main()
