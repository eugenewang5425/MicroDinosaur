"""单脚站立:时长与稳定性测试。

之前只报了姿态质量(摆动脚高/质心距/倾角),没报"能站多久"。
训练时 episode 只有 6 秒,超过 6 秒是**外推**,必须单独测。

判据(与训练同源):
  - 硬失败:躯干倾角 > 70°(fell_over 阈值)
  - 软失败:摆动脚触地(foot_contact_penalty 的门)
  - 出界:  底座离出生点 > 1.5m
输出:存活时长 + 分段稳定性(每 10 秒的姿态统计),判断是"稳如磐石"还是"缓慢发散"。
"""
import argparse
import math
import sys
import json
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from flamingo_cfg import build_config  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from evaluate_policy import sha  # noqa: E402

FALL_TILT_DEG = 70.
DRIFT_LIMIT_M = 1.5


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--seconds', type=float, default=60.)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    task, cfg = build_config(envs=len(a.seeds), seed=47, standing_prob=0.)
    env = ManagerBasedRlEnv(cfg.env, device=a.device)
    sess = ort.InferenceSession(str(a.checkpoint), providers=['CPUExecutionProvider'])
    robot = env.scene['robot']
    sites = [n.split('/')[-1] for n in robot.site_names]
    sw = sites.index('left_foot')
    root_bid = int(robot.indexing.root_body_id)
    obs, _ = env.reset()
    origin = env.sim.data.qpos[:, :2].clone()

    n = len(a.seeds)
    alive = np.ones(n, bool)
    die_t = np.full(n, np.nan)
    die_why = [''] * n
    hist = {i: [] for i in range(n)}
    steps = int(a.seconds / .02)
    for k in range(steps):
        o = obs['actor'] if isinstance(obs, dict) else obs
        if o.dim() > 2:
            o = o.flatten(1)
        onx = o.detach().cpu().numpy().astype(np.float32)
        act = np.stack([sess.run(None, {'obs': onx[b:b + 1]})[0][0] for b in range(n)])
        obs, rew, term, trunc, _ = env.step(torch.as_tensor(act, device=env.device))
        d = env.sim.data
        gz = robot.data.projected_gravity_b[:, 2].cpu().numpy()
        tilt = np.degrees(np.arccos(np.clip(np.abs(gz), 0, 1)))
        sz = robot.data.site_pos_w[:, sw, 2].cpu().numpy() * 1000
        com = d.subtree_com[:, root_bid, :2].cpu().numpy()
        drift = np.linalg.norm(d.qpos[:, :2].cpu().numpy() - origin.cpu().numpy(), axis=1)
        for i in range(n):
            if not alive[i]:
                continue
            hist[i].append((k * .02, tilt[i], sz[i]))
            why = None
            if tilt[i] > FALL_TILT_DEG:
                why = '倾角超 70°'
            elif sz[i] < 2.0:
                why = '摆动脚触地'
            elif drift[i] > DRIFT_LIMIT_M:
                why = f'漂移 {drift[i]:.2f}m'
            if why:
                alive[i] = False
                die_t[i] = k * .02
                die_why[i] = why

    print(f'策略 {a.checkpoint.name} sha={sha(a.checkpoint)[:12]} 测试时长 {a.seconds:.0f}s '
          f'({"GPU" if a.device == "cuda" else "CPU"})', flush=True)
    results = []
    for i in range(n):
        h = np.array(hist[i]) if hist[i] else np.zeros((0, 3))
        row = dict(seed=a.seeds[i],
                   survived_s=float(die_t[i]) if not np.isnan(die_t[i]) else a.seconds,
                   failed=not alive[i], reason=die_why[i],
                   still_alive=bool(alive[i]))
        if len(h):
            row['tilt_mean_deg'] = round(float(h[:, 1].mean()), 2)
            row['tilt_max_deg'] = round(float(h[:, 1].max()), 2)
            row['swingz_mean_mm'] = round(float(h[:, 2].mean()), 1)
            row['swingz_min_mm'] = round(float(h[:, 2].min()), 1)
            # 分段(每 10s)看是否缓慢发散
            seg = []
            for s in range(0, int(a.seconds), 10):
                m = (h[:, 0] >= s) & (h[:, 0] < s + 10)
                if m.sum() > 10:
                    seg.append(f'{s}-{s+10}s: 倾角{h[m, 1].mean():4.1f}° '
                               f'摆动脚{h[m, 2].mean():4.0f}mm')
            row['segments'] = seg
        results.append(row)
        tag = '存活' if alive[i] else f'失败({die_why[i]})'
        print(f"  seed{a.seeds[i]}: 站立 {row['survived_s']:5.1f}s  {tag}"
              + (f"  倾角均 {row.get('tilt_mean_deg')}° / 峰 {row.get('tilt_max_deg')}°"
                 f"  摆动脚均 {row.get('swingz_mean_mm')}mm (最低 {row.get('swingz_min_mm')}mm)"
                 if 'tilt_mean_deg' in row else ''), flush=True)
        for s in row.get('segments', []):
            print(f'        {s}', flush=True)
    dest = ROOT / '20260914_corrective/evaluation' / f'flamingo_duration_{a.checkpoint.stem}'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'duration.json').write_text(json.dumps(
        dict(checkpoint=str(a.checkpoint), sha=sha(a.checkpoint),
             seconds=a.seconds, results=results), indent=2), encoding='utf-8')
    print(f'[write] {dest / "duration.json"}')


if __name__ == '__main__':
    main()
