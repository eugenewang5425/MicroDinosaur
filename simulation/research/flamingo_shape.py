"""抬脚形态分析:验证"前伸 / 后折"两种稳定形态是否真的存在、是否会发生切换。

用户观察(2026-09-19):"有两种抬脚的稳定形态,一种是前伸,一种是后折,
站久了有概率两者之间切换"。
做法:长时连续站立,记录摆动腿的髋/膝角时间序列,看分布是否**双峰**(两个吸引子),
以及是否发生**状态跳变**(切换)。同时记录对应的躯干高度/倾角和尾/头配平量,
看两种形态在平衡上有什么差别。
"""
import argparse
import sys
import json
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from flamingo_cfg import build_config, FLAMINGO_POSE_BY_NAME  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from evaluate_policy import sha  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--seconds', type=float, default=120.)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    task, cfg = build_config(envs=1, seed=47, standing_prob=0.)
    env = ManagerBasedRlEnv(cfg.env, device=a.device)
    sess = ort.InferenceSession(str(a.checkpoint), providers=['CPUExecutionProvider'])
    robot = env.scene['robot']
    names = [n.split('/')[-1] for n in robot.joint_names]
    jid = {k: robot.find_joints([k])[0][0] for k in
           ('left_hip_pitch', 'left_knee', 'left_ankle', 'tail_yaw', 'tail_pitch',
            'head_yaw')}
    pose = FLAMINGO_POSE_BY_NAME
    print(f'姿态参考: 左髋俯仰 {np.degrees(pose["left_hip_pitch"]):+.0f}° '
          f'左膝 {np.degrees(pose["left_knee"]):+.0f}° '
          f'(前伸=髋正/膝负; 后折=髋负/膝正)')
    obs, _ = env.reset()
    rows = []
    for k in range(int(a.seconds / .02)):
        o = obs['actor'] if isinstance(obs, dict) else obs
        o = o.flatten(1) if o.dim() > 2 else o
        onx = o.detach().cpu().numpy().astype(np.float32)
        act = sess.run(None, {'obs': onx})[0]
        obs, *_ = env.step(torch.as_tensor(act, device=env.device))
        d = env.sim.data
        q = d.qpos[0]
        rows.append([k * .02,
                     np.degrees(float(q[7 + jid['left_hip_pitch']].cpu())),
                     np.degrees(float(q[7 + jid['left_knee']].cpu())),
                     np.degrees(float(q[7 + jid['left_ankle']].cpu())),
                     np.degrees(float(q[7 + jid['tail_yaw']].cpu())),
                     np.degrees(float(q[7 + jid['tail_pitch']].cpu())),
                     np.degrees(float(q[7 + jid['head_yaw']].cpu())),
                     float(d.qpos[0, 2].cpu()) * 1000])
    h = np.array(rows)
    hip, knee = h[:, 1], h[:, 2]
    print(f'\n=== 站立 {a.seconds:.0f}s 的摆动腿形态 ===')
    print(f'  左髋俯仰: 均 {hip.mean():+6.1f}°  范围 [{hip.min():+6.1f}, {hip.max():+6.1f}]°')
    print(f'  左膝    : 均 {knee.mean():+6.1f}°  范围 [{knee.min():+6.1f}, {knee.max():+6.1f}]°')
    # 双峰检测:把髋角分三档统计占比
    lo, hi = hip.min(), hip.max()
    if hi - lo > 1:
        e1 = (hip < lo + (hi - lo) / 3).mean()
        e2 = ((hip >= lo + (hi - lo) / 3) & (hip < lo + 2 * (hi - lo) / 3)).mean()
        e3 = (hip >= lo + 2 * (hi - lo) / 3).mean()
        print(f'  髋角分布: 低档 {e1*100:.0f}% / 中档 {e2*100:.0f}% / 高档 {e3*100:.0f}%'
              + ('  <- 双端聚集,支持"两种形态"' if e2 < .2 and e1 > .2 and e3 > .2
                 else '  (无明显双峰)'))
    # 切换检测:髋角在相邻 0.5s 内变化 > 25° 视为一次切换
    jump = np.abs(np.diff(hip))
    idx = np.where(jump > 25.)[0]
    if len(idx):
        print(f'  疑似切换 {len(idx)} 次, 发生在 t = '
              + ', '.join(f'{h[i, 0]:.1f}s' for i in idx[:8]))
    else:
        print('  未检测到形态切换(髋角连续)')
    print(f'  分段均值(每 20s): ' + ' | '.join(
        f'{int(s)}-{int(s)+20}s 髋{hip[(h[:,0]>=s)&(h[:,0]<s+20)].mean():+5.0f}°'
        for s in range(0, int(a.seconds), 20)))
    print(f'  尾偏航范围 [{h[:,4].min():+5.0f}, {h[:,4].max():+5.0f}]°  '
          f'尾俯仰 [{h[:,5].min():+5.0f}, {h[:,5].max():+5.0f}]°  '
          f'头偏航 [{h[:,6].min():+5.0f}, {h[:,6].max():+5.0f}]°')
    dest = ROOT / '20260914_corrective/evaluation' / f'flamingo_shape_{a.checkpoint.stem}'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'shape.json').write_text(json.dumps(dict(
        checkpoint=str(a.checkpoint), sha=sha(a.checkpoint),
        hip_mean=float(hip.mean()), hip_min=float(hip.min()), hip_max=float(hip.max()),
        knee_mean=float(knee.mean()), switches=len(idx),
        switch_times=[float(h[i, 0]) for i in idx[:20]],
        series=h[::25].tolist()), indent=2), encoding='utf-8')
    print(f'[write] {dest / "shape.json"}')


if __name__ == '__main__':
    main()
