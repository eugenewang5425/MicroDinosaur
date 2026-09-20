"""单脚站立可行性探针:脚本强制抬左脚,测物理上的抬脚极限与稳定性。

为什么做这个(2026-09-18):单脚站立训练四轮都没抬起来,日志显示
`single_leg_lift` 单步值几乎为 0 —— **策略从未让脚离地过**,不是"抬了站不住"。
所以要先回答:这台机器人**物理上**能不能静态抬起一条腿?
  - 若能(抬到 X mm 仍不倒) -> 是探索/奖励问题,继续调训练;
  - 若一抬就倒 -> 是"静态单脚平衡"这个新控制问题,需要先教平衡。

方法:策略照常控制全身,但在左腿的髋/膝/踝上叠加常数偏置(模拟抬腿),
扫描偏置幅度,记录脚高度峰值、是否跌倒、站住时长。
"""
from pathlib import Path
import sys
import itertools
import json

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from corrective_common import OUT  # noqa: E402
from evaluate_owned_terrain import experiment, evaluate  # noqa: E402
from evaluate_policy import sha  # noqa: E402

POLICY = Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_single_leg'
              '/20260918_C_cmd_lift10/candidate.onnx')


def run_bias(knee_deg, hip_deg, ankle_deg, seed=1, seconds=8.0):
    """某组偏置下的表现。校准门会随机拒绝部分种子(RECIPE 惯例),自动换种子重试。"""
    for sd in (seed, seed + 1, seed + 2, seed + 3, seed + 4):
        try:
            return _run_bias_once(knee_deg, hip_deg, ankle_deg, sd, seconds)
        except ValueError as exc:
            if 'Calibration' not in repr(exc):
                raise
    raise RuntimeError('连续 5 个种子校准被拒')


def _run_bias_once(knee_deg, hip_deg, ankle_deg, seed, seconds=16.0):
    case = dict(terrain='flat', scenario='stand', program='legacy', depth=0,
                delay=10, speed=None, seconds=seconds)
    e = experiment(str(POLICY), case)
    s = e.sim
    m = s.model
    fl = m.geom('robot/left_foot_collision').id
    jn = s.names
    idx = {k: jn.index(k) for k in ('left_knee', 'left_hip_pitch', 'left_ankle')}
    bias = {idx['left_knee']: np.deg2rad(knee_deg),
            idx['left_hip_pitch']: np.deg2rad(hip_deg),
            idx['left_ankle']: np.deg2rad(ankle_deg)}
    rows = []
    prev_tt = s.transform_target

    def biased(target, obs):
        t = target.copy()
        # 只在 rollout 期加偏置:校准期要求静止,加偏置会让校准以角运动超限被拒
        if getattr(e, 'active', False):
            for i, b in bias.items():
                t[i] += b
        return prev_tt(t, obs)

    s.transform_target = biased
    prev = s.substep_callback

    def cap():
        prev()
        if not e.active:
            return
        t = s.data.time - 6.
        if t <= 1e-8 or (rows and t - rows[-1]['t'] < .02 - 1e-9):
            return
        d = s.data
        mid = m.geom_dataid[fl]
        a = m.mesh_vertadr[mid]
        n = m.mesh_vertnum[mid]
        v = m.mesh_vert[a:a + n] @ d.geom_xmat[fl].reshape(3, 3).T + d.geom_xpos[fl]
        rot = d.xmat[s.body].reshape(3, 3)
        rows.append(dict(t=t, foot_l=float(v[:, 2].min()),
                         tilt=float(np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1))))))
    s.substep_callback = cap
    mets, _ = evaluate(e, case, seed)
    s.substep_callback = prev
    if not rows:
        return None
    peak = max(r['foot_l'] for r in rows) * 1000
    tilt = max(r['tilt'] for r in rows)
    bal = [r for r in rows if r['tilt'] < 10.]
    return dict(knee_deg=knee_deg, hip_deg=hip_deg, ankle_deg=ankle_deg,
                fall=bool(mets.get('fell')), foot_peak_mm=round(peak, 1),
                tilt_max_deg=round(tilt, 1),
                balanced_frac=round(len(bal) / len(rows), 3))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print(f'策略 {POLICY.name} sha={sha(POLICY)[:12]}', flush=True)
    grid = [(k, h, 0.) for k, h in itertools.product((0, 20, 40, 60), (0, -20))]
    out = []
    for knee, hip, ankle in grid:
        r = run_bias(knee, hip, ankle)
        if r is None:
            continue
        out.append(r)
        print(f"  膝{knee:+3.0f}° 髋{hip:+3.0f}°: 脚峰 {r['foot_peak_mm']:6.1f}mm "
              f"倾角max {r['tilt_max_deg']:5.1f}° 平衡占比 {r['balanced_frac'] * 100:3.0f}% "
              f"{'倒' if r['fall'] else '站住'}", flush=True)
    (OUT / 'single_leg_feasibility.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    ok = [r for r in out if not r['fall'] and r['balanced_frac'] > .8]
    if ok:
        best = max(ok, key=lambda r: r['foot_peak_mm'])
        print(f"\n可行:抬到 {best['foot_peak_mm']:.1f}mm 仍平衡 "
              f"(膝{best['knee_deg']:+.0f}° 髋{best['hip_deg']:+.0f}°)")
    else:
        print('\n**所有偏置都倒了** -> 静态单脚支撑本身是这个机型的新控制问题')


if __name__ == '__main__':
    main()
