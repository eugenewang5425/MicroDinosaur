"""Aggregate all confirmation trials, including rejections and adverse cases."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent/'20260914_head_attitude'
METRICS = ('camera_heading_error_rms_deg', 'camera_heading_detrended_rms_deg',
    'camera_yaw_rate_error_rms_deg_s', 'camera_pitch_rms_deg', 'camera_roll_rms_deg',
    'body_vx_mean_m_s', 'heading_error_rms_deg', 'head_joint_torque_peak_nm',
    'head_joint_torque_saturation_fraction', 'camera_relative_lateral_p90_range_mm')


def main():
    records = [json.loads(p.read_text()) for p in sorted((ROOT/'trials').glob('*.json'))]
    groups = {}
    for row in records:
        job = row['job']; key = '__'.join(job[k] for k in ('case', 'policy', 'scenario', 'mode'))
        groups.setdefault(key, []).append(row)
    aggregate = {}
    for key, rows in groups.items():
        valid = [r['metrics'] for r in rows if r['status'] == 'COMPLETE']
        aggregate[key] = dict(attempts=len(rows), completed=len(valid), rejected=len(rows)-len(valid),
            falls=sum(m['fell'] for m in valid),
            mean={k: float(np.mean([m[k] for m in valid])) for k in METRICS} if valid else {})
    result = dict(attempts=len(records), completed=sum(r['status'] == 'COMPLETE' for r in records),
        rejected=sum(r['status'] != 'COMPLETE' for r in records),
        falls=sum(r.get('metrics', {}).get('fell', False) for r in records), groups=aggregate)
    (ROOT/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# 全部确认组', '', '| 条件/策略/任务/模式 | 完成/尝试 | 相机朝向RMS° | 周期偏航RMS° | 偏航角速RMS°/s | 俯仰RMS° | 横滚RMS° | 速度m/s | 跌倒 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key, group in aggregate.items():
        m = group['mean']
        values = ' | '.join(f'{m[k]:.3f}' if m else '—' for k in METRICS[:6])
        lines.append(f'| {key} | {group["completed"]}/{group["attempts"]} | {values} | {group["falls"]} |')
    (ROOT/'TABLES.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for row, policy in enumerate(('v7', 's42_no_neck')):
        for col, (key, label) in enumerate(zip(METRICS[:3], ('Camera heading RMS (deg)', 'Periodic yaw RMS (deg)', 'Yaw rate RMS (deg/s)'))):
            ax = axes[row, col]; modes = ('off', 'filter_only', 'imu')
            values = [aggregate[f'nominal__{policy}__straight__{mode}']['mean'][key] for mode in modes]
            bars = ax.bar(['Body loop', '+ Head filter', '+ Head IMU'], values, color=['#98a4af', '#dbb765', '#318b85'])
            ax.bar_label(bars, fmt='%.2f', padding=3); ax.set_title(policy+' | '+label, fontsize=10)
            ax.set_ylim(0, max(values)*1.22); ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Frozen gait, same body heading loop | 3 paired initial states | 12 s walking', fontsize=12)
    fig.tight_layout(); fig.savefig(ROOT/'comparison.png', dpi=160); plt.close(fig)
    print(json.dumps({k: result[k] for k in ('attempts', 'completed', 'rejected', 'falls')}))
    for key in aggregate:
        if key.startswith('nominal') and '__straight__' in key:
            print(key, json.dumps(aggregate[key]['mean']))


if __name__ == '__main__':
    main()
