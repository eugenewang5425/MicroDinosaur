"""Aggregate paired CPU confirmation, keeping falls out of speed comparisons."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

OUT = Path(__file__).parent/'20260914_terrain_skills'
KEYS = ('body_vx_mean_m_s', 'camera_heading_error_rms_deg',
        'camera_yaw_rate_error_rms_deg_s', 'camera_pitch_rms_deg',
        'camera_roll_rms_deg', 'height_error_rms_mm', 'height_plateau_mean_m',
        'height_returned_mean_m', 'terrain_exposure_fraction', 'forward_displacement_m')


def summarize():
    groups = defaultdict(list)
    for path in sorted((OUT/'evaluation').glob('*/*/matrix.json')):
        label, suite = path.parent.parent.name, path.parent.name
        for record in json.loads(path.read_text()):
            if record['status'] != 'COMPLETE':
                raise AssertionError(f'Incomplete confirmation: {path}')
            m = record['metrics']
            groups[(label, suite, m['terrain'], m['posture'], m['scenario'])].append(m)
    rows = []
    for key, ms in groups.items():
        valid = [m for m in ms if not m['fell']]
        row = dict(zip(('policy', 'suite', 'terrain', 'posture', 'scenario'), key))
        row.update(n=len(ms), falls=sum(m['fell'] for m in ms), seeds=[m['seed'] for m in ms])
        for metric in KEYS:
            values = [m[metric] for m in valid if m[metric] is not None]
            row[metric] = float(np.mean(values)) if values else None
        row['mean_crouch_drop_mm'] = (1000*(row['height_returned_mean_m']-row['height_plateau_mean_m'])
            if valid and row['height_returned_mean_m'] is not None else None)
        rows.append(row)
    (OUT/'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    lines = ['# 配对结果', '', '速度与姿态均值只统计未跌倒试验；失败次数单列。每条12秒，起步前另有6秒v7静止校准；所有确认均开启机身与头部IMU外环。', '',
        'crouch_slow的前进指令为0.2m/s；crouch中的straight为0.55m/s范围外保留性检查。', '',
        '|模型|套件|地形/动作|场景|n|跌倒|速度cm/s|相机朝向RMS°|高度误差mm|回升−低姿mm|',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        fmt = lambda k, scale=1: '—' if r[k] is None else f'{r[k]*scale:.2f}'
        lines.append(f"|{r['policy']}|{r['suite']}|{r['terrain']}/{r['posture']}|{r['scenario']}|{r['n']}|{r['falls']}|{fmt('body_vx_mean_m_s',100)}|{fmt('camera_heading_error_rms_deg')}|{fmt('height_error_rms_mm')}|{fmt('mean_crouch_drop_mm') if r['posture']!='none' else '—'}|")
    (OUT/'TABLES.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(groups=len(rows), trials=sum(r['n'] for r in rows),
        falls=sum(r['falls'] for r in rows)), indent=2))


if __name__ == '__main__': summarize()
