"""Summarize paired jump probes without hiding rejected startup cases."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('D:/项目/miro_dinosaur/research/20260914_jump_refine')


def summarize(rows):
    m = [r['metrics'] for r in rows if r['status'] == 'COMPLETE']
    result = dict(planned=len(rows), completed=len(m), startup_or_execution_failed=len(rows)-len(m))
    for key, predicate in {
        'jump_success': lambda x: x['jump_success'],
        'qualified_air': lambda x: bool(x['qualified_flight_segments']),
        'settled': lambda x: x['settled_final_second'],
        'joint_limit_failed': lambda x: x['joint_limit_excess_rad'] > .02,
        'fell': lambda x: x['fell'],
        'body_contact': lambda x: x['nonfoot_ground_contact'],
    }.items():
        result[key] = sum(bool(predicate(x)) for x in m)
    # Heights include zero for completed runs without qualified flight. Failed
    # calibration has no measured jump height and is not fabricated as zero.
    for key, factor in {
        'max_qualified_com_height_m': 1000,
        'max_clear_flight_s': 1000,
        'maximum_both_foot_clearance_m': 1000,
        'maximum_mesh_floor_penetration_m': 1000,
        'final_second_speed_m_s': 1000,
        'joint_limit_excess_rad': 1,
        'peak_ground_force_n': 1,
        'camera_heading_error_rms_deg': 1,
        'camera_pitch_rms_deg': 1,
        'camera_roll_rms_deg': 1,
    }.items():
        values = np.array([x[key]*factor for x in m])
        result[key] = dict(scale=factor, mean=float(values.mean()), minimum=float(values.min()),
                           maximum=float(values.max())) if len(values) else None
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--labels', nargs='+', default=['source', 'candidate'])
    p.add_argument('--output', default='comparison'); args = p.parse_args()
    data = {label: json.loads((OUT/'evaluation'/label/'matrix.json').read_text()) for label in args.labels}
    summaries = {}; lines = ['# 固定请求的配对确认', '',
        '请求：下蹲30mm保持1.5秒，伸展10mm保持0.7秒，随后回零；每条6秒。',
        '计数分母包含启动拒绝。高度/速度等统计仅来自已执行记录；未达到合格腾空的已执行记录，合格COM跳高记0。', '',
        '| 策略 | 电机包络 | 执行方式 | 完整通过 | 合格腾空 | 停稳 | 越限 | COM跳高均值/范围 mm | 残余速度均值 mm/s |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for label, rows in data.items():
        summaries[label] = {}
        for curve in [False, True]:
            for ret in [False, True]:
                core = [r for r in rows if r['seed'] != 404 and r['hardware']['motor_curve'] == curve
                        and r['return_to_base'] == ret]
                s = summarize(core); key = f'curve{int(curve)}_return{int(ret)}'; summaries[label][key] = s
                n = s['planned']; h = s['max_qualified_com_height_m']; v = s['final_second_speed_m_s']
                lines.append(f"| {label} | {'12V速度降额' if curve else '旧恒峰值'} | {'2.8秒交回v7' if ret else '专家单独'} | "
                    f"{s['jump_success']}/{n} | {s['qualified_air']}/{n} | {s['settled']}/{n} | {s['joint_limit_failed']}/{n} | "
                    f"{h['mean']:.2f} / {h['minimum']:.2f}–{h['maximum']:.2f} | {v['mean']:.2f} |")
                for delay in [5, 10, 15]:
                    subset = [r for r in core if r['hardware']['command_ms'] == delay]
                    if subset: summaries[label][f'{key}_d{delay}'] = summarize(subset)
        for row in rows:
            if row['seed'] == 404:
                summaries[label][f"stress_{row['hardware']['name']}_return{int(row['return_to_base'])}"] = summarize([row])
    lines += ['', '## 压力条件', '', '| 策略 | 条件 | 执行方式 | 完整通过 | 合格COM跳高 mm | 净空腾空 ms | 越限 rad | 停稳 |',
              '|---|---|---|---|---:|---:|---:|---|']
    for label, rows in data.items():
        for r in rows:
            if r['seed'] != 404: continue
            m = r.get('metrics', {})
            if not m:
                lines.append(f"| {label} | {r['hardware']['name']} | {r['return_to_base']} | 启动/执行拒绝 | — | — | — | — |")
            else:
                lines.append(f"| {label} | {r['hardware']['name']} | {'交回v7' if r['return_to_base'] else '专家单独'} | "
                    f"{m['jump_success']} | {1000*m['max_qualified_com_height_m']:.2f} | {1000*m['max_clear_flight_s']:.2f} | "
                    f"{m['joint_limit_excess_rad']:.5f} | {m['settled_final_second']} |")
    lines += ['', '## 拒绝记录', '']
    for label, rows in data.items():
        for r in rows:
            if r['status'] != 'COMPLETE': lines.append(f"- {label}/{r['name']}: {r['error']}")
    (OUT/f'{args.output}.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    (OUT/f'{args.output}.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    colors = ['#697787', '#177e73', '#ba6e16']
    for curve, col in [(False, 0), (True, 1)]:
        name = f'curve{int(curve)}_d10_s401_s288_protocol_return0'
        for color, (label, rows) in zip(colors, data.items()):
            file = OUT/'evaluation'/label/(name+'.npz')
            if not file.exists(): continue
            a = np.load(file)['physics']; mask = (a[:,0] >= 1.3) & (a[:,0] <= 2.4)
            # Absolute COM height cannot be mistaken for jump apex relative to release.
            axes[0,col].plot(a[mask,0], a[mask,3]*1000, color=color, label=label, lw=1.8)
            axes[1,col].plot(a[mask,0], np.min(a[mask,14:16],axis=1)*1000, color=color, label=label, lw=1.8)
        axes[0,col].set_title('Constant peak torque (legacy)' if not curve else 'Speed-dependent torque at 12 V')
        axes[0,col].set_ylabel('Whole-body COM above floor [mm]')
        axes[1,col].set_ylabel('Minimum of both complete foot meshes [mm]')
        axes[1,col].axhline(2, color='#8e554c', ls='--', lw=1, label='2 mm clearance threshold')
        for ax in axes[:,col]:
            ax.set_xlabel('Simulation time [s]'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Same seed 401 / 10 ms command delay / jump expert alone')
    fig.savefig(OUT/f'{args.output}.png', dpi=170); plt.close(fig)
    print(json.dumps({label: {key: {k: value[k] for k in ['planned','completed','jump_success','qualified_air','settled','joint_limit_failed']}
          for key,value in groups.items() if len(key.split('_')) == 2} for label,groups in summaries.items()}, indent=2))


if __name__ == '__main__': main()
