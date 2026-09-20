"""Report fixed gates without promoting a partial terrain success."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from corrective_common import OUT,PREVIOUS
from evaluate_owned_terrain import cases,key


def summarize(rows):
    avg=lambda selected,k:float(np.mean([r['metrics'][k] for r in selected]))
    walk=[r for r in rows if r['case']['terrain']=='flat' and r['case']['scenario']=='straight']
    stop=[r for r in rows if r['case']['program']=='resume']
    crouch=[r for r in rows if r['case']['depth']>0]
    holdout=[r for r in rows if r['suite']=='holdout']
    gates=dict(no_falls=not any(r['metrics']['fell'] for r in rows),
        holdout_low_steps=all(r['metrics']['terrain_traversed'] for r in holdout),
        joint_ranges=all(r['metrics']['joint_limit_violation_max_rad']<=.02 for r in rows),
        flat_speed=avg(walk,'body_vx_mean_m_s')>=.258571,
        camera_yaw=avg(walk,'camera_heading_error_rms_deg')<=1.5,camera_pitch=avg(walk,'camera_pitch_rms_deg')<=2,
        crouch=all(abs(r['metrics']['actual_drop_mm']-r['case']['depth'])<=6 and abs(r['metrics']['return_height_error_mm'])<=5 for r in crouch),
        braking=all(r['metrics']['braking_net_displacement_mm']<=80 for r in stop),
        stopping=all(r['metrics']['post_stop']['planar_speed_mean_mm_s']<=10 for r in stop))
    return dict(trials=len(rows),falls=sum(r['metrics']['fell'] for r in rows),
        flat_vx=avg(walk,'body_vx_mean_m_s'),flat_yaw=avg(walk,'camera_heading_error_rms_deg'),flat_pitch=avg(walk,'camera_pitch_rms_deg'),
        holdout_traversals=sum(r['metrics']['terrain_traversed'] for r in holdout),
        holdout_by_delay={str(d):dict(passed=sum(r['metrics']['terrain_traversed'] for r in holdout if r['case']['delay']==d),
            vx=avg([r for r in holdout if r['case']['delay']==d],'body_vx_mean_m_s')) for d in (5,10,15)},
        stop_passes=sum(r['metrics']['post_stop']['planar_speed_mean_mm_s']<=10 for r in stop),
        stop_speeds={str(d):[r['metrics']['post_stop']['planar_speed_mean_mm_s'] for r in stop if r['case']['delay']==d] for d in (5,10,15)},
        braking_max_mm=max(r['metrics']['braking_net_displacement_mm'] for r in stop),
        joint_violation_max_rad=max(r['metrics']['joint_limit_violation_max_rad'] for r in rows),
        crouch_passes=sum(abs(r['metrics']['actual_drop_mm']-r['case']['depth'])<=6 and abs(r['metrics']['return_height_error_mm'])<=5 for r in crouch),
        gates=gates,promotion_eligible=all(gates.values()))


def main():
    source=[dict(r,suite='retention') for r in json.loads((PREVIOUS/'evaluation/demonstration/matrix.json').read_text())]
    source+=json.loads((OUT/'evaluation/source_holdout/matrix.json').read_text())
    data={'source':source}
    for arm in ('control','corrective'):
        done=json.loads((OUT/f'evaluation/{arm}/complete.json').read_text());assert done['trials']==54
        data[arm]=json.loads((OUT/f'evaluation/{arm}/matrix.json').read_text())
    result={arm:summarize(rows) for arm,rows in data.items()}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# 纠正示范逐条件确认','','单训练seed47；retention为原有初态1–3，holdout为独立初态201–203。',
        '','|组|初态集|条件|种子|通过台阶|跌倒|vx m/s|关节越限rad|停止后速度mm/s|',
        '|---|---|---|---:|---|---|---:|---:|---:|']
    for arm,rows in data.items():
        for r in rows:
            m=r['metrics'];speed=m.get('post_stop',{}).get('planar_speed_mean_mm_s')
            lines.append(f"|{arm}|{r['suite']}|{key(r['case'],0).removesuffix('_s0')}|{r['seed']}|{m['terrain_traversed']}|{m['fell']}|{m['body_vx_mean_m_s']:.4f}|{m['joint_limit_violation_max_rad']:.4f}|{speed if speed is not None else '—'}|")
    (OUT/'TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    case=next(c for c in cases() if c['terrain']=='steps_10' and c['speed']==.2)
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for arm in data:
        folder='source_holdout' if arm=='source' else arm
        with np.load(OUT/f'evaluation/{folder}/{key(case,201)}.npz') as z:
            f=z['gaze'];f=f[f[:,0]>=6];t=f[:,0]-6
            axes[0].plot(t[::8],f[::8,1],label=arm)
            axes[1].plot(t[::8],f[::8,13],label=arm)
    for x in (.30,.48,.66):axes[0].axhline(x,color='gray',ls=':',alpha=.5)
    axes[0].axhline(.8,color='gray',ls='--');axes[1].axhline(.2,color='gray',ls='--')
    axes[0].set_ylabel('Body x (m)');axes[1].set_ylabel('Body vx (m/s)');axes[1].set_xlabel('Seconds after calibration')
    axes[0].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Independent seed 201 | 10mm stairs | 10ms delay | command 0.2m/s')
    fig.tight_layout();fig.savefig(OUT/'holdout_comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
