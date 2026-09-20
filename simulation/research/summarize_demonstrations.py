"""Report every arm against predeclared gates and original frozen baseline."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from demonstration_expert import OUT,ROOT
from evaluate_owned_terrain import cases,key


def main():
    labels=('legacy','archive','demonstration')
    data={label:json.loads((OUT/f'evaluation/{label}/matrix.json').read_text()) for label in labels}
    baseline=json.loads((ROOT/'20260914_imu_owned_terrain/evaluation/frozen_yaw/matrix.json').read_text())
    baseline_speed=float(np.mean([r['metrics']['body_vx_mean_m_s'] for r in baseline if r['case']['terrain']=='flat' and r['case']['scenario']=='straight']))
    groups={};gates={}
    def mean(rows,key):return float(np.mean([r['metrics'][key] for r in rows]))
    for label,rows in data.items():
        assert len(rows)==45 and all(r['status']=='COMPLETE' for r in rows);groups[label]={}
        for case in cases():
            r=[r for r in rows if r['case']==case];assert len(r)==3
            name=key(case,0).removesuffix('_s0')
            g=dict(falls=sum(x['metrics']['fell'] for x in r),traversals=sum(x['metrics']['terrain_traversed'] is True for x in r),
                vx=mean(r,'body_vx_mean_m_s'),distance=mean(r,'forward_displacement_m'),
                yaw=mean(r,'camera_heading_error_rms_deg'),pitch=mean(r,'camera_pitch_rms_deg'),yaw_rate=mean(r,'camera_yaw_rate_error_rms_deg_s'),
                drop=mean(r,'actual_drop_mm'),return_error=mean(r,'return_height_error_mm'),
                max_joint_violation_rad=max(x['metrics']['joint_limit_violation_max_rad'] for x in r))
            if case['program']=='resume':
                g['braking_mm']=mean(r,'braking_net_displacement_mm')
                speeds=[x['metrics']['post_stop']['planar_speed_mean_mm_s'] for x in r]
                g['stop_speeds_mm_s']=speeds;g['stop_passes']=sum(v<=10 for v in speeds)
            groups[label][name]=g
        terrain=[r for r in rows if r['case']['terrain']!='flat'];crouch=[r for r in rows if r['case']['depth']>0]
        walking=[r for r in rows if r['case']['terrain']=='flat' and r['case']['scenario']=='straight'];stops=[r for r in rows if r['case']['program']=='resume']
        gates[label]=dict(no_falls=not any(r['metrics']['fell'] for r in rows),terrain_all_traversed=all(r['metrics']['terrain_traversed'] for r in terrain),
            flat_speed_preserved=mean(walking,'body_vx_mean_m_s')>=.95*baseline_speed,
            walking_yaw_under1p5=mean(walking,'camera_heading_error_rms_deg')<=1.5,walking_pitch_under2=mean(walking,'camera_pitch_rms_deg')<=2,
            crouch_depth_and_return=all(abs(r['metrics']['actual_drop_mm']-r['case']['depth'])<=6 and abs(r['metrics']['return_height_error_mm'])<=5 for r in crouch),
            braking_under80mm=all(r['metrics']['braking_net_displacement_mm']<=80 for r in stops),
            stopped_speed_under10mm_s=all(r['metrics']['post_stop']['planar_speed_mean_mm_s']<=10 for r in stops))
    summary=dict(status='COMPLETE',formal_trials=135,training_seeds=[43],baseline_flat_speed=baseline_speed,groups=groups,gates=gates,
        promotion_eligible={label:all(g.values()) for label,g in gates.items()})
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# 三组完整确认','','各条件三个初态，单训练seed43。legacy保留原约束；archive仅旧示范；demonstration加入新示范。',
        '','|组/条件|跌倒|穿越|vx m/s|朝向°|俯仰°|偏航角速°/s|蹲幅mm|最大关节越限rad|停止后实际速度mm/s|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for label,batches in groups.items():
        for name,g in batches.items():
            stop=' / '.join(f'{x:.2f}' for x in g.get('stop_speeds_mm_s',[])) or '—'
            traversal=f"{g['traversals']}/3" if not name.startswith('flat') else '—'
            drop=f"{g['drop']:.2f}" if '_d0_' not in name else '—'
            lines.append(f"|{label}/{name}|{g['falls']}/3|{traversal}|{g['vx']:.3f}|{g['yaw']:.2f}|{g['pitch']:.2f}|{g['yaw_rate']:.2f}|{drop}|{g['max_joint_violation_rad']:.4f}|{stop}|")
    lines+=['','## 原定门槛','']+[f'- {label}: '+', '.join(f'{k}={"PASS" if v else "FAIL"}' for k,v in g.items()) for label,g in gates.items()]
    (OUT/'TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    low=next(c for c in cases() if c['terrain']=='steps_10' and c['speed']==.2)
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for label in labels:
        with np.load(OUT/f'evaluation/{label}/{key(low,1)}.npz') as z:
            f=z['gaze'];f=f[f[:,0]>=6];t=f[:,0]-6
            axes[0].plot(t[::8],f[::8,1]-f[0,1],label=label)
            axes[1].plot(t[::8],f[::8,13],label=label)
            axes[2].plot(t[::8],np.rad2deg(f[::8,6]),label=label)
    for ax,name in zip(axes,['Forward distance (m)','Body vx (m/s)','Camera pitch (deg)']):ax.set_ylabel(name);ax.grid(alpha=.25)
    axes[0].axhline(.8,color='gray',ls='--');axes[0].legend();axes[1].axhline(.2,color='gray',ls='--')
    axes[-1].set_xlabel('Seconds after calibration');fig.suptitle('10mm steps | requested 0.2m/s | seed 1 | no expert at evaluation')
    fig.tight_layout();fig.savefig(OUT/'low_speed_comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
