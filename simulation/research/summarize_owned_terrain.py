"""Full matched evaluation; never replace zero falls with average reward."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_owned_terrain import OUT,cases,key


def read(path):return json.loads(path.read_text())
def avg(rows,k):return float(np.mean([r['metrics'][k] for r in rows]))


def main():
    datasets={label:read(OUT/f'evaluation/{label}/matrix.json') for label in ('frozen_yaw','trained_yaw')}
    groups={}
    for label,rows in datasets.items():
        assert len(rows)==45 and all(r['status']=='COMPLETE' for r in rows)
        groups[label]={}
        for case in cases():
            batch=[r for r in rows if r['case']==case];assert len(batch)==3
            value=dict(n=3,falls=sum(r['metrics']['fell'] for r in batch),
                traversals=sum(r['metrics']['terrain_traversed'] is True for r in batch),
                vx=avg(batch,'body_vx_mean_m_s'),yaw=avg(batch,'camera_heading_error_rms_deg'),
                pitch=avg(batch,'camera_pitch_rms_deg'),yaw_rate=avg(batch,'camera_yaw_rate_error_rms_deg_s'),
                displacement=avg(batch,'forward_displacement_m'),drop=avg(batch,'actual_drop_mm'),
                return_error=avg(batch,'return_height_error_mm'))
            if case['program']=='resume':
                value.update(braking_mm=avg(batch,'braking_net_displacement_mm'),
                    stop_speed_mm_s=float(np.mean([r['metrics']['post_stop']['planar_speed_mean_mm_s'] for r in batch])))
            groups[label][key(case,0).removesuffix('_s0')]=value
    def select(label,**kwargs):return [r for r in datasets[label] if all(r['case'][k]==v for k,v in kwargs.items())]
    baseline=select('frozen_yaw',terrain='flat',scenario='straight',program='legacy')
    walking=select('trained_yaw',terrain='flat',scenario='straight',program='legacy')
    def gates(label):
        rows=datasets[label];terrain=[r for r in rows if r['case']['terrain']!='flat'];crouch=[r for r in rows if r['case']['depth']>0]
        walk=select(label,terrain='flat',scenario='straight',program='legacy');stops=select(label,program='resume')
        return dict(no_falls=not any(r['metrics']['fell'] for r in rows),terrain_all_traversed=all(r['metrics']['terrain_traversed'] for r in terrain),
            flat_speed_preserved=avg(walk,'body_vx_mean_m_s')>=.95*avg(baseline,'body_vx_mean_m_s'),
            walking_direction_rms_under1p5=avg(walk,'camera_heading_error_rms_deg')<=1.5,
            walking_pitch_rms_under2=avg(walk,'camera_pitch_rms_deg')<=2.,
            crouch_depth_and_return=all(abs(r['metrics']['actual_drop_mm']-r['case']['depth'])<=6 and abs(r['metrics']['return_height_error_mm'])<=5 for r in crouch),
            braking_under80mm=all(r['metrics']['braking_net_displacement_mm']<=80 for r in stops),
            stopped_planar_speed_under10mm_s=all(r['metrics']['post_stop']['planar_speed_mean_mm_s']<=10 for r in stops))
    allgates={label:gates(label) for label in datasets}
    summary=dict(status='COMPLETE',formal_trials=90,probe_trials=24,groups=groups,gates=allgates,
        promoted=all(allgates['trained_yaw'].values()),candidate_falls=sum(r['metrics']['fell'] for r in datasets['trained_yaw']),
        source_policy_sha256=datasets['frozen_yaw'][0]['policy_sha256'],trained_policy_sha256=datasets['trained_yaw'][0]['policy_sha256'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# 偏航控制固定后的地形续训对照','','每行3个初态，同一控制/物理链，单个训练seed。失败后的全程速度不能作成功通行速度。',
        '','|组/条件|跌倒/3|穿越/3|vx m/s|朝向RMS°|俯仰RMS°|偏航角速°/s|实际蹲幅mm|回升误差mm|制动mm|停稳后实际速度mm/s|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,values in groups.items():
        for name,v in values.items():
            brake=f"{v['braking_mm']:.2f}" if 'braking_mm' in v else '—'
            stop=f"{v['stop_speed_mm_s']:.2f}" if 'stop_speed_mm_s' in v else '—'
            drop=f"{v['drop']:.2f}" if '_d0_' not in name else '—'
            height=f"{v['return_error']:.2f}" if '_d0_' not in name else '—'
            lines.append(f"|{label}/{name}|{v['falls']}|{v['traversals'] if not name.startswith('flat') else '—'}|{v['vx']:.3f}|{v['yaw']:.2f}|{v['pitch']:.2f}|{v['yaw_rate']:.2f}|{drop}|{height}|{brake}|{stop}|")
    lines+=['','## 预先登记门槛','']+[f'- {label}: '+', '.join(f'{k}={"PASS" if v else "FAIL"}' for k,v in g.items()) for label,g in allgates.items()]
    (OUT/'TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    case=next(c for c in cases() if c['program']=='resume' and c['delay']==10)
    fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
    for label,color in [('frozen_yaw','#b18138'),('trained_yaw','#267e9a')]:
        z=np.load(OUT/f'evaluation/{label}/{key(case,1)}.npz');f=z['gaze'];f=f[f[:,0]>=6];t=f[:,0]-6
        values=[f[:,3]*1000,np.rad2deg(np.unwrap(f[:,5])-f[0,4]),np.rad2deg(f[:,6]),f[:,13]]
        for ax,y in zip(axes,values):ax.plot(t[::8],y[::8],color=color,label=label)
    for ax,name in zip(axes,['Height (mm)','Camera yaw error (deg)','Camera pitch (deg)','Body vx (m/s)']):
        ax.set_ylabel(name);ax.grid(alpha=.2);ax.axvspan(2,7,color='#267e9a',alpha=.07);ax.axvspan(11,17,color='#b18138',alpha=.07)
    axes[0].legend();axes[-1].set_xlabel('Seconds after stationary calibration')
    fig.suptitle('40mm crouch - walk - stop | yaw IMU ownership | nominal delay | seed 1')
    fig.tight_layout();fig.savefig(OUT/'transition_traces.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(gates=allgates,groups=groups),indent=2))


if __name__=='__main__':main()
