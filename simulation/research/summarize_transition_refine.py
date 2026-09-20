"""Summarize all predeclared trials, gates, stage traces, and startup diagnostics."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_transition_refine import OUT


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def avg(rows,key):return float(np.mean([r['metrics'][key] for r in rows]))
def phase_avg(rows,phase,key):return float(np.mean([r['metrics']['phases'][phase][key] for r in rows]))


def main():
    previous=read(OUT/'evaluation/previous/transitions/matrix.json')
    candidate=read(OUT/'evaluation/candidate/transitions/matrix.json')
    confirm=read(OUT/'evaluation/candidate/confirmation/matrix.json')
    assert len(previous)==len(candidate)==18 and len(confirm)==12
    all_new=candidate+confirm
    for r in all_new+previous:
        assert r['status']=='COMPLETE'
        assert r['hardware']['command_ms'] in (5,10,15) and r['hardware']['position_ms']==20
    assert len({r['policy_sha256'] for r in all_new})==1
    by_program={}
    for label,rows in [('previous',previous),('candidate',candidate)]:
        by_program[label]={}
        for program in ('hold','resume'):
            for lag in (5,10,15):
                group=[r for r in rows if r['case']['program']==program and r['case']['lag']==lag]
                by_program[label][f'{program}_lag{lag}']=dict(n=len(group),
                    successes=sum(r['metrics']['transition_success'] for r in group),
                    falls=sum(r['metrics']['fell'] for r in group),
                    drop_mm=avg(group,'actual_drop_mm'),return_height_error_mm=avg(group,'return_height_error_mm'),
                    late_net_drift_mm_s=phase_avg(group,'late','net_drift_mm_s'),
                    late_planar_speed_mm_s=phase_avg(group,'late','planar_speed_mean_mm_s'),
                    returned_net_drift_mm_s=phase_avg(group,'returned','net_drift_mm_s'),
                    camera_pitch_rms_deg=avg(group,'camera_pitch_rms_deg'),
                    camera_yaw_rms_deg=avg(group,'camera_heading_error_rms_deg'),
                    late_pitch_mean_deg=phase_avg(group,'late','camera_pitch_mean_deg'),
                    late_pitch_std_deg=phase_avg(group,'late','camera_pitch_std_deg'),
                    resumed_vx_m_s=phase_avg(group,'walk_window','body_vx_mean_m_s'))
    old=read(OUT.parent/'20260914_deep_crouch/summary.json')['groups']['deep']
    anchor=read(OUT/'v7_anchor.json')['trials']
    legacy={}
    keys=['body_vx_mean_m_s','camera_heading_error_rms_deg','camera_yaw_rate_error_rms_deg_s','camera_pitch_rms_deg']
    for name,terrain,scenario in [('stand','flat','stand'),('straight','flat','straight'),('steps10','steps_10','straight')]:
        current=[r for r in confirm if r['case']['program']=='legacy' and r['case']['terrain']==terrain and r['case']['scenario']==scenario]
        v7=[r for r in anchor if r['terrain']==terrain and r['scenario']==scenario]
        assert len(current)==len(v7)==3
        if name=='steps10':
            deep=[r for r in read(OUT.parent/'20260914_deep_crouch/retention/matrix.json') if r['kind']=='terrain']
            old_metrics={k:avg(deep,k) for k in keys}
        else:old_metrics={k:old[f'd0_{scenario}_lag10'][k]['mean'] for k in keys}
        legacy[name]={'v7':{k:avg(v7,k) for k in keys},'previous':old_metrics,'candidate':{k:avg(current,k) for k in keys}}
        legacy[name]['candidate']['falls']=sum(r['metrics']['fell'] for r in current)
        legacy[name]['v7']['falls']=sum(r['metrics']['fell'] for r in v7)
        legacy[name]['previous']['falls']=sum(r['metrics']['fell'] for r in deep) if name=='steps10' else old[f'd0_{scenario}_lag10']['falls']
    crouches=[r for r in all_new if r['case']['depth']>0]
    gates=dict(no_falls=all(not r['metrics']['fell'] for r in all_new),
        depth_and_return=all(abs(r['metrics']['actual_drop_mm']-r['case']['depth'])<=6 and abs(r['metrics']['return_height_error_mm'])<=5 for r in crouches),
        positive_delay_stop_coverage=all(r['metrics']['phases']['late']['net_drift_mm_s']<=10 for r in candidate),
        v7_walking_speed_retained=legacy['straight']['candidate']['body_vx_mean_m_s']>=.95*legacy['straight']['v7']['body_vx_mean_m_s'],
        stand_pitch_improves_20_percent=legacy['stand']['candidate']['camera_pitch_rms_deg']<=.8*legacy['stand']['previous']['camera_pitch_rms_deg'],
        stand_yaw_rate_preserved=legacy['stand']['candidate']['camera_yaw_rate_error_rms_deg_s']<=legacy['stand']['previous']['camera_yaw_rate_error_rms_deg_s'])
    diagnostics={}
    refs=read(OUT/'reference_audit/matrix.json')
    for posture,scenario in [('none','stand'),('crouch40','stand'),('none','straight')]:
        diagnostics[f'{posture}_{scenario}']={}
        for startup in ('calibrated','single_still','single_moving'):
            rows=[r for r in refs if r['posture']==posture and r['scenario']==scenario and r['startup']==startup]
            diagnostics[f'{posture}_{scenario}'][startup]={k:avg(rows,k) for k in keys}
            diagnostics[f'{posture}_{scenario}'][startup]['reference_yaw_disagreement_rms_deg']=float(np.mean([r['reference_yaw_disagreement_rms_deg'] for r in rows]))
    thirty=[r for r in confirm if r['case']['depth']==30]
    data=dict(status='COMPLETE',formal_trials=48,candidate_trials=30,baseline_transition_trials=18,
        startup_diagnostic_trials=27,teacher_capture_trials=15,legacy_parity_trials=1,
        candidate_falls=sum(r['metrics']['fell'] for r in all_new),primary_gates=gates,promoted=all(gates.values()),
        transition_groups=by_program,legacy=legacy,startup_diagnostics=diagnostics,
        nominal30mm=dict(drop_mm=avg(thirty,'actual_drop_mm'),return_height_error_mm=avg(thirty,'return_height_error_mm'),
            late_net_drift_mm_s=phase_avg(thirty,'late','net_drift_mm_s'),successes=sum(r['metrics']['transition_success'] for r in thirty)))
    (OUT/'summary.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    lines=['# 全部固定衔接对照','','每组3个评估初态，单个训练seed。末段为17–20秒；包含停机减速过渡。',
        '','|模型/阶段/延迟|蹲幅mm|回升误差mm|末段净漂移mm/s|末段平均平面速度mm/s|再走速度m/s|俯仰RMS°|朝向RMS°|合格/3|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,groups in by_program.items():
        for name,g in groups.items():
            speed=f"{g['resumed_vx_m_s']:.3f}" if name.startswith('resume') else '—'
            lines.append(f"|{label}/{name}|{g['drop_mm']:.2f}|{g['return_height_error_mm']:.2f}|{g['late_net_drift_mm_s']:.2f}|{g['late_planar_speed_mm_s']:.2f}|{speed}|{g['camera_pitch_rms_deg']:.2f}|{g['camera_yaw_rms_deg']:.2f}|{g['successes']}|")
    lines+=['','## 12秒步态保持','','台阶失败的全程均值包含跌倒后的运动，不作为成功通行速度。', '', '|场景/模型|前向速度m/s|相机朝向RMS°|偏航角速度RMS°/s|俯仰RMS°|跌倒/3|','|---|---:|---:|---:|---:|---:|']
    for name,models in legacy.items():
        for model,m in models.items():lines.append('|'+f'{name}/{model}'+'|'+ '|'.join(f'{m[k]:.3f}' for k in keys)+f"|{m['falls']}|")
    lines+=['','## 预定门槛','']+[f'- {k}: {"PASS" if v else "FAIL"}' for k,v in gates.items()]
    (OUT/'TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
    for label,color in [('previous','#b28139'),('candidate','#267d9a')]:
        z=np.load(OUT/f'evaluation/{label}/transitions/resume_flat_stand_d40_lag10_s1.npz')
        f=z['gaze'];f=f[f[:,0]>=6];t=f[:,0]-6
        axes[0].plot(t,f[:,3]*1000,color=color,label=label)
        axes[1].plot(t,np.rad2deg(np.unwrap(f[:,5])-f[0,4]),color=color)
        axes[2].plot(t,np.rad2deg(f[:,6]),color=color)
        # 0.1s block averages in the speed panel, explicitly labelled.
        n=len(f)//80;axes[3].plot(t[:n*80].reshape(n,80).mean(1),f[:n*80,13].reshape(n,80).mean(1),color=color)
        if label=='candidate':axes[3].plot(z['user_commands'][:,0],z['user_commands'][:,1],'--',color='#686868',label='user vx')
    for ax,title in zip(axes,['Trunk height (mm)','Camera yaw error (deg)','Camera pitch (deg)','Body vx, 0.1s mean (m/s)']):
        ax.set_ylabel(title);ax.grid(alpha=.2);ax.axvspan(2,7,color='#267d9a',alpha=.07);ax.axvspan(11,17,color='#b28139',alpha=.07)
    axes[0].legend();axes[3].legend();axes[-1].set_xlabel('Seconds after 6s stationary calibration')
    fig.suptitle('Preselected 40mm crouch - walk - stop | 10/20/20ms delays | seed 1')
    fig.tight_layout();fig.savefig(OUT/'transition_traces.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(gates=gates,legacy=legacy,thirty=data['nominal30mm']),indent=2))


if __name__=='__main__':main()
