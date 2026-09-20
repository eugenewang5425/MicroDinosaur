"""Keep follow-up startup and transport results distinct from the frozen matrix."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent/'20260914_imu_heading'


def main():
    records=json.loads((ROOT/'stable_start/matrix.json').read_text());assert len(records)==24
    original=json.loads((ROOT/'analysis/summary.json').read_text())
    groups=[];paired=[]
    for task in ('stand','straight','left_then_hold','right_then_hold'):
        for mode in ('open','imu'):
            ts=[r['metrics'] for r in records if r['scenario']==task and r['mode']==mode];assert len(ts)==3
            groups.append({'scenario':task,'mode':mode,'count':3,
                **{k:float(np.mean([t[k] for t in ts])) for k in ('heading_error_rms_deg','camera_heading_error_rms_deg',
                    'camera_heading_detrended_rms_deg','camera_yaw_rate_error_rms_deg_s','body_vx_mean_m_s',
                    'body_estimate_error_rms_deg','head_relative_estimate_error_rms_deg')},
                'absolute_final_heading_error_deg':float(np.mean([abs(t['heading_error_final_deg']) for t in ts])),
                'absolute_lateral_displacement_mm':float(np.mean([abs(t['body_lateral_displacement_mm']) for t in ts])),
                'falls':sum(t['fell'] for t in ts)})
        for seed in (1,2,3):
            a,b=[next(r['metrics'] for r in records if r['scenario']==task and r['mode']==m and r['seed']==seed) for m in ('open','imu')]
            assert a['calibration']==b['calibration'] and a['reference_final_deg']==b['reference_final_deg']
            paired.append({'scenario':task,'seed':seed,'heading_rms_difference':b['heading_error_rms_deg']-a['heading_error_rms_deg']})
    slow=json.loads((ROOT/'isolated_transport.json').read_text());assert slow['status']=='COMPLETE' and len(slow['records'])==4
    transport=[]
    for mode in ('open','imu'):
        ts=[t for t in slow['records'] if t['mode']==mode]
        transport.append({'mode':mode,'count':len(ts),**{k:float(np.mean([t[k] for t in ts])) for k in (
            'heading_error_rms_deg','body_estimate_error_rms_deg','heading_error_final_deg','sensor_age_max_ms')},'falls':sum(t['fell'] for t in ts)})
    report={'status':'COMPLETE','stable_start_trials':24,'stable_start_groups':groups,'stable_start_pairs':paired,
        'isolated_slow_transport_trials':4,'isolated_slow_transport_groups':transport,
        'all_phases_attempted':original['attempted']+28,'all_phases_completed':original['completed']+28,
        'calibration_rejections_retained':original['calibration_rejected'],
        'all_phases_falls':original['falls']+sum(g['falls'] for g in groups)+sum(g['falls'] for g in transport)}
    (ROOT/'followups_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# 启动流程与独立运输延迟补测','',
        '原94条矩阵完整保留，包括49条校准拒绝。这里是针对失败新增的两项诊断，未修改原矩阵的阈值、控制参数或结果。','',
        '## 稳定v7站立校准后切换高偏航候选','',
        '共24条：3初态×4任务×开环/IMU；切换时保留实际last_action、动作滤波和延迟历史，避免人为制造指令跳变。',
        '两版只改变进入策略的转向速度，不改变候选权重；双IMU均在切换前的同一静止数据上校准。','',
        '| 任务 | 航向RMS°：开环→IMU | 末端绝对误差°：开环→IMU | 相机朝向RMS°：开环→IMU | 速度m/s：开环→IMU |','|---|---:|---:|---:|---:|']
    for task in ('stand','straight','left_then_hold','right_then_hold'):
        a,b=[g for g in groups if g['scenario']==task]
        cells=[f'{a[k]:.3f} → {b[k]:.3f}' for k in ('heading_error_rms_deg','absolute_final_heading_error_deg','camera_heading_error_rms_deg','body_vx_mean_m_s')]
        lines.append(f'| {task} | '+' | '.join(cells)+' |')
    lines+=['','## 仅降低外环IMU速率并增加运输延迟','',
        '保持原来的1.25ms物理步长和10/20/20ms关节执行/反馈，仅把外环IMU变成40ms采样、60ms运输。4条v7直走对照：','',
        '| 模式 | 航向误差RMS° | IMU估计误差RMS° | 最大数据年龄ms |','|---|---:|---:|---:|']
    for g in transport:lines.append(f'| {g["mode"]} | {g["heading_error_rms_deg"]:.3f} | {g["body_estimate_error_rms_deg"]:.3f} | {g["sensor_age_max_ms"]:.1f} |')
    lines+=['','这项补测不把外环采样/延迟退步混同于前轮15/40/20ms关节延迟下的静止校准拒绝。当前固定参数并未覆盖所有IMU采样配置。','']
    (ROOT/'FOLLOWUPS.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    tasks=['straight','left_then_hold','right_then_hold'];xs=np.arange(3)
    for ax,label,data in ((axes[0],'v7: normal startup',[g for g in original['groups'] if g['case']=='nominal' and g['policy']=='v7']),
        (axes[1],'High-drift policy: calibrated using v7',groups)):
        for offset,mode,color in ((-.18,'open','#9aa6b4'),(.18,'imu','#237aa5')):
            vals=[next(g['heading_error_rms_deg'] for g in data if g['scenario']==task and g['mode']==mode) for task in tasks]
            ax.bar(xs+offset,vals,.36,label=mode,color=color)
        ax.set_xticks(xs,['Straight','Left then hold','Right then hold']);ax.set_title(label)
        ax.set_ylabel('Body heading error RMS (deg)');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Sensor-only heading feedback; three held-out initial states per condition')
    fig.savefig(ROOT/'analysis/validated_heading_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
