"""Summarize all fixed cases, including rejected calibration and adverse outcomes."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent/'20260914_imu_heading'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');args=p.parse_args()
    status=read(ROOT/'status.json')
    if not args.partial:assert status['status']=='COMPLETE' and status['completed']==94
    records=[read(p) for p in sorted((ROOT/'trials').glob('*.json'))]
    dest=ROOT/('preview' if args.partial else 'analysis');dest.mkdir(exist_ok=True)
    plan=read(ROOT/'plan.json');seen=set()
    for r in records:
        key=tuple(r[k] for k in ('case','policy','scenario','seed','mode'));assert key not in seen;seen.add(key)
        assert r['source_hashes']==plan['source_hashes']
    if not args.partial:assert len(records)==94
    valid=[r for r in records if r['status']=='COMPLETE'];rejected=[r for r in records if r['status']!='COMPLETE']
    metrics=['heading_error_rms_deg','heading_error_final_deg','heading_error_last2s_mean_deg',
        'body_estimate_error_rms_deg','head_relative_estimate_error_rms_deg','camera_heading_error_rms_deg',
        'camera_heading_detrended_rms_deg','camera_yaw_rate_error_rms_deg_s','body_vx_mean_m_s',
        'body_lateral_displacement_mm','sensor_age_max_ms','feedback_stale_fraction','estimator_gap_count']
    groups=[]
    for key in sorted(set((r['case'],r['policy'],r['scenario'],r['mode']) for r in valid)):
        rows=[r for r in valid if (r['case'],r['policy'],r['scenario'],r['mode'])==key]
        groups.append(dict(zip(('case','policy','scenario','mode'),key))|{'count':len(rows),
            'seeds':[r['seed'] for r in rows],'falls':sum(r['metrics']['fell'] for r in rows),
            **{m:float(np.mean([r['metrics'][m] for r in rows])) for m in metrics},
            'absolute_final_heading_error_deg':float(np.mean([abs(r['metrics']['heading_error_final_deg']) for r in rows])),
            'absolute_lateral_displacement_mm':float(np.mean([abs(r['metrics']['body_lateral_displacement_mm']) for r in rows]))})
    lookup={tuple(g[k] for k in ('case','policy','scenario','mode')):g for g in groups}
    paired=[]
    for r in valid:
        if r['mode']!='open':continue
        pair=next((t for t in valid if t['mode']=='imu' and all(t[k]==r[k] for k in ('case','policy','scenario','seed'))),None)
        if pair:
            assert r['metrics']['calibration']==pair['metrics']['calibration']
            assert r['metrics']['reference_final_deg']==pair['metrics']['reference_final_deg']
            paired.append({k:r[k] for k in ('case','policy','scenario','seed')}|{
                'heading_rms_difference_deg':pair['metrics']['heading_error_rms_deg']-r['metrics']['heading_error_rms_deg'],
                'camera_rms_difference_deg':pair['metrics']['camera_heading_error_rms_deg']-r['metrics']['camera_heading_error_rms_deg'],
                'speed_difference_m_s':pair['metrics']['body_vx_mean_m_s']-r['metrics']['body_vx_mean_m_s']})
    report={'status':'PARTIAL' if args.partial else 'COMPLETE','attempted':len(records),'completed':len(valid),
        'calibration_rejected':len(rejected),'falls':sum(r['metrics']['fell'] for r in valid),
        'groups':groups,'pairs':paired,'rejected':rejected,'same_initial_calibration_and_user_reference_per_pair':True}
    (dest/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# IMU航向闭环：全部对照表','',
        f'状态：{report["status"]}；尝试{len(records)}条、正常记录{len(valid)}条、校准拒绝{len(rejected)}条、跌倒判据触发{report["falls"]}条。',
        '以下都是固定检查点的评估初态重复，不是新训练seed。原策略与外环控制器参数固定；open与imu仅送入策略的转向命令不同。','',
        '## 主条件（1.25ms、10/20/20ms电机/反馈、20ms外环IMU运输）','',
        '| 策略/任务 | 航向误差RMS°：开环→IMU | 末端绝对航向误差°：开环→IMU | 相机朝向误差RMS°：开环→IMU | 速度m/s：开环→IMU |',
        '|---|---:|---:|---:|---:|']
    for policy in ('v7','s42_no_neck'):
        for task in ('stand','straight','left_then_hold','right_then_hold'):
            keys=[('nominal',policy,task,m) for m in ('open','imu')]
            if not all(k in lookup for k in keys):continue
            a,b=[lookup[k] for k in keys]
            cells=[f'{a[k]:.3f} → {b[k]:.3f}' for k in ('heading_error_rms_deg','absolute_final_heading_error_deg','camera_heading_error_rms_deg','body_vx_mean_m_s')]
            lines.append(f'| {policy}/{task} | '+' | '.join(cells)+' |')
    lines+=['','## 直走敏感性与30s零偏（含全部不利结果）','',
        '| 条件/策略 | 样本数：开环/IMU | 航向RMS°：开环→IMU | 末端绝对误差°：开环→IMU | IMU估计误差RMS° | 最大数据年龄ms |','|---|---:|---:|---:|---:|---:|']
    for case in plan['cases']:
        if case=='nominal':continue
        for policy in ('v7','s42_no_neck'):
            keys=[(case,policy,'straight',m) for m in ('open','imu')]
            if not all(k in lookup for k in keys):continue
            a,b=[lookup[k] for k in keys]
            lines.append(f'| {case}/{policy} | {a["count"]}/{b["count"]} | {a["heading_error_rms_deg"]:.3f} → {b["heading_error_rms_deg"]:.3f} | {a["absolute_final_heading_error_deg"]:.3f} → {b["absolute_final_heading_error_deg"]:.3f} | {b["body_estimate_error_rms_deg"]:.3f} | {b["sensor_age_max_ms"]:.1f} |')
    lines+=['','## 显式仿真真值控制参考','',
        '| 策略/任务 | 真值反馈航向误差RMS° |','|---|---:|']
    for g in groups:
        if g['mode']=='oracle':lines.append(f'| {g["policy"]}/{g["scenario"]} | {g["heading_error_rms_deg"]:.3f} |')
    lines+=['','真值反馈仅用来分辨估计和控制问题；误差更小不保证每条非线性轨迹表现都更好。主结论以传感器闭环为准。','',
        '## 校准拒绝','']
    lines += [f'- {r["case"]}/{r["policy"]}/seed{r["seed"]}/{r["mode"]}: {r["reason"]}' for r in rejected] or ['无。']
    lines+=['','所有控制器参数、IMU假设、完整个体指标和原始轨迹分别在plan.json、trials/和traces/，不因测试结果改变参数。',
        '头部估计误差是相对起步头朝向的变化误差；没有使用动态仿真姿态给IMU解算纠偏。旧策略的重力投影仍保留原来的理想观测，此轮不等于全链路感知或实机验收。','']
    (dest/'TABLES.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,policy in zip(axes,('v7','s42_no_neck')):
        tasks=['straight','left_then_hold','right_then_hold'];xs=np.arange(3);width=.35
        for offset,mode,color in ((-.5,'open','#9aa6b4'),(.5,'imu','#237aa5')):
            vals=[lookup.get(('nominal',policy,t,mode),{}).get('heading_error_rms_deg',np.nan) for t in tasks]
            ax.bar(xs+offset*width,vals,width,label=mode,color=color)
        ax.set_xticks(xs,['Straight','Left then hold','Right then hold']);ax.set_title(policy)
        ax.set_ylabel('Body heading error RMS (deg)');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fixed policies, held-out initial states 1 / 2 / 3; mean per condition')
    fig.savefig(dest/'heading_comparison.png',dpi=160);plt.close(fig)
    for policy in ('v7','s42_no_neck'):
        fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,constrained_layout=True)
        loaded=False
        for mode,color in (('open','#9aa6b4'),('imu','#237aa5')):
            path=ROOT/f'traces/nominal__{policy}__1__straight__{mode}.npz'
            if not path.exists():continue
            a=np.load(path)['trace'];record=next(r for r in valid if r['case']=='nominal' and r['policy']==policy and r['mode']==mode and r['scenario']=='straight' and r['seed']==1)
            # Reconstruct the scoring zero from the final measured error.
            zero=a[-1,2]-a[-1,1]-np.deg2rad(record['metrics']['heading_error_final_deg'])
            body=np.unwrap(a[:,2])-zero
            axes[0].plot(a[:,0],np.rad2deg(body),color=color,label=mode)
            if mode=='imu':axes[0].plot(a[:,0],np.rad2deg(a[:,4]),color='#ce792b',lw=.8,label='IMU estimate (delayed)')
            axes[1].plot(a[:,0],np.rad2deg(np.unwrap(a[:,3])-zero),color=color,label=mode)
            axes[2].plot(a[:,0],a[:,6],color=color,label=mode);loaded=True
        if loaded:
            axes[0].axhline(0,color='#428351',ls='--',label='requested heading');axes[0].legend(fontsize=9)
            for ax,label in zip(axes,['Body yaw (deg)','Camera yaw (deg)','Applied yaw command (rad/s)']):ax.set_ylabel(label);ax.grid(alpha=.2)
            axes[-1].set_xlabel('Time after stationary calibration (s)');fig.suptitle(f'{policy}, preselected initial state 1: straight walking')
            fig.savefig(dest/f'trace_{policy}.png',dpi=160)
        plt.close(fig)
    print(json.dumps({'attempted':len(records),'complete':len(valid),'rejected':len(rejected),'falls':report['falls'],
        'primary_straight':[g for g in groups if g['case']=='nominal' and g['scenario']=='straight']},indent=2))


if __name__=='__main__':main()
