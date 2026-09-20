"""Summarize all preregistered pairs; evaluation resets are not training replicates."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent/'20260914_head_gaze_ablation'
SEEDS=(42,43,44)
ARMS=('control','no_neck_cost')
PRIMARY='nominal_1p25ms'
METRICS={
    'abs_body_yaw_drift_deg':('机身绝对转偏 (°/6s)','Absolute body yaw drift (deg / 6 s)'),
    'abs_contact_duty_diff_pp':('左右接触占比差绝对值 (百分点)','Absolute L-R contact duty difference (pp)'),
    'camera_yaw_rate_error_rms_deg_s':('相机偏航角速度误差RMS (°/s)','Camera yaw-rate error RMS (deg/s)'),
    'camera_yaw_detrended_rms_deg':('相机去趋势偏航RMS (°)','Detrended camera yaw RMS (deg)'),
    'camera_yaw_error_rms_deg':('相机对期望航向误差RMS (°)','Camera heading error RMS (deg)'),
    'body_vx_mean_m_s':('前进速度 (m/s)','Forward body velocity (m/s)'),
    'camera_pitch_error_rms_deg':('相机俯仰误差RMS (°)','Camera pitch error RMS (deg)'),
    'body_yaw_drift_deg':('机身带符号转偏 (°)','Signed body yaw drift (deg)'),
    'contact_duty_diff_pp':('左右接触占比差 L-R (百分点)','Signed L-R contact duty difference (pp)'),
    'camera_relative_lateral_p90_range_mm':('相机相对机身横移P95-P5 (mm)','Relative camera lateral P95-P5 (mm)'),
    'body_yaw_rate_mean_rad_s':('机身平均偏航角速度 (rad/s)','Mean body yaw rate (rad/s)'),
    'body_lateral_displacement_mm':('机身横移 (mm)','Body lateral displacement (mm)'),
}


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def mean_metric(trials,key):
    if key.startswith('abs_'):return float(np.mean([abs(t[key[4:]]) for t in trials]))
    return float(np.mean([t[key] for t in trials]))


def summarize_records(records):
    groups=[]
    for r in records:
        for task in dict.fromkeys(t['task'] for t in r['trials']):
            ts=[t for t in r['trials'] if t['task']==task]
            assert {t['seed'] for t in ts}=={0,1,2} and len(ts)==3
            group={'policy':r['policy'],'case':r['case']['name'],'task':task,
                   'evaluation_initial_states':3,'falls':sum(t['fell'] for t in ts),
                   **{k:mean_metric(ts,k) for k in METRICS}}
            group['complete_stance_duration_mean_LR_s']=[
                float(np.mean(vals)) if (vals:=[t['complete_stance_duration_mean_LR_s'][i]
                    for t in ts if t['complete_stance_duration_mean_LR_s'][i] is not None]) else None
                for i in range(2)]
            group['complete_stance_count_LR']=[sum(t['complete_stance_count_LR'][i] for t in ts) for i in range(2)]
            group['foot_contact_fraction_LR']=np.mean([t['foot_contact_fraction_LR'] for t in ts],axis=0).tolist()
            if task.startswith('turn_'):
                target=ts[0]['command'][2]
                group['command_yaw_rate_rad_s']=target
                group['yaw_rate_tracking_ratio']=group['body_yaw_rate_mean_rad_s']/target
                group['wrong_direction_trials']=sum(t['body_yaw_rate_mean_rad_s']*target<=0 for t in ts)
            groups.append(group)
    return groups


def plot_pairs(groups,dest):
    lookup={(g['policy'],g['case'],g['task']):g for g in groups}
    fig,axes=plt.subplots(2,3,figsize=(13,7.6),constrained_layout=True)
    for ax,key in zip(axes.flat,list(METRICS)[:6]):
        labels_to_place=[]
        for seed in SEEDS:
            labels=[f's{seed}_{arm}' for arm in ARMS]
            if not all((l,PRIMARY,'forward') in lookup for l in labels):continue
            vals=[lookup[(l,PRIMARY,'forward')][key] for l in labels]
            ax.plot([0,1],vals,color='#a4acb5',alpha=.8,zorder=1)
            ax.scatter([0,1],vals,c=['#245780','#dc6d28'],s=45,zorder=2)
            labels_to_place.append((vals[1],seed))
        lo,hi=ax.get_ylim();gap=.055*(hi-lo);previous=-np.inf
        for value,seed in sorted(labels_to_place):
            label_y=max(value,previous+gap);previous=label_y
            ax.annotate(str(seed),(1,value),xytext=(1.12,label_y),fontsize=9,va='center',
                arrowprops={'arrowstyle':'-','color':'#a4acb5','lw':.5})
        ax.set_title(METRICS[key][1],fontsize=10)
        ax.set_xticks([0,1],['Neck cost -0.2','Neck cost 0'])
        ax.set_xlim(-.25,1.35);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same v7 checkpoint, three paired continuation seeds\n'
        'Each point averages 3 evaluation resets | physics 1.25 ms | delays 10 / 20 / 20 ms',fontsize=13)
    fig.savefig(dest/'paired_forward.png',dpi=160);plt.close(fig)


def plot_traces(dest):
    fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True,constrained_layout=True)
    for arm,color,label in zip(ARMS,('#245780','#dc6d28'),('Neck cost -0.2','Neck cost 0')):
        path=ROOT/f'evaluation/s42_{arm}/s42_{arm}__forward.npz'
        if not path.exists():plt.close(fig);return
        a=np.load(path)['trace'];time=a[:,0]-a[0,0]
        axes[0].plot(time,np.rad2deg(np.unwrap(a[:,5])),color=color,label=label,lw=1)
        axes[1].plot(time,np.rad2deg(np.unwrap(a[:,4])-a[0,4]),color=color,lw=1)
        ax=axes[2 if arm=='control' else 3]
        ax.plot(time,1000*a[:,11],color='#36845b',label='Left foot',lw=1)
        ax.plot(time,1000*a[:,12],color='#8b59a5',label='Right foot',lw=1)
        ax.set_ylabel(f'{label}\nFoot height (mm)');ax.legend(loc='upper right',fontsize=8)
    axes[0].set_ylabel('Camera world yaw (deg)');axes[0].legend()
    axes[1].set_ylabel('Body yaw change (deg)');axes[3].set_xlabel('Measured forward time after 2 s warm-up (s)')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Preselected seed 42, nominal reset: optical heading and foot motion\n'
        'Forward command 0.55 m/s; yaw command 0; physics 1.25 ms',fontsize=13)
    fig.savefig(dest/'seed42_forward_traces.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    dest=ROOT/('preview' if args.partial else 'analysis');dest.mkdir(exist_ok=True)
    status=read(ROOT/'train_status.json');eval_status=read(ROOT/'evaluation_status.json')
    if not args.partial:
        assert status['status']==eval_status['status']=='COMPLETE'
        assert len(status['jobs'])==len(eval_status['jobs'])==6
        assert all(read(ROOT/f'pair_s{s}_check.json')['status']=='PASS' for s in SEEDS)
    records=read(ROOT/'reference_eval/matrix.json')
    for seed in SEEDS:
        for arm in ARMS:
            path=ROOT/f'evaluation/s{seed}_{arm}/matrix.json'
            if path.exists():records+=read(path)
            elif not args.partial:raise FileNotFoundError(path)
    seen=set();cases={};hashes=records[0]['implementation_hashes'];plant=records[0]['plant_sha256']
    for r in records:
        key=(r['policy'],r['case']['name']);assert key not in seen;seen.add(key)
        assert r['plant_sha256']==plant and r['implementation_hashes']==hashes
        old=cases.setdefault(r['case']['name'],r['case']);assert old==r['case']
        assert min(r['case'][k] for k in ('command_ms','position_ms','velocity_ms'))>0
        assert r['case']['kd_scale']==1
    count=sum(len(r['trials']) for r in records)
    if not args.partial:assert count==210 and len(seen)==28
    groups=summarize_records(records);lookup={(g['policy'],g['case'],g['task']):g for g in groups}
    paired=[]
    for case in cases:
        for task in ('forward','stand','turn_left','turn_right'):
            for key in METRICS:
                pairs=[]
                for seed in SEEDS:
                    keys=[(f's{seed}_{arm}',case,task) for arm in ARMS]
                    if not all(k in lookup for k in keys):continue
                    c,t=[lookup[k][key] for k in keys]
                    pairs.append({'training_seed':seed,'control':c,'no_neck_cost':t,'difference':t-c})
                if not pairs:continue
                c=float(np.mean([p['control'] for p in pairs]));t=float(np.mean([p['no_neck_cost'] for p in pairs]))
                paired.append({'case':case,'task':task,'metric':key,'pairs':pairs,
                    'control_mean':c,'no_neck_cost_mean':t,'mean_difference':t-c,
                    'lower_in_seed_count':sum(p['difference']<0 for p in pairs)})
    report={'status':'PREVIEW' if args.partial else 'COMPLETE','trial_count':count,
        'falls':sum(g['falls'] for g in groups),'independent_training_units':'3 paired warm-start seeds, not 9 evaluation resets',
        'groups':groups,'paired_metrics':paired,'plant_sha256':plant,'evaluation_implementation_hashes':hashes}
    (dest/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# 颈部角速度惩罚配对消融结果','',
        '状态：'+report['status']+'。所有结果均保留；不按最好训练seed挑选候选。','',
        '三个配对续训seed（42/43/44），各组从同一v7检查点继续1024环境×101更新。'
        '每组仅将head_gaze权重−0.2改为0，保留相机朝向与头部姿态奖励。'
        '相同初始权重不等于从零训练的三个独立策略；每点先平均3个评估初态。','',
        f'评估共{count}条轨迹，{report["falls"]}条触发跌倒判据。主条件为1.25ms物理步长、10/20/20ms指令/位置/速度反馈；名义kd=0.8，未测试零延迟/零阻尼。','',
        '## 主条件：前进0.55m/s，零转向指令','',
        '| 指标 | 保留−0.2 | 关闭0 | 关闭后数值较低的seed |','|---|---:|---:|---:|']
    for p in paired:
        if p['case']==PRIMARY and p['task']=='forward':
            lines.append(f'| {METRICS[p["metric"]][0]} | {p["control_mean"]:.3f} | {p["no_neck_cost_mean"]:.3f} | {p["lower_in_seed_count"]}/{len(p["pairs"])} |')
    lines+=['','“数值较低”不总是更好：速度需接近指令，带符号漂移需接近0；绝对误差先对每条轨迹取绝对值再平均，避免正负抵消。','',
        '![三个训练seed的配对结果](paired_forward.png)','','## 各seed的主指标','',
        '| 训练seed | 机身绝对转偏°：保留→关闭 | 接触差百分点：保留→关闭 | 相机偏航误差°：保留→关闭 | 去趋势偏航°：保留→关闭 |',
        '|---|---:|---:|---:|---:|']
    for seed in SEEDS:
        keys=[(f's{seed}_{arm}',PRIMARY,'forward') for arm in ARMS]
        if not all(k in lookup for k in keys):continue
        c,t=[lookup[k] for k in keys]
        cells=[f'{c[k]:.3f} → {t[k]:.3f}' for k in ('abs_body_yaw_drift_deg','abs_contact_duty_diff_pp','camera_yaw_error_rms_deg','camera_yaw_detrended_rms_deg')]
        lines.append(f'| {seed} | '+' | '.join(cells)+' |')
    lines+=['','## 步长与延迟覆盖（前进）','',
        '| 条件 | 机身绝对转偏°：保留→关闭 | 相机偏航误差°：保留→关闭 | 相机偏航角速度°/s：保留→关闭 | 速度m/s：保留→关闭 |',
        '|---|---:|---:|---:|---:|']
    for case in cases:
        cells=[]
        for key in ('abs_body_yaw_drift_deg','camera_yaw_error_rms_deg','camera_yaw_rate_error_rms_deg_s','body_vx_mean_m_s'):
            ps=[p for p in paired if p['case']==case and p['task']=='forward' and p['metric']==key]
            cells.append(f'{ps[0]["control_mean"]:.3f} → {ps[0]["no_neck_cost_mean"]:.3f}' if ps else '待完成')
        lines.append(f'| {case} | '+' | '.join(cells)+' |')
    lines+=['','## 站立与双向转向（主条件）','',
        '| 策略 | 站立相机偏航角速度RMS °/s | 站立偏航误差RMS ° | 左转实际rad/s（指令+.45） | 右转实际rad/s（指令−.45） | 错向轨迹 |',
        '|---|---:|---:|---:|---:|---:|']
    for label in dict.fromkeys(g['policy'] for g in groups):
        st=lookup[(label,PRIMARY,'stand')];lt=lookup[(label,PRIMARY,'turn_left')];rt=lookup[(label,PRIMARY,'turn_right')]
        lines.append(f'| {label} | {st["camera_yaw_rate_error_rms_deg_s"]:.3f} | {st["camera_yaw_error_rms_deg"]:.3f} | {lt["body_yaw_rate_mean_rad_s"]:.3f} | {rt["body_yaw_rate_mean_rad_s"]:.3f} | {lt["wrong_direction_trials"]+rt["wrong_direction_trials"]}/6 |')
    lines+=['','## 原始轨迹与解释边界','',
        'seed42在汇总前已选定用于轨迹与视频展示，主结论使用全部三个训练seed。',
        '![seed42时序](seed42_forward_traces.png)','',
        '- 相机偏航取真实camera site的光轴−Z；期望方向=初始机身航向+HOME光轴偏置+转向指令积分。去趋势RMS去掉常量与线性漂移，用于分开看周期转动；绝对朝向误差仍单独报告。',
        '- 机身转偏记录2s预热后的6s变化；相机绝对误差仍相对初始期望前方，二者参考窗口不同，不能混为一个指标。',
        '- 接触判据为法向力>.05N，支撑时长排除窗口截断段。细小离地/触地可分裂接触事件，时长只是诊断；两脚接触时间近似相等不保证落脚位置、力或转矩相等。',
        '- 保留的head_world_gaze训练奖励仍跟随机身当前yaw，本次没有把它改成惯性锁向；评估采用独立目标来暴露这个差距。',
        '- 本次只检验仍启用的颈部三轴角速度惩罚，在既有v7权重上的短续训效果。未启用lean_drift/contact_timing，不能据此否定历史EMA残留问题，也不能证明原偏航只有一个成因。',
        '- 侧向平移只记录，不作为相机稳定性否决项。名义S288协议、正阻尼和当前v07质量几何保持一致；尚缺实测系统辨识、全身碰撞及电流/热模型，短仿真无跌倒不代表实机验收。',
        '- 原始逐条指标在各策略matrix.json中；配对均值、带符号量和完整支撑时长/次数保存在summary.json。没有用9个评估初态冒充9次训练，也未计算小样本显著性。','']
    (dest/'TABLES.md').write_text('\n'.join(lines),encoding='utf-8')
    plot_pairs(groups,dest);plot_traces(dest)
    print(json.dumps({'status':report['status'],'trials':count,'falls':report['falls'],
        'primary':[{k:p[k] for k in ('metric','control_mean','no_neck_cost_mean','lower_in_seed_count')}
            for p in paired if p['case']==PRIMARY and p['task']=='forward']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
