"""Reassess existing evidence within the user's target delay/damping recipe."""
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent
ROOT = BASE/'20260913_delay_coverage'
paths = [BASE/'20260913_reward_delay_audit'/p/'matrix.json'
         for p in ('delay_matrix','delay_native_extra','delay_rewardfix','delay_control101','delay_alpha07')]
paths += [ROOT/'delay_matrix/matrix.json']
all_rows = [r for p in paths for r in json.loads(p.read_text())]
selected = [r for r in all_rows if r['command_ms'] in (5,10,15)
            and r['position_ms'] in (20,40) and r['velocity_ms']==20]
labels = {'v7':'原v7参考','candidate':'首轮CAD校准200次', 'rewardfix':'奖励修复200次',
          'alpha09_control101':'奖励修复101次控制', 'alpha07':'滤波0.7的101次', 'delay0':'本次延迟扩展101次'}
aggregate = {}
for name in labels:
    rows = [r for r in selected if r['policy']==name]
    assert len(rows)==6
    assert len({(r['command_ms'],r['position_ms'],r['velocity_ms']) for r in rows})==6
    s = [t for r in rows for t in r['trials']['stand']]
    f = [t for r in rows for t in r['trials']['forward']]
    aggregate[name] = {
        'standing_head_rms_deg_s':float(np.mean([t['head_angular_speed_rms_deg_s'] for t in s])),
        'standing_worst_yaw_deg_5s':max(abs(t['yaw_drift_deg']) for t in s),
        'forward_vx_m_s':float(np.mean([t['vx_body_m_s'] for t in f])),
        'forward_abs_lateral_mm_6s':float(np.mean([abs(t['lateral_displacement_mm']) for t in f])),
        'forward_abs_yaw_deg_6s':float(np.mean([abs(t['yaw_drift_deg']) for t in f])),
        'falls':sum(t['fell'] for t in s+f)}
assert len({r['plant_sha256'] for r in selected})==1
report = {'criteria':'current provisional S288 operating envelope; zero delay excluded from selection',
          'nominal_damping':0.8,'training_damping_range':[.8*.62,.8*1.25],
          'command_delay_ms':[5,10,15],'position_delay_ms':[20,40],'velocity_delay_ms':20,
          'imu_delay_ms':0,'nominal_dynamics':True,'new_training':False,
          'hardware_timings_measured':False, 'aggregate':aggregate,
          'reused_trial_count':sum(len(ts) for r in selected for ts in r['trials'].values()),
          'selected_rows':selected}
(ROOT/'operating_envelope.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
lines = ['# 当前验收：只看目标延迟与阻尼区间','',
 '**本页按用户最新要求修正验收口径：不优化零延迟或零阻尼。零延迟只保留为历史诊断，不作为策略选择的否决条件。本页优先于此前报告中把零延迟列入主要门槛的表述。**','',
 '本轮没有进行零阻尼训练：名义kd始终为0.8，训练随机化为0.496–1.0。本次已完成的延迟扩展实验将下限加到0，但默认训练配置始终保持原范围；该实验候选没有替换发布策略。后续不继续围绕零延迟做优化。','',
 '## 目标区间内重算已有证据','',
 '电机目标5/10/15 ms × 位置反馈20/40 ms × 速度反馈20 ms，共6种组合；每种3个初态的站立与前进。名义kd=0.8，IMU新鲜，无噪声与物理随机化。这里是当前S288名义配方，不是已测得的硬件时序。','',
 '| 策略 | 站立头速RMS °/s | 最差站立偏航 °/5s | 前进速度 m/s | 横移 mm/6s | 前进偏航 °/6s |',
 '|---|---:|---:|---:|---:|---:|']
for name,label in labels.items():
    a = aggregate[name]
    lines.append('| '+label+' | '+' | '.join(f'{a[k]:.3f}' for k in a if k!='falls')+' |')
lines += ['',
 '本次延迟扩展相对于同预算奖励修复控制组，在目标区间的平均横移940→344 mm、前进偏航52.2→24.0°、最差站立偏航35.8→2.8°，有明确局部收益。但它比原v7的直走横移146 mm更大、站立头速也更高，因此还不能宣称整体步态更好。此判断不依赖任何零延迟结果。', '',
 '原v7更慢（0.220 m/s，目标0.55），首轮CAD校准提高到0.276 m/s但横移218 mm；必须同时看速度、抖动、偏航和横移，不能只用某一项选默认模型。所有比较只覆盖同名义物理和少数初态。', '',
 f'本页重用了{report["reused_trial_count"]}条已有轨迹，没有重训、没有新增仿真次数。不同训练预算/滤波的策略用于行为参考；只有101次同源同配方的控制与延迟扩展构成该变量的匹配训练对照。', '',
 '## 后续工作','',
 '- 保留kd=0.8及当前5–15/20–40/20 ms名义延迟配方；继续检查目标区间内的抖动、偏航、横移和指令跟踪。',
 '- 左右臂指令镜像错误已经独立修复，4项回归与64×5训练/导出冒烟通过；该修复尚无正式步态结果。',
 '- 下一步使用修正镜像的81维配方建立一致控制，再做头部gyro输入对照。无需先解决零延迟测试才能进入gyro实验。ERPO继续暂缓。',
 '- 原发布ONNX保留；未控制硬件、未提交或推送。', '',
 '[机器可读明细](operating_envelope.json) · [此次训练与历史诊断](RESULTS.md) · [10/20 ms视频](video_delay/rollouts.mp4)', '']
(ROOT/'OPERATING_ENVELOPE.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'target_conditions':6,'reused_trials':report['reused_trial_count'],'new_training':False},indent=2))
