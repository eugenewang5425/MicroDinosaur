"""Post-hoc command controllability probe prompted by weak one-sided pivot turns."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from evaluate_gaze_ablation import GazeSim,CASES,run_gaze_trial
from evaluate_policy import sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_head_gaze_ablation'


def main():
    status=json.loads((OUT/'train_status.json').read_text())
    assert status['status']=='COMPLETE'
    policies=[('v7',ROOT/'20260913_handoff/v7_reference.onnx')]
    policies += [(f"s{j['seed']}_{j['arm']}",Path(j['run'])/'candidate.onnx') for j in status['jobs']]
    case=CASES[0];records=[]
    for label,path in policies:
        sim=GazeSim(ROOT/'20260913_handoff/native_v07',path,case);trials=[]
        for name,wz in (('walking_left',.45),('walking_right',-.45)):
            command=np.zeros(18);command[0]=.35;command[2]=wz
            for seed in range(3):
                t,_=run_gaze_trial(sim,command,seed,4);t['task']=name;trials.append(t)
        records.append({'policy':label,'onnx_sha256':sha(path),'case':asdict(case),'trials':trials})
        print(label,flush=True)
    groups=[]
    for r in records:
        for task in ('walking_left','walking_right'):
            ts=[t for t in r['trials'] if t['task']==task];target=ts[0]['command'][2]
            groups.append({'policy':r['policy'],'task':task,'command_wz':target,'command_vx':.35,
                'body_wz_mean_rad_s':float(np.mean([t['body_yaw_rate_mean_rad_s'] for t in ts])),
                'body_vx_mean_m_s':float(np.mean([t['body_vx_mean_m_s'] for t in ts])),
                'yaw_tracking_ratio':float(np.mean([t['body_yaw_rate_mean_rad_s']/target for t in ts])),
                'wrong_direction_trials':sum(t['body_yaw_rate_mean_rad_s']*target<=0 for t in ts),
                'below_half_command_trials':sum(t['body_yaw_rate_mean_rad_s']/target<.5 for t in ts),
                'falls':sum(t['fell'] for t in ts)})
    report={'status':'COMPLETE','post_hoc':True,
        'reason':'Some policies barely responded to one direction of the preregistered pivot turn; test walking turns relevant to later visual path following',
        'primary_ablation_conclusions_unchanged_by_probe':True,'trial_count':sum(len(r['trials']) for r in records),
        'groups':groups,'records':records,'plant_sha256':sha(ROOT/'20260913_handoff/native_v07/nominal.mjb'),
        'implementation_hashes':{p.name:sha(p) for p in [Path(__file__),ROOT/'evaluate_gaze_ablation.py',ROOT/'hardware_sim.py',ROOT/'evaluate_policy.py']}}
    assert report['trial_count']==42
    (OUT/'walking_turn_probe.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# 补充：走动中的双向转向响应','',
        '这是看到原地转向单侧响应弱之后增加的诊断，不属于预注册210条消融评估。'
        '全部7策略、3初态、1.25ms物理步长、10/20/20ms延迟，指令vx=.35m/s、wz=±.45rad/s；2s预热+4s记录。','',
        '| 策略 | 左转实际rad/s | 右转实际rad/s | 左/右转时实际前速m/s | 错向轨迹 | 低于指令一半的轨迹 |','|---|---:|---:|---:|---:|---:|']
    for label,_ in policies:
        l,r=[g for g in groups if g['policy']==label]
        lines.append(f'| {label} | {l["body_wz_mean_rad_s"]:.3f} | {r["body_wz_mean_rad_s"]:.3f} | {l["body_vx_mean_m_s"]:.3f}/{r["body_vx_mean_m_s"]:.3f} | {l["wrong_direction_trials"]+r["wrong_direction_trials"]}/6 | {l["below_half_command_trials"]+r["below_half_command_trials"]}/6 |')
    lines+=['','低于指令一半仅为本次诊断标记，不是事先确定的产品合格线。'
        '正确转向符号与接近目标角速度都需要检查；不能把接近零的同号响应描述为转向已可靠。'
        '有限几条短轨迹通过也不代表全速度范围的控制能力已验证。','']
    (OUT/'WALKING_TURNS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'trials':42,'falls':sum(g['falls'] for g in groups),'groups':groups},indent=2))


if __name__=='__main__':main()
