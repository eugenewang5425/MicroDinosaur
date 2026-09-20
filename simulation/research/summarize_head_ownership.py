"""Separate the frozen-policy controller intervention from subsequent learning."""
import json
from pathlib import Path
import numpy as np
from probe_head_ownership import OUT,PREVIOUS
from evaluate_policy import sha


def main():
    old=json.loads((OUT.parent/'20260914_transition_refine/summary.json').read_text())['legacy']
    new=json.loads((OUT/'evaluation/frozen_yaw/matrix.json').read_text())
    assert len(new)==45 and all(r['policy_sha256']==sha(PREVIOUS) for r in new)
    results={}
    for name,terrain,scenario,speed in [('stand','flat','stand',None),('straight','flat','straight',None),('steps10','steps_10','straight',.55)]:
        rows=[r for r in new if r['case']['terrain']==terrain and r['case']['scenario']==scenario and r['case']['speed']==speed and r['case']['program']=='legacy']
        assert len(rows)==3
        keys=['body_vx_mean_m_s','camera_heading_error_rms_deg','camera_pitch_rms_deg','camera_yaw_rate_error_rms_deg_s']
        before={k:old[name]['candidate'][k] for k in keys};before['falls']=old[name]['candidate']['falls']
        after={k:float(np.mean([r['metrics'][k] for r in rows])) for k in keys};after['falls']=sum(r['metrics']['fell'] for r in rows)
        results[name]=dict(policy_baseline=before,encoder_yaw_baseline=after)
    probes={label:json.loads((OUT/f'{label}/matrix.json').read_text()) for label in ('ownership_probe','yaw_probe')}
    p=next(r['metrics'] for r in probes['ownership_probe'] if r['name']=='policy_flat_straight_none_lag10')
    q=next(r['metrics'] for r in probes['yaw_probe'] if r['name']=='yaw_flat_straight_none_lag10')
    gates=dict(no_flat_falls=all(not r['metrics']['fell'] for r in probes['yaw_probe'] if r['terrain']=='flat'),
        nominal_speed_preserved=q['body_vx_mean_m_s']>=.95*p['body_vx_mean_m_s'],
        nominal_yaw_improves30pct=q['camera_heading_error_rms_deg']<=.7*p['camera_heading_error_rms_deg'],
        nominal_pitch_within10pct=q['camera_pitch_rms_deg']<=1.1*p['camera_pitch_rms_deg'])
    (OUT/'control_intervention.json').write_text(json.dumps(dict(policy_sha256=sha(PREVIOUS),groups=results,
        probe_gates=gates,full_ownership_probe_trials=14,yaw_probe_trials=10,frozen_yaw_trials=45),indent=2))
    lines=['# 固定策略的偏航控制干预','','相同model_16150、相同名义10/20/20ms物理/反馈延迟、三个初态和6秒静止校准。旧控制数据来自上一轮最终候选，新偏航控制在本轮重跑。参数及模型哈希核对一致；这是控制执行链的干预，不是新增RL效果。',
        '','原来head_yaw目标以策略输出为基准；现在以延迟编码器读数为基准，叠加同一头部IMU反馈产生的有界增量，再经过原动作滤波、量化和电机延迟。只切换偏航轴；俯仰和横滚继续保留策略前馈及IMU反馈。',
        '','|场景/目标基准|vx m/s|相机朝向RMS°|俯仰RMS°|偏航角速度°/s|跌倒/3|','|---|---:|---:|---:|---:|---:|']
    for name,groups in results.items():
        for label,m in groups.items():
            lines.append(f"|{name}/{label}|{m[keys[0]]:.3f}|{m[keys[1]]:.3f}|{m[keys[2]]:.3f}|{m[keys[3]]:.3f}|{m['falls']}|")
    lines += ['', '同一策略仅改变头部偏航执行链，台阶从2/3跌倒变为0/3；因此上一轮失败不能只归因于遗忘地形。头部动力学与姿态反馈对腿部闭环有影响；具体接触/重心/反作用力矩的机制尚未隔离验证。三个初态的改善不代表任意地形或硬件上都可靠。',
        '', '三轴全部改由编码器/IMU主导的14条先导实验，名义直走俯仰从1.46升至3.74°，没有采用。仅偏航的10条先导实验仍在单个初态上出现1.46→1.67°的俯仰退步，未通过原10%门槛；该失败保留，随后作为研究配置进入有独立绝对门槛的地形续训。没有回改探针门槛。',
        '', '已有45条冻结策略+偏航控制结果与随后45条训练后结果必须分别报告。地形训练增加上/下阶物理交互，目的在巩固分速度通过及走停，而不能把上述控制器收益归给训练。']
    (OUT/'CONTROL_FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(groups=results,probe_gates=gates),indent=2))


if __name__=='__main__':main()
