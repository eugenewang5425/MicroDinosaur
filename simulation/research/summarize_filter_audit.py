"""Matched 101-update EMA experiment, separate from frozen-policy interventions."""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).parent / '20260913_reward_delay_audit'
run = Path(r'D:\microduck_rl\logs\rsl_rl\microdinosaur_v07_calibration\20260913_alpha07_4096x101_retry')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def metrics(d):
    s=d['trials']['stand']; f=[r for r in d['trials']['forward'] if r['amplitude']==1.]
    return {'站立头速RMS（°/s）': np.mean([r['head_angular_speed_rms_deg_s'] for r in s]),
            '站立平均绝对偏航（°/5s）': np.mean([abs(r['yaw_drift_deg']) for r in s]),
            '站立最大绝对偏航（°/5s）': max(abs(r['yaw_drift_deg']) for r in s),
            '前进速度（m/s，指令0.55）': np.mean([r['vx_body_m_s'] for r in f]),
            '前进平均绝对横移（mm/6s）': np.mean([abs(r['lateral_displacement_mm']) for r in f]),
            '前进平均绝对偏航（°/6s）': np.mean([abs(r['yaw_drift_deg']) for r in f]),
            '竖尾最低速度（m/s）': min(r['vx_body_m_s'] for r in d['trials']['tail_up']),
            '跌倒数（16项）': sum(r['fell'] for t in d['trials'].values() for r in t)}


reports={key:read(root/f'{key}.json') for key in ('control101_zero','alpha07_zero','control101_delay','alpha07_delay')}
assert all(r['reset_reproducible'] for r in reports.values())
assert len({r['plant_sha256'] for r in reports.values()})==1
for name,r in reports.items():
    assert r['contract']['action_filter']['alpha_new']==(.7 if name.startswith('alpha07') else .9)
provenance=read(run/'run_provenance.json')
assert provenance['status']=='COMPLETE' and provenance['iterations']==101
assert provenance['action_filter_alpha_new']==.7
onnx_hash=hashlib.sha256((run/'candidate.onnx').read_bytes()).hexdigest()
assert all(reports[n]['onnx_sha256']==onnx_hash for n in ('alpha07_zero','alpha07_delay'))
matrix=read(root/'delay_control101/matrix.json')+read(root/'delay_alpha07/matrix.json')
assert len(matrix)==40
summary={k:metrics(r) for k,r in reports.items()}
summary.update(run=provenance, configuration_check=read(root/'filter_configuration_comparison.json'),
               candidate_onnx_sha256=onnx_hash, matrix_trials=240,
               matrix_falls=sum(t['fell'] for r in matrix for ts in r['trials'].values() for t in ts),
               matrix_aggregate={})
for label in ('alpha09_control101','alpha07'):
    rows=[r for r in matrix if r['policy']==label]
    summary['matrix_aggregate'][label]={
        'mean_stand_head_rms_deg_s': float(np.mean([t['head_angular_speed_rms_deg_s'] for r in rows for t in r['trials']['stand']])),
        'worst_stand_yaw_deg': float(max(abs(t['yaw_drift_deg']) for r in rows for t in r['trials']['stand'])),
        'mean_absolute_lateral_mm': float(np.mean([abs(t['lateral_displacement_mm']) for r in rows for t in r['trials']['forward']]))}
(root/'filter_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# MicroDinosaur：滤波0.7的匹配训练对照', '',
       '**结论：本轮两个新训练候选均不晋级。奖励修复保留；匹配滤波0.7在10/20 ms下改善站立，但横移更差，零延迟站立仍失败。原ONNX保留。**', '',
       '本轮已先修复奖励生命周期和确定HOME问题，并完成200迭代对照；该奖励修复候选站立回归，未替换旧策略。随后固定策略探针发现alpha 0.7是有依据的进一步实验方向，再进行本页的匹配训练。', '',
       '两边均从v7的model_15500.pt开始更新101次，使用相同修复后的奖励代码、物理、seed 42、PPO、时延和课程。唯一训练参数改动为EMA新目标系数0.9→0.7。控制组取200轮运行中第101次更新保存的model_15600.pt；处理组正常完成101轮并保存同编号检查点。课程步计数随源检查点恢复，未重启课程。', '',
       '第一次0.7运行在第25次更新附近中断，只有第1次更新落盘；已标记INTERRUPTED并保留。当前结果来自同源同seed重跑并正常完成的独立retry目录。', '',
       '## 同预算、匹配滤波的16项测试', '',
       '名义v07物理；零延迟列全部反馈新鲜，10/20 ms列为10 ms电机目标、20 ms位置/速度反馈、IMU新鲜；无噪声、无随机物理。每项独立reset，站立5次、前进6次、转向2次、竖尾3次。', '',
       '| 指标 | alpha0.9零延迟 | alpha0.7零延迟 | alpha0.9的10/20 ms | alpha0.7的10/20 ms |',
       '|---|---:|---:|---:|---:|']
for key in summary['control101_zero']:
    lines.append('| '+key+' | '+' | '.join(f'{summary[n][key]:.3f}' for n in reports)+' |')
lines+=['', '## 固定延迟范围', '',
        f"2策略×20时延组合×6次＝240次，跌倒{summary['matrix_falls']}次。时序定义、扰动和压力测试范围见 [审计说明](AUDIT.md)。", '',
        '| 指标 | alpha0.9控制组 | alpha0.7处理组 |', '|---|---:|---:|']
for key,label in [('mean_stand_head_rms_deg_s','跨条件站立头速RMS均值（°/s）'),('worst_stand_yaw_deg','最差单次站立偏航（°/5s）'),('mean_absolute_lateral_mm','前进平均绝对横移（mm/6s）')]:
    lines.append(f"| {label} | {summary['matrix_aggregate']['alpha09_control101'][key]:.3f} | {summary['matrix_aggregate']['alpha07'][key]:.3f} |")
lines+=['', '跨条件均值不能掩盖个别时延点的退步；完整矩阵见 `delay_control101/`、`delay_alpha07/`。单个训练seed和短时仿真不能证明统计显著性或硬件性能。', '',
        '## 固定策略探针与本次训练的区别', '',
        '之前把上一轮200迭代候选的滤波临时改成0.7，在零延迟下站立头速约0.34°/s，10/20 ms下横移约22 mm。这只是同一固定策略的滤波干预；本次训练从原v7检查点开始，且采用修复后的奖励实现，两者起点不同。不能把探针结果当作本次训练结果，也不能据此断言继续训练必然抵消滤波收益。原始探针在 `filter_probe.json`。', '',
        '代码修复解决的是奖励的定义和生命周期，不保证短训策略更好。本轮没有把失败模型选为后续默认基线，也没有以某个安静延迟点宣称完整步态通过。头部gyro输入实验继续排在基础稳定性对照之后。', '',
        '## 使用约束与证据', '',
        '- 该ONNX必须匹配50 Hz策略频率、alpha=0.7、max_delta=0.6 rad/策略步、原19轴HOME和81维观测契约。不要用旧0.9演示器配置直接评价它。',
        f"- 候选：`{provenance['onnx']}`；检查点：`{provenance['final_checkpoint']}`。",
        '- 原发布ONNX和上一轮候选均保留。本轮未进入头部gyro输入实验，未移植ERPO，未控制硬件或推送仓库。',
        '- [滤波对照热图](filter_heatmap.png) · [零延迟视频](video_alpha07_zero/rollouts.mp4) · [10/20 ms视频](video_alpha07_delay/rollouts.mp4)。',
        '- [完整指标](filter_summary.json) · [参数对照](filter_configuration_comparison.json) · [奖励修复结果](RESULTS.md) · [原因审计](AUDIT.md)。', '']
(root/'FILTER_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ('alpha07_zero','alpha07_delay','matrix_aggregate')},ensure_ascii=False,indent=2))
