"""Write the bounded experiment report from saved numerical evidence."""
import json
import re
from pathlib import Path
import numpy as np

root = Path(__file__).parent / '20260913_handoff'
run = Path(r'D:\microduck_rl\logs\rsl_rl\microdinosaur_v07_calibration\20260913_calibration_4096x200')


def load(name):
    return json.loads((root / name).read_text(encoding='utf-8'))


def metrics(report):
    t = report['trials']
    forward = [x for x in t['forward'] if x['amplitude'] == 1.0]
    large = [x for x in t['forward'] if x['amplitude'] == 1.15]
    return {'站立平均绝对偏航（°/5s）': np.mean([abs(x['yaw_drift_deg']) for x in t['stand']]),
            '站立最大绝对偏航（°/5s）': max(abs(x['yaw_drift_deg']) for x in t['stand']),
            '前进速度（m/s，指令0.55）': np.mean([x['vx_body_m_s'] for x in forward]),
            '大幅度前进速度（m/s，scale1.15）': np.mean([x['vx_body_m_s'] for x in large]),
            '前进平均绝对横移（mm/6s）': np.mean([abs(x['lateral_displacement_mm']) for x in forward]),
            '前进抬脚不对称比（较高/较低）': np.mean([x['lift_asymmetry'] for x in forward]),
            '转向平均达成率（%）': np.mean([x['yaw_rate_rad_s'] / x['command_yaw_rate'] * 100 for x in t['turn']]),
            '竖尾平均速度（m/s，指令0.35）': np.mean([x['vx_body_m_s'] for x in t['tail_up']]),
            '竖尾最低速度（m/s）': min(x['vx_body_m_s'] for x in t['tail_up']),
            '测试跌倒数（共16项）': sum(x['fell'] for rows in t.values() for x in rows)}


before = load('v7_reference_v07_before.json')
v12 = load('v12_model16000_v07_before.json')
after = load('calibration_final.json')
delay = load('calibration_delay_probe.json')
delay_before = load('v7_delay_probe.json')
assert len({r['plant_sha256'] for r in (before, v12, after, delay_before, delay)}) == 1
assert all(r['reset_reproducible'] for r in (before, v12, after, delay_before, delay))
assert delay['command_delay_ms'] == delay_before['command_delay_ms'] == 10
assert delay['joint_feedback_delay_ms'] == delay_before['joint_feedback_delay_ms'] == 20
provenance = json.loads((run / 'run_provenance.json').read_text())
assert provenance['status'] == 'COMPLETE'
summary = {'before': metrics(before), 'v12': metrics(v12), 'after': metrics(after),
           'delay_probe': metrics(delay), 'delay_before': metrics(delay_before), 'run': provenance,
           'baseline_checkpoint': 'v7 / model_15500.pt',
           'frozen_snapshot': 'latest CAD v07; current zcode reward recipe; no PPO loss change'}
(root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
lines = ['# MicroDinosaur v07：接管与首轮校准结果', '',
         '本轮已完成母对话接入、私有最新 CAD 校验、转换器修复、基线重测、64×5 冒烟、4096×200 校准、正常保存与 ONNX 导出。', '',
         '**结论：候选在固定延迟下改善站立和速度，但横移退步、零延迟站立未达标，未通过整体步态验收。** 原策略保留。', '',
         '## 同一 v07 名义物理上的对照', '',
         '每项独立 reset；站立5次（含4次±1°扰动）、前进2幅度各3次、双向转向2次、竖尾3次；先稳定2秒，再测4–6秒。无传感器噪声、无延迟、无随机物理。速度为机身坐标，横移为测量起始朝向坐标。', '',
         '| 指标 | v7 原策略 | v12 原策略 | 200迭代候选 |',
         '|---|---:|---:|---:|']
for key in summary['before']:
    lines.append(f"| {key} | {summary['before'][key]:.3f} | {summary['v12'][key]:.3f} | {summary['after'][key]:.3f} |")
lines += ['', '## 零延迟验收判断', '',
          f"- 站立目标约≤5°/5s：{'达到' if summary['after']['站立平均绝对偏航（°/5s）'] <= 5 else '未达到'}（采用5次平均绝对漂移，同时保留最大值）。",
          f"- 竖尾行走目标≥0.15m/s：{'达到' if summary['after']['竖尾最低速度（m/s）'] >= .15 else '未达到'}（3种竖尾指令的最低速度）。",
          '- 左右协调、横漂、跌倒和视频须一起评估；不能用奖励或单项改善替代整体验收。',
          '- 本轮从 v7 检查点开始，继承当前 zcode 的 v12 奖励/随机化配置，换入 v07 名义物理；不是 v7 原配方的单变量因果实验。', '',
          '## 显式延迟探针', '',
          f"固定指令延迟 {delay['command_delay_ms']:.0f} ms、关节反馈延迟 {delay['joint_feedback_delay_ms']:.0f} ms；IMU保持新鲜。没有声称覆盖完整随机时延分布。",
          f"站立平均绝对漂移 {summary['delay_probe']['站立平均绝对偏航（°/5s）']:.2f}°/5s，前进速度 {summary['delay_probe']['前进速度（m/s，指令0.55）']:.3f}m/s，跌倒 {summary['delay_probe']['测试跌倒数（共16项）']} 次。", '',
          '| 延迟条件下的指标 | 校准前v7 | 校准后候选 |', '|---|---:|---:|',
          *[f"| {key} | {summary['delay_before'][key]:.3f} | {summary['delay_probe'][key]:.3f} |" for key in summary['delay_before']], '',
          '该固定延迟条件下，站立和竖尾两项数值目标达到，但前进横移由104 mm增至298 mm/6s，左右抬脚不对称也略增。两项通过不等于整体通过；零延迟回归说明策略仍对时序敏感。', '',
          '## 模型与产物', '',
          '- CAD：`design_source/current/MicroDinosaur_v1.blender`；提交 `15674adf4fcc467da1c0905f641ba1a9de53ec5b`。',
          '- 新仿真质量1.0982148kg；672项台账全部计入。原仿真为1.0548531kg。台账质量未实测。',
          '- 19轴顺序、轴向、轴点和限位不变；头部IMU外参来自仓库，未加入81维actor。',
          f"- 最后检查点：`{provenance['final_checkpoint']}`。",
          f"- 正常结束后的候选：`{provenance['onnx']}`，输入81、输出19，含归一化。",
          '- [零延迟校准前视频](video_before/rollouts.mp4) · [零延迟校准后视频](video_after/rollouts.mp4) · [固定10/20ms延迟候选视频](video_after_delay/rollouts.mp4)。',
          '- [训练曲线](training_curve.png) · [完整指标](summary.json) · [转换验收](integration_checks.json)。',
          '- 旧 `microdinosaur_p2.onnx` 与原CAD保留；没有提交、推送或硬件控制。', '',
          '下一轮先审计奖励历史状态和确定HOME参考，固定配方做延迟覆盖与横移对照；之后再做头部gyro输入实验。暂不移植ERPO。详见 [研究复核](RESEARCH_NOTES.md)。']
(root / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps(summary['after'], ensure_ascii=False, indent=2))
