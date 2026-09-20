"""Final report from saved A/B evidence, with explicit timing conditions."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).parent / '20260913_reward_delay_audit'
prior = root.parent / '20260913_handoff'
run = Path(r'D:\microduck_rl\logs\rsl_rl\microdinosaur_v07_calibration\20260913_rewardfix_4096x200')


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def metrics(report):
    t = report['trials']
    f = [r for r in t['forward'] if r['amplitude'] == 1.]
    return {
        '站立头部角速度RMS（°/s）': np.mean([r['head_angular_speed_rms_deg_s'] for r in t['stand']]),
        '站立平均绝对偏航（°/5s）': np.mean([abs(r['yaw_drift_deg']) for r in t['stand']]),
        '站立最大绝对偏航（°/5s）': max(abs(r['yaw_drift_deg']) for r in t['stand']),
        '前进速度（m/s，指令0.55）': np.mean([r['vx_body_m_s'] for r in f]),
        '前进平均绝对偏航（°/6s）': np.mean([abs(r['yaw_drift_deg']) for r in f]),
        '前进平均绝对横移（mm/6s）': np.mean([abs(r['lateral_displacement_mm']) for r in f]),
        '前进头部角速度RMS（°/s）': np.mean([r['head_angular_speed_rms_deg_s'] for r in f]),
        '竖尾最低速度（m/s）': min(r['vx_body_m_s'] for r in t['tail_up']),
        '跌倒数（共16项）': sum(r['fell'] for rows in t.values() for r in rows),
    }


reports = {'control_zero': read(prior/'calibration_final.json'),
           'control_delay': read(prior/'calibration_delay_probe.json'),
           'rewardfix_zero': read(root/'rewardfix_zero.json'),
           'rewardfix_delay': read(root/'rewardfix_delay.json')}
assert len({r['plant_sha256'] for r in reports.values()}) == 1
assert all(r['reset_reproducible'] for r in reports.values())
summary = {k: metrics(v) for k, v in reports.items()}
summary['run'] = read(run/'run_provenance.json')
assert summary['run']['status'] == 'COMPLETE'
summary['configuration_check'] = read(root/'configuration_comparison.json')
summary['lifecycle_check'] = read(root/'lifecycle_checks.json')
matrix = read(root/'delay_matrix/matrix.json') + read(root/'delay_native_extra/matrix.json') + read(root/'delay_rewardfix/matrix.json')
assert len(matrix) == 60
summary['matrix_conditions'] = len(matrix)
summary['matrix_trials'] = sum(len(t) for r in matrix for t in r['trials'].values())
summary['matrix_falls'] = sum(t['fell'] for r in matrix for ts in r['trials'].values() for t in ts)
summary['matrix_aggregate'] = {}
for label in ('v7', 'candidate', 'rewardfix'):
    selected = [r for r in matrix if r['policy'] == label]
    summary['matrix_aggregate'][label] = {
        'mean_stand_head_rms_deg_s': float(np.mean([t['head_angular_speed_rms_deg_s'] for r in selected for t in r['trials']['stand']])),
        'worst_stand_yaw_deg': float(max(abs(t['yaw_drift_deg']) for r in selected for t in r['trials']['stand'])),
        'mean_absolute_lateral_mm': float(np.mean([abs(t['lateral_displacement_mm']) for r in selected for t in r['trials']['forward']])),
    }
(root/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
lines = ['# MicroDinosaur：奖励修复与延迟对照结果', '',
         '已修复奖励历史跨回合残留、头部HOME首帧采样问题；完成3项CPU回归、真实8环境生命周期验证、64×5冒烟和4096×200受控续训。', '',
         '本轮与上一轮从同一v7检查点出发，seed 42、物理、奖励权重、PPO、滤波和时延配置一致，只修改奖励实现。参数差异仅运行名称；三处修复作为一组比较，不单独归因。', '',
         '## 受控续训前后', '',
         '每列同一名义物理，16项独立reset测试；站立5次、前进6次、转向2次、竖尾3次。固定延迟列为电机目标10 ms、位置/速度反馈20 ms、IMU新鲜；没有噪声或随机物理。', '',
         '| 指标 | 上轮零延迟 | 修复后零延迟 | 上轮10/20 ms | 修复后10/20 ms |',
         '|---|---:|---:|---:|---:|']
for key in summary['control_zero']:
    lines.append('| '+key+' | '+' | '.join(f'{summary[col][key]:.3f}' for col in ('control_zero','rewardfix_zero','control_delay','rewardfix_delay'))+' |')
lines += ['', '## 延迟范围对照', '',
          f"3种策略×20组固定延迟×每组6次，共{summary['matrix_trials']}次，跌倒{summary['matrix_falls']}次。覆盖的时序、初始扰动和边界见 [审计说明](AUDIT.md)。这不是完整随机时变延迟或硬件验证。", '',
          '| 策略 | 跨20组站立头速RMS均值（°/s） | 最差单次站立偏航（°/5s） | 前进平均绝对横移（mm/6s） |', '|---|---:|---:|---:|']
for label, name in [('v7','原v7'),('candidate','上一轮候选'),('rewardfix','奖励修复候选')]:
    r=summary['matrix_aggregate'][label]
    lines.append(f"| {name} | {r['mean_stand_head_rms_deg_s']:.3f} | {r['worst_stand_yaw_deg']:.3f} | {r['mean_absolute_lateral_mm']:.3f} |")
lines += ['', '跨条件均值只供整体比较，不能掩盖某个延迟点的退步；完整逐条件数据在 `delay_*/`。', '',
          '## 验收与产物', '',
          f"- 零延迟站立平均偏航≤5°/5s：{'达到' if summary['rewardfix_zero']['站立平均绝对偏航（°/5s）'] <= 5 else '未达到'}。",
          f"- 零延迟竖尾最低速度≥0.15 m/s：{'达到' if summary['rewardfix_zero']['竖尾最低速度（m/s）'] >= .15 else '未达到'}。",
          '- 头部抖动、行走偏航与横移需共同验收；奖励上升不能替代行为判断。',
          '- 本轮继续保持81→19，不加入头部gyro，不移植ERPO。原发布ONNX保留，候选独立保存。',
          f"- 检查点：`{summary['run']['final_checkpoint']}`；导出：`{summary['run']['onnx']}`。",
          '- [延迟热图](delay_heatmap_all.png) · [零延迟候选视频](video_zero/rollouts.mp4) · [10/20 ms候选视频](video_delay/rollouts.mp4)。',
          '- [代码与原因审计](AUDIT.md) · [生命周期验证](lifecycle_checks.json) · [配置逐项对照](configuration_comparison.json) · [完整指标](summary.json)。', '']
(root/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps(summary['matrix_aggregate'], ensure_ascii=False, indent=2))
