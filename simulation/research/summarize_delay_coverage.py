"""Finalize artifacts only after training/export/evaluation and independent smoke finish."""
import difflib
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).parent / '20260913_delay_coverage'
OLD = ROOT.parent / '20260913_reward_delay_audit'
RUNS = Path(r'D:\microduck_rl\logs\rsl_rl\microdinosaur_v07_calibration')
RUN = RUNS / '20260913_delay0_4096x101'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(d):
    s = d['trials']['stand']
    f = [r for r in d['trials']['forward'] if r['amplitude']==1]
    return {'站立头速RMS（°/s）':float(np.mean([r['head_angular_speed_rms_deg_s'] for r in s])),
            '站立平均绝对偏航（°/5s）':float(np.mean([abs(r['yaw_drift_deg']) for r in s])),
            '前进速度（m/s，指令0.55）':float(np.mean([r['vx_body_m_s'] for r in f])),
            '前进平均绝对横移（mm/6s）':float(np.mean([abs(r['lateral_displacement_mm']) for r in f])),
            '前进平均绝对偏航（°/6s）':float(np.mean([abs(r['yaw_drift_deg']) for r in f])),
            '竖尾最低速度（m/s）':float(min(r['vx_body_m_s'] for r in d['trials']['tail_up']))}


def main():
    provenance = read(RUN/'run_provenance.json')
    assert provenance['status']=='COMPLETE' and provenance['finite_output_check']
    assert provenance['iterations']==101 and provenance['envs']==4096
    smoke = read(RUNS/'20260913_armmirror_smoke_64x5/run_provenance.json')
    assert smoke['status']=='COMPLETE' and smoke['finite_output_check']
    reports = {'control_zero':read(OLD/'control101_zero.json'), 'delay0_zero':read(ROOT/'delay0_zero.json'),
               'control_delay':read(OLD/'control101_delay.json'), 'delay0_delay':read(ROOT/'delay0_delay.json')}
    assert all(d['reset_reproducible'] for d in reports.values())
    assert len({d['plant_sha256'] for d in reports.values()})==1
    assert all(d['contract']['action_filter']['alpha_new']==.9 for d in reports.values())
    assert all(reports[n]['onnx_sha256']==sha(RUN/'candidate.onnx') for n in ('delay0_zero','delay0_delay'))
    matrix = read(ROOT/'delay_matrix/matrix.json')
    control_matrix = read(OLD/'delay_control101/matrix.json')
    assert len(matrix)==len(control_matrix)==20
    assert all(len(r['trials'][k])==3 for r in matrix for k in ('stand','forward'))
    aggregate = {}
    for name, rows in [('control101',control_matrix),('delay0',matrix)]:
        aggregate[name] = {
            'mean_stand_head_rms_deg_s':float(np.mean([t['head_angular_speed_rms_deg_s'] for r in rows for t in r['trials']['stand']])),
            'worst_stand_yaw_deg':max(abs(t['yaw_drift_deg']) for r in rows for t in r['trials']['stand']),
            'mean_absolute_lateral_mm':float(np.mean([abs(t['lateral_displacement_mm']) for r in rows for t in r['trials']['forward']]))}
    responses = {p.stem:read(p) for p in (ROOT/'walk_response').glob('*.json')}
    assert set(responses)=={'v7','control101','delay0'}
    response_summary = {}
    for name, d in responses.items():
        response_summary[name] = {}
        for command_name in dict.fromkeys(t['name'] for t in d['trials']):
            rows = [t for t in d['trials'] if t['name']==command_name]
            response_summary[name][command_name] = {k:float(np.mean([r[k] for r in rows]))
                for k in ('vx_body_m_s','vy_body_m_s','yaw_rate_rad_s','lateral_mm')}
    source = torch.load(provenance['checkpoint'],map_location='cpu',weights_only=False)
    final = torch.load(provenance['final_checkpoint'],map_location='cpu',weights_only=False)
    before, after = (c['infos']['env_state']['common_step_counter'] for c in (source,final))
    assert after-before==101*24
    assert final['iter']==15600
    logs = EventAccumulator(str(RUN)); logs.Reload()
    log_checks = {}
    for name in ('Episode_Reward/lean_drift','Episode_Reward/contact_timing','Episode_Reward/action_rate_l2',
                 'Episode_Termination/nan_state'):
        events = logs.Scalars(name)
        values = [e.value for e in events]
        assert len(values)==101 and np.isfinite(values).all()
        assert max(values)<=0, (name,max(values))
        log_checks[name] = {'count':len(values),'min':min(values),'max':max(values)}
    new_trials = [t for r in matrix for ts in r['trials'].values() for t in ts]
    new_trials += [t for name in ('delay0_zero','delay0_delay') for ts in reports[name]['trials'].values() for t in ts]
    new_trials += [t for d in responses.values() for t in d['trials']]
    assert len(new_trials)==215
    release = Path(r'D:\microduck_rl\microdinosaur_p2.onnx')
    assert sha(release)==sha(ROOT.parent/'20260913_handoff/v12_model16000.onnx')
    old_sym = RUN/'source_snapshot/5_symmetry_microdinosaur.py'
    new_sym = Path(r'D:\microduck_rl\src\mjlab_microduck\tasks\symmetry_microdinosaur.py')
    patch = ''.join(difflib.unified_diff(old_sym.read_text().splitlines(True),new_sym.read_text().splitlines(True),
                   fromfile='training_snapshot/symmetry_microdinosaur.py',tofile='current/symmetry_microdinosaur.py'))
    (ROOT/'arm_mirror_fix.patch').write_text(patch,encoding='utf-8')
    summary = {'decision':'NOT_PROMOTED', 'provenance':provenance,
               'metrics':{n:metrics(d) for n,d in reports.items()}, 'matrix_aggregate':aggregate,
               'command_response':response_summary, 'new_trials':215,
               'new_trial_falls':sum(t['fell'] for t in new_trials),
               'curriculum_counter_before':before,'curriculum_counter_after':after,
               'training_log_checks':log_checks, 'release_unchanged_sha256':sha(release),
               'candidate_onnx_sha256':sha(RUN/'candidate.onnx'),
               'independent_mirror_fix_smoke':smoke['status'],
               'mirror_fix_in_this_101_update_run':False}
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# MicroDinosaur：延迟覆盖与偏航指令对照','',
      '> 验收口径已按用户最新要求更新：零延迟仅作历史诊断，阻尼始终为0.8。以下保留本次实验记录；策略选择请以[目标工作区间复核](OPERATING_ENVELOPE.md)为准，不以零延迟结果否决候选。','',
      '**结论：扩大延迟支持域改善了部分指标，但零延迟站立和直走仍不合格，新策略不晋级。另确认并修复左右臂指令镜像缺少交换的独立错误；该修复尚未做正式步态训练。**','',
      '## 训练变量与验收范围','',
      '同源v7检查点、seed 42、4096环境、101次PPO更新，保持奖励修复实现、权重、HOME、alpha 0.9、v07物理和81维输入不变。只把电机目标/actor关节位置/actor关节速度延迟下限降为0：分别成为0–15、0–40、0–20 ms。IMU、critic、各延迟上限及保持概率未变。参数对照通过，课程计数372144→374568。','',
      '控制组取已有奖励修复运行的第101次更新检查点，不使用其200次更新终点。扩展随机延迟范围不等于整回合固定零延迟训练；本轮是单训练seed的短训筛查，不是统计显著性结论。见[实验约束](EXPERIMENT.md)及[完整参数差异](configuration_comparison.json)。','',
      '## 同预算的完整测试','',
      '站立取5个初态，每次5秒；前进表取幅度1的3个初态，每次6秒、指令0.55 m/s。10/20 ms指10 ms电机目标与20 ms位置/速度反馈，IMU新鲜；每项先独立重置并预热2秒。名义仿真，无噪声及物理随机化。','',
      '| 指标 | 原范围零延迟 | 扩展范围零延迟 | 原范围10/20 ms | 扩展范围10/20 ms |',
      '|---|---:|---:|---:|---:|']
    for metric in summary['metrics']['control_zero']:
        lines.append('| '+metric+' | '+' | '.join(f'{summary["metrics"][n][metric]:.3f}' for n in reports)+' |')
    lines += ['', '10/20 ms站立改善明显，但前进速度下降且横移仍达658 mm/6秒。零延迟的头速下降，站立偏航和前进横移反而恶化，不能把减抖当作全面稳定。原v7同10/20 ms参考横移约104 mm/6秒；新策略仍明显落后于这一已有直走行为。','',
      '## 20组时延与指令响应','',
      '| 跨20个时延条件的指标 | 原范围控制组 | 扩展范围 |','|---|---:|---:|']
    for key,label in [('mean_stand_head_rms_deg_s','站立头速RMS均值（°/s）'),('worst_stand_yaw_deg','最差单次站立偏航（°/5s）'),('mean_absolute_lateral_mm','前进平均绝对横移（mm/6s）')]:
        lines.append(f'| {label} | {aggregate["control101"][key]:.3f} | {aggregate["delay0"][key]:.3f} |')
    lines += ['', '左右转向的指令符号与实际方向一致，安装的转向奖励正负号检查也通过。问题包含零转向指令下的持续偏置，不是简单把yaw正负号写反。10/20 ms的直走实测平均转速：','',
      '| 策略 | 直走时转速（rad/s，目标0） |','|---|---:|']
    for label in ('v7','control101','delay0'):
        lines.append(f'| {label} | {response_summary[label]["forward_yaw_+0.0"]["yaw_rate_rad_s"]:.4f} |')
    lines += ['', '纯横移±0.25 m/s指令的实测速度也远低于目标；图中保留左右响应及完整直走轨迹，避免仅看某个坐标系下的侧向速度。误差棒表示3个初态的范围，不是置信区间。','',
      f'本轮新增215条短轨迹：120条延迟矩阵、32条扩展行为测试、63条指令响应；跌倒{summary["new_trial_falls"]}次。先前矩阵累计600条，本轮后为720条。无跌倒不代表步态验收通过。', '',
      '已核对零延迟与10/20 ms四场景的中段渲染画面及完整直走轨迹；机器人能够站立、行走、转向、竖尾，但弧线偏移和振动指标仍失败。静态画面本身不能用于证明消抖。', '',
      '- [延迟热图](delay_heatmap.png) · [转向/横移响应与轨迹](walk_response.png)',
      '- [零延迟视频](video_zero/rollouts.mp4) · [10/20 ms视频](video_delay/rollouts.mp4)', '',
      '## 独立镜像修复','',
      '原臂部指令[左0.4，右−0.2]应镜像为[左0.2，右−0.4]，旧代码却得到[左−0.4，右0.2]。实际臂部跟踪奖励由1降到0.2369，说明镜像样本的指令与姿态不一致。现已将81维观测中臂指令76/77交换后再取反。', '',
      '保留了修复前失败日志；修复后4项回归测试和64环境×5次更新/正式ONNX导出通过。本次101次更新运行使用修复前的冻结镜像代码，因此上表收益不能归功于这处修复。该错误在零臂指令时直接消失，是否通过训练影响走路偏航还需要独立对照。', '',
      '- [复现数据](command_symmetry_before.json) · [修复diff](arm_mirror_fix.patch) · [回归结果](unit_tests_after.log) · [修复后冒烟](mirror_smoke.log)', '',
      '## 后续顺序与交付状态','',
      '保留原发布ONNX，延迟扩展仅作为实验参数，不改默认训练范围。下一轮先使用修正镜像的81维配方建立一致对照，再进入头部gyro输入实验。若继续研究固定零延迟，应增加整回合固定时延分组，不能把本次随机下限扩展当作已经验证该方案。ERPO暂不移植。', '',
      f'- 本次候选：`{provenance["onnx"]}`，仅研究保存，未晋级。',
      '- 未控制硬件、未提交、未推送。以上证据只覆盖名义仿真。',
      '- [机器可读结果](summary.json) · [运行日志](train.log) · [文件哈希](artifact_manifest.json)', '']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    paths = [ROOT/'RESULTS.md', ROOT/'summary.json', ROOT/'configuration_comparison.json',
             ROOT/'arm_mirror_fix.patch',new_sym,RUN/'candidate.onnx',Path(provenance['final_checkpoint']),
             ROOT/'delay_matrix/matrix.json',ROOT/'delay_heatmap.png',ROOT/'walk_response.png',
             ROOT/'video_zero/rollouts.mp4',ROOT/'video_delay/rollouts.mp4']
    manifest = [{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in paths]
    (ROOT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('decision','metrics','matrix_aggregate','new_trials','new_trial_falls')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
