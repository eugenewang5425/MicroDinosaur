"""Summarize only the verified hardware matrix; preserve per-initial-state data."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260913_hardware_physics'
LABELS={'v7':'v7','calibration200':'CAD calibration 200','s288_v7recipe101':'S288 / v7 recipe 101'}
COLORS=['#2866a3','#85919e','#d46229']


def summarize(r):
    tasks={t:[x for x in r['trials'] if x['task']==t] for t in ('stand','forward','tail_up')}
    st,walk=tasks['stand'],tasks['forward']
    mean=lambda xs,key:float(np.mean([x[key] for x in xs]))
    return {'policy':r['policy'],'case':r['case']['name'],
        'stand_head_rms_deg_s':mean(st,'head_angular_speed_rms_deg_s'),
        'stand_yaw_abs_mean_deg':float(np.mean([abs(x['yaw_drift_deg']) for x in st])),
        'forward_vx_m_s':mean(walk,'vx_body_m_s'),
        'forward_lateral_abs_mean_mm':float(np.mean([abs(x['lateral_displacement_mm']) for x in walk])),
        'forward_yaw_abs_mean_deg':float(np.mean([abs(x['yaw_drift_deg']) for x in walk])),
        'forward_head_specific_force_peak_g':max(x['physics']['head_specific_force_peak_g'] for x in walk),
        'forward_joint_speed_peak_rad_s':max(x['physics']['joint_speed_max_rad_s'] for x in walk),
        'forward_positive_energy_mean_j':float(np.mean([x['physics']['positive_mechanical_energy_j'] for x in walk])),
        'forward_peak_positive_power_w':max(x['physics']['peak_positive_mechanical_power_w'] for x in walk),
        'forward_saturation_fraction':float(np.mean([x['physics']['dynamic_torque_saturation_fraction'] for x in walk])),
        'tail_pitch_mean_deg':mean(tasks['tail_up'],'tail_mean_deg'),
        'falls':sum(x['fell'] for x in r['trials'])}


def main():
    records=[]
    for subdir in ('reference_matrix_verified','candidate_matrix_verified'):
        records+=json.loads((OUT/subdir/'matrix.json').read_text())
    # Only the slope fall classifier changed; flat-ground clearance equals
    # world Z exactly. Preserve completed flat trials and replace all 54 slope
    # trials with reruns, comparing every policy under the same implementation.
    replacements=json.loads((OUT/'slope_clearance_verified/matrix.json').read_text())
    replace_map={(r['policy'],r['case']['name']):r for r in replacements}
    assert len(replace_map)==6
    records=[replace_map.get((r['policy'],r['case']['name']),r) for r in records]
    impl=hashlib.sha256((ROOT/'hardware_sim.py').read_bytes()).hexdigest()
    prior_impl=hashlib.sha256((OUT/'evaluation_source_v1/hardware_sim.py').read_bytes()).hexdigest()
    assert len(records)==87
    assert len({(r['policy'],r['case']['name']) for r in records})==87
    assert all(r['implementation_sha256'] in (impl,prior_impl) and len(r['trials'])==9 for r in records)
    for name in {r['case']['name'] for r in records}:
        assert len({r['implementation_sha256'] for r in records if r['case']['name']==name})==1
    assert len({r['plant_sha256'] for r in records})==1
    rows=[summarize(r) for r in records]
    (OUT/'final_matrix.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    lookup={(r['policy'],r['case']):r for r in rows}
    policies=list(LABELS)
    count=sum(len(r['trials']) for r in records)
    falls=sum(r['falls'] for r in rows)
    result={'verified_trials':count,'falls':falls,'policies':policies,'cases_per_policy':29,
        'initial_states_per_task':3,'measurement_seconds':{'stand':5,'forward':6,'tail_up':6},
        'warmup_seconds':2,'fixed_delay_ms':[10,20,20],
        'implementation_sha256':impl,'flat_implementation_sha256':prior_impl,'candidate_promoted':False,
        'slope_fall_classifier':'Ground-normal clearance; replaces 54 earlier world-Z-classified trials',
        'old_unverified_matrix_excluded':True,'case_means':rows}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    selected=['s288_protocol','curve_11v1','effort_80pct','combined_hypothesis']
    titles=['Protocol only','11.1 V curve*','80% effort*','Combined*']
    metrics=[('stand_head_rms_deg_s','Standing head speed (deg/s)'),
        ('forward_lateral_abs_mean_mm','Forward lateral drift (mm / 6 s)'),
        ('forward_vx_m_s','Forward speed (m/s; command 0.55)')]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    x=np.arange(len(selected))
    for ax,(key,title) in zip(axes,metrics):
        for i,policy in enumerate(policies):
            ax.bar(x+(i-1)*.25,[lookup[policy,c][key] for c in selected],width=.23,color=COLORS[i],label=LABELS[policy])
        ax.set_title(title,loc='left',fontsize=11);ax.set_xticks(x,titles,rotation=18,ha='right')
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('v07 CAD hardware sensitivity | 3 initial states | 10 / 20 / 20 ms delays',fontsize=14,x=.055,ha='left')
    fig.text(.055,.015,'* Unidentified motor/electrical hypotheses. All results use 5 ms physics; candidate not promoted.',fontsize=9,color='#50565f')
    fig.tight_layout(rect=(.015,.06,.995,.92));fig.savefig(OUT/'hardware_comparison.png',dpi=170);plt.close(fig)
    dts=[5,2.5,1.25,.625]
    dtcases=['s288_protocol','protocol_dt_2p5ms','protocol_dt_1p25ms','protocol_dt_0p625ms']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,(key,title) in zip(axes,metrics[:2]):
        for i,policy in enumerate(policies):
            ax.plot(dts,[lookup[policy,c][key] for c in dtcases],marker='o',color=COLORS[i],label=LABELS[policy])
        ax.set_xscale('log',base=2);ax.set_xticks(dts,[str(t) for t in dts]);ax.invert_xaxis()
        ax.set_xlabel('Physics timestep (ms)');ax.set_title(title,loc='left',fontsize=11);ax.grid(alpha=.18)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Timestep sensitivity | same policy period and physical delay in milliseconds',fontsize=13)
    fig.text(.06,.02,'Smaller timestep is a numerical check, not zero-delay optimization. Means across the same 3 initial states.',fontsize=9,color='#50565f')
    fig.tight_layout(rect=(.015,.06,.995,.92));fig.savefig(OUT/'timestep_sensitivity.png',dpi=170);plt.close(fig)
    lines=['# 配件约束、物理覆盖与训练结果，2026-09-13','',
        '**新候选未晋级。它在名义协议条件下前进更快，但站立头部抖动和直走横移明显退步；加入输出限幅可以改变表现，不能据此宣称模型已匹配真实S288。**','',
        f'正式统计为3个策略 × 29种条件 × 3种任务 × 3个初始状态，共{count}条短轨迹，跌倒判据触发{falls}次。每条先稳定2s，再记录站立5s、前进6s或竖尾6s；扰动初始状态使用固定seed1/2，seed0为HOME。跌倒判据是基座离当地地面的法向高度<.06m或世界倾角>60°，且不覆盖热衰退、长距离、完整碰撞和实物。','',
        '正式逐轨迹数据以 `final_matrix.json` 为准。原始 `reference_matrix/`、`candidate_matrix/` 中坡面旋转未生效；全部条件曾复跑到两个 `*_matrix_verified/`。之后发现平地世界Z跌倒判据会误报下坡，又以相对坡面的高度重跑54条坡面轨迹到 `slope_clearance_verified/` 并替换汇总中的旧条目。平地判据数值不变，物理动力学也未变；各条件的三个策略始终采用相同实现hash。排错/替换轨迹不额外计入783条。','',
        '## 同条件结果','',
        '以下均为S288协议量化、5ms物理步长、电机目标10ms/角位置20ms/角速度20ms、alpha=.9、名义请求Kp=7/Kd=.8。编码后的Kp/Kd略有舍入，未改变名义阻尼设计。数值为三个初始状态的均值，横移先取绝对值再平均。','',
        '| 策略 | 站立头速RMS °/s ↓ | 前进速度 m/s | 横移 mm/6s ↓ | 前进偏航绝对值 °/6s ↓ |','|---|---:|---:|---:|---:|']
    for p in policies:
        r=lookup[p,'s288_protocol'];lines.append(f"| {p} | {r['stand_head_rms_deg_s']:.2f} | {r['forward_vx_m_s']:.3f} | {r['forward_lateral_abs_mean_mm']:.1f} | {r['forward_yaw_abs_mean_deg']:.1f} |")
    lines+=['','前进指令都是.55m/s，三者均有欠跟踪；不能把速度更接近指令单独当晋级依据。','',
        '![硬件条件对照](hardware_comparison.png)','',
        '## 电机和数值敏感性','',
        '| 条件 | 策略 | 站立头速 °/s | 横移 mm/6s | 前进关节速度峰值 rad/s | 头部比力峰值 g |','|---|---|---:|---:|---:|---:|']
    for c in ['curve_11v1','combined_hypothesis','protocol_dt_1p25ms','protocol_dt_0p625ms']:
        for p in ['v7','s288_v7recipe101']:
            r=lookup[p,c];lines.append(f"| {c} | {p} | {r['stand_head_rms_deg_s']:.2f} | {r['forward_lateral_abs_mean_mm']:.1f} | {r['forward_joint_speed_peak_rad_s']:.2f} | {r['forward_head_specific_force_peak_g']:.2f} |")
    lines+=['','电压曲线、负载降额和组合条件都是未辨识假设。更小步长仍明显影响站立和横移，当前不能声明数值收敛；毫秒延迟始终不变，没有训练或优化零延迟。头部比力统计来自每个物理步，而非只看20ms策略采样点。','',
        '![步长敏感性](timestep_sensitivity.png)','',
        '## 本轮训练及后续修复的边界','',
        '- 从原v7 `model_15500.pt` 继续，4096环境、101次更新，每次24步，新增9,928,704条环境转移；公共计数372144→374568，最终 `model_15600.pt`。64×5先行冒烟、正常保存、官方ONNX导出和有限输出检查通过。',
        '- 采用当前v07质量/惯量，恢复v7保存的奖励参数（不含后加的lean_drift/contact_timing），保留修正的HOME和臂指令镜像，新增协议观测/指令/gain量化。学习器仍PPO、81维输入/19维输出、seed42；未加头部gyro输入。',
        '- 保存的奖励参数逐项与v7一致，但源码修复、机械来源与协议共同发生变化；这是工程候选，不是证明“量化单独造成变化”的单变量训练实验。11处可读YAML参数差异见 `training_parameter_diff.json`；Python类型/函数标签须结合源码快照审查。',
        '- 训练没有速度—力矩曲线或真实电源/热模型。摩擦随机化的旧BAM事件对S288无效；这个问题在训练后发现并修正，已完成候选不包含修复。修复另做8环境、15次局部重置和64×5冒烟，不能用冒烟代表步态改善。',
        '- 57项TensorBoard标量均有限。CPU物理回归4项、协议单元回归3项、已有奖励/镜像回归4项通过。滚动摩擦数值警告和步长敏感性保留为后续问题，不以无NaN代替动力学验收。','',
        '## 可核验产物','',
        '- [配件与物理模型说明](HARDWARE_MODEL.md)、[硬件/训练审计](hardware_audit.json)、[全部条件汇总](summary.json)。',
        '- [带协议执行的视频](video_protocol/rollouts.mp4)、[11.1V假设包络的视频](video_curve_11v1/rollouts.mp4)：站立、前进、转向、竖尾。渲染合同在相邻JSON中；静帧已检查，四段视频不是统计矩阵的替代。',
        '- 候选 `D:/microduck_rl/logs/rsl_rl/microdinosaur_v07_calibration/20260913_s288_v7recipe_4096x101/candidate.onnx`，对应 `candidate_contract.json` 和冻结源码快照。',
        '- 复现：工作目录为项目根，用 `D:/microduck_rl/.venv/Scripts/python.exe research/evaluate_hardware.py --plant research/20260913_handoff/native_v07 --policy 标签=ONNX路径 --out 一个新目录`。已有目录须显式 `--resume` 并通过模型/实现hash检查。','',
        '原发布ONNX和CAD源文件保持原样。下一步优先处理接触/数值时序与有效摩擦控制，再独立比较头部gyro输入；不要求解决零延迟，暂不移植ERPO。','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'verified_trials':count,'falls':falls,'nominal':[lookup[p,'s288_protocol'] for p in policies]},indent=2))


if __name__=='__main__':main()
