"""Summarize measured contact trials and separate motion skill acceptance."""
from pathlib import Path
import json,re
import numpy as np
OUT=Path(__file__).parent/'20260914_contact_motion'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
groups=['scan','confirmation','timestep_confirmation','normal_contact_probe','normal_contact_dt_confirmation',
    'final_squat_delay_matrix','final_squat_low_voltage','normal005_original_torsion',
    'preferred_squat_delays','preferred_squat_low_voltage','preferred_squat_halfstep']
trials=[dict(group=group,**r) for group in groups for r in read(OUT/group/'summary.json')]
(OUT/'all_squat_trials.json').write_text(json.dumps(trials,indent=2))
table=[]
for name in ['baseline','slide_12','torsion_15mm','roll_01mm','roll_05mm','combined']:
    rows=[r for r in trials if r['group']=='scan' and r['case']['name']==name]
    paired=[r for r in rows if r['seed'] in (902,903)]
    table.append(dict(profile=name,passed=sum(r['passed'] for r in rows),total=len(rows),
        paired_highfreq_head_deg_s=float(np.mean([r['hold_head_gyro_above_5hz_rms_deg_s'] for r in paired])),
        paired_contact_p99_mm_s=float(np.mean([r['hold_loaded_contact_tangent_speed_p99_mm_s'] for r in paired]))))
new=[r for r in trials if r['group'] in ('preferred_squat_delays','preferred_squat_low_voltage')]
old=next(r for r in trials if r['group']=='confirmation' and r['seed']==912 and r['case']['name']=='baseline')
normal=next(r for r in trials if r['group']=='normal_contact_probe' and r['seed']==912)
fine=next(r for r in trials if r['group']=='normal_contact_dt_confirmation' and r['seed']==912)
original_torsion=next(r for r in trials if r['group']=='normal005_original_torsion' and r['seed']==912)
geometry=read(OUT/'final_geometry_gates.json')
summary=dict(squat_trials=len(trials),squat_passes=sum(r['passed'] for r in trials),friction_screen=table,
    final_squat=dict(passed=sum(r['passed'] for r in new),total=len(new),
        depth_range_mm=[min(r['actual_depth_mm'] for r in new),max(r['actual_depth_mm'] for r in new)],
        highfreq_head_range_deg_s=[min(r['hold_head_gyro_above_5hz_rms_deg_s'] for r in new),max(r['hold_head_gyro_above_5hz_rms_deg_s'] for r in new)]),
    fixed_seed_912=dict(old=old,normal_5ms_torsion_15mm=normal,normal_5ms_torsion_15mm_half_physics_step=fine,
        normal_5ms_original_torsion=original_torsion),
    measured_contact_parameters=False)
(OUT/'contact_summary.json').write_text(json.dumps(summary,indent=2))
lines=['# 摩擦、接触响应、独立蹲起与跑步／起身研究','',
'当前接触设置是待实物标定的仿真假设。CAD、质量分布、电池位置、S288 能力上限和生产 ONNX 未改。',
'','## 接触调整及证据','',
'最终首选保留原三项足底摩擦：滑动 1.0、扭转 0.010 m、滚动 geom 系数 0.000001 m（实际接触钳制到 0.00001 m），主要改动为法向接触响应 20 → 5 ms。0.015 m 扭转仅作为已完成 PPO 训练时的固定研究配方，记录在 training_contact.json。',
'原足底优先级高于地面，因此仅改地面摩擦不会生效。扭转／滚动项是扭矩与法向力之比，不能当作无量纲滑动系数。[MuJoCo 接触参数文档](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)。',
'','| 初筛配方 | 蹲起通过 | 相同种子头部 >5 Hz 角速度 RMS（°/s） | 相同种子接触点微滑 P99（mm/s） |',
'|---|---:|---:|---:|']
for r in table:lines.append(f"| {r['profile']} | {r['passed']}/{r['total']} | {r['paired_highfreq_head_deg_s']:.4f} | {r['paired_contact_p99_mm_s']:.4f} |")
lines+=['','这不是摩擦越大越稳定的关系：提高滑动摩擦和同时提高三项反而增加了蹲住时的抖动。扭转候选的平均改善较小，单个新种子也可能退步。法向响应改为 5 ms 后，保留原扭转的两个样本高频指标均约 0.01027°/s，额外增大扭转已无明确收益，因此最终回到原值。两个扭转设置的左右转向共 8 次均未跌倒；末端航向误差约 1.28–3.08°。',
'','随后发现默认 20 ms 法向接触响应在受推倒地时产生约 17–19 mm 瞬时穿透。把法向接触响应设为 5 ms，保留阻尼比 1、物理步长 1.25 ms 和控制周期 20 ms，另用 0.625 ms 复核；没有给地面增加弹性来制造跳跃。',
f"同种子 912、保留原扭转 0.010 m：蹲住阶段头部 >5 Hz 角速度 RMS 从 {old['hold_head_gyro_above_5hz_rms_deg_s']:.4f} 降至 {original_torsion['hold_head_gyro_above_5hz_rms_deg_s']:.4f} °/s。另一个扭转 0.015 m 配方在原步长／半步长为 {normal['hold_head_gyro_above_5hz_rms_deg_s']:.4f}/{fine['hold_head_gyro_above_5hz_rms_deg_s']:.4f} °/s；最终原扭转配方也另做半步长复核并通过。",
'','蹲姿的高频改善不能外推为跑动时的全面改善。同一个旧快走策略、相同种子 922：改接触后两档运动的相机偏航角速度 RMS 由 8.59/7.46 升到 11.74/10.69 °/s；相机朝向误差 RMS 为 0.48/0.54°。该变化同时减少足底穿透，仍需要控制器适应和实物辨识。',
'','## 独立蹲起','',
f"固定的参考轨迹＋已学腿部有界修正，通过 5/10/15 ms 指令延迟的 9 次及 11.1 V、40 ms 位置反馈的 2 次，共 {sum(r['passed'] for r in new)}/{len(new)}。实际下蹲 {summary['final_squat']['depth_range_mm'][0]:.2f}–{summary['final_squat']['depth_range_mm'][1]:.2f} mm。每次保持和站回均按双脚受力、身体不支地、速度及角速度验收。",
'仍使用上一轮独立蹲起残差 ONNX，输出必须经过参考轨迹适配器；本轮没有把它当成普通位置策略直接执行。',
f"最终 11 条延迟／电压轨迹加 1 条半步长轨迹的 {geometry['preferred_squat_cad']['sampled_motion_poses']} 个运动姿态完成 CAD 实体采样检查，新增交叠小于 0.001 mm³ 阈值。此处是约 25 mm 的浅蹲，不是照片中的完全折腿深蹲。",
'',f'![独立蹲起]({OUT.as_posix()}/squat_contact_optimized.png)','',
'## 起身准备中的几何问题','',
'倒地试验发现头部俯仰约 32° 时，S288 头俯仰舵机壳与颈部支架开始出现 CAD 实体交集；33.84° 的一个样本约 11.19 mm³。起身控制把头俯仰目标限制到 5–27°，物理关节限位不变。',
'全部 23 个候选重置姿态做了实体交集检查，剔除 3 个同一初态的头盘—颈支架轻微交集（约 0.0160 mm³），最终保留 20 个站立、倾倒中和躺倒状态。外力仅用于事先生成倒地初始状态，学习与验收起身过程没有外力或位姿重置。',
'采样 CAD 检查不等于全机连续自碰撞或实物配合验证。嘴部仍保持小开角；没有宣称所有自碰撞都已启用。',
'','## 训练和独立验收','']
for folder,label in [('run_final','跑步第一轮'),('run_sole_final','脚底净空强化'),('getup_dense_final','起身连续姿态反馈'),('getup_preferred_contact','起身转回首选扭转配方')]:
    path=OUT/folder/'summary.json'
    if not path.exists():lines.append(f'- {label}：训练或验收进行中。');continue
    rows=read(path);done=[r for r in rows if r['status']=='COMPLETE']
    lines.append(f"- {label}：验收 {sum(r['passed'] for r in rows)}/{len(rows)}，完整执行 {len(done)} 次。")
    if folder.startswith('run') and done:
        for speed in sorted({r['speed'] for r in done}):
            group=[r for r in done if r['speed']==speed]
            lines.append(f"  - 请求 {speed:.2f} m/s：平均实速 {np.mean([r['metrics']['moving_speed_m_s'] for r in group]):.3f} m/s，合格连续腾空 {sum(bool(r['metrics']['qualified_flight_segments']) for r in group)}/{len(group)}。")
    elif done:
        for kind in sorted({r.get('start_class','unknown') for r in done}):
            group=[r for r in rows if r.get('start_class')==kind]
            lines.append(f"  - {kind}：{sum(r['passed'] for r in group)}/{len(group)}。")
lines+=['','起身早期姿态奖励在倾斜超过 90° 时存在零反馈区。独立中期验收未站起，因此在第 162 条已打印更新后停止该配方，并从已保存的第 151 次更新检查点另建连续反馈分支。没有把中断的 401 次计划写成已完成 401 次。',
'跑步两支完成 301＋201 次更新；起身完成 162 条更新后停止旧配方，再完成 201 次连续反馈更新。每次 512 个环境、24 步，共 10,629,120 次已打印正式环境转移；不含短训。停止分支最后 11 次更新未进入后续检查点。所有正式 PPO 本轮固定在扭转 0.015 m、法向响应 5 ms 的配方，不能写成已在最终 0.010 m 配方重训。',
'跑步第一轮依然属于快走；一个 0.6 m/s 请求样本的整只双脚共同净空最高只有 0.36 mm、无支撑段最长约 21 ms。后续强化奖励只在实际无支撑并且整只双脚净空超过 0.2 mm 时开始增加，正式验收仍需 >2 mm 持续 ≥40 ms、起飞质心速度、前进与停稳条件。',
'脚底净空强化后最高档三次平均约 0.46 m/s，仍未出现合格双脚腾空，9 次均未通过停稳；不能称为学会跑步。两个跑步分支分别 62、63 个采样运动姿态没有新增超过阈值的 CAD 实体交叠。',
'起身最终 8＋4 次完整验收共 0/12。三个前倾约 32°、身体支撑的初态能扶正到约 3°、身体高度恢复且身体不再支地，但末尾 2 秒仍以约 0.058–0.064 m/s 交替踏步，双脚同时承重仅约 0.5% 时间，不能算稳定站起；这也不是从完全躺倒起身。详见 getup_failure_analysis.json。',
f"**起身还存在实际几何失败：最终主矩阵 {geometry['getup_dense_final_cad']['sampled_motion_poses']} 个采样运动姿态中，{geometry['getup_dense_final_cad']['failed_pose_count']} 个产生新的实体交叠。最大为头部横滚舵机壳与电池包络约 {geometry['getup_dense_final_cad']['max_increase_mm3']:.1f} mm³。** 电池 ENVELOPE 是选定配件的空间占用近似，不能因为不是外观实体就允许穿过。当前模型仅有身体对地代理，尚无覆盖这些接触的全机自碰撞。头俯仰局部限幅不足以保证整条头颈链安全，起身候选应判为几何无效诊断结果。",
'首选接触配方的额外 4 次起身已经行为失败，未另做完整 CAD 审计；不得据此宣称几何通过。嘴／上颚全程自碰撞也仍未解决。',
'','## 下一轮顺序','',
'1. 保留首选接触假设和已通过的浅蹲，不把所有摩擦一起增大。实物用同一脚垫、同一地面做拖动／扭转及落地响应辨识，再替换假设值。',
'2. 起身先补头颈—电池及其他已发现部位的物理自碰撞，固定经过核验的初态；先练前倾支撑→双脚停稳，再推进侧躺和仰躺。不能继续只靠角度限幅筛除碰撞。',
'3. 跑步保留独立分支，先处理当前踏步不停和实际脚底净空不足，再扩大速度；不以没有脚底净空的高速快走代替跑步验收。',
'本轮没有替换生产基础步态、改动 CAD 或移动电池。',
'','## 复现入口','',
'- `selected_contact.json`：研究配方及当前物理模型入口。',
'- `gpu_cpu_plant_audit.json`：CPU 与 GPU 编译模型质量、惯量、驱动和接触数组一致。',
f'- `contact_summary.json` / `all_squat_trials.json`：{len(trials)} 次蹲姿对照，所有启动拒绝保留在分母。',
'- `train_contact_motion.py`、`contact_motion_cfg.py`、`evaluate_contact_motion.py`：独立专项训练与每个物理子步的验收。',
'- `run_provenance.json`、`source_snapshot` 和最终清单：每个检查点、源文件及训练状态的证据。',
'','接触、关节被动参数及 S288 主动扭矩—速度曲线尚未实测。当前仿真结果不是硬件极限，也不是实机动作放行。']
(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(summary['final_squat'])
