"""Consolidated final report, including failed and post-hoc experiments."""
import json
import numpy as np
from corrective_common import OUT


def read(path):return json.loads((OUT/path).read_text())


def main():
    summary=read('summary.json');development={}
    for label in ('before','fitted','rehearsal'):
        assert read(f'offline_probe/evaluation/{label}/complete.json')['trials']==13
        rows=read(f'offline_probe/evaluation/{label}/matrix.json')
        stairs=[r for r in rows if r['case']['terrain']=='steps_10'];stops=[r for r in rows if r['case']['program']=='resume']
        development[label]=dict(trials=len(rows),stairs_passed=sum(r['metrics']['terrain_traversed'] for r in stairs),
            falls=sum(r['metrics']['fell'] for r in rows),
            stair_by_delay={str(d):sum(r['metrics']['terrain_traversed'] for r in stairs if r['case']['delay']==d) for d in (5,10,15)},
            complete_stop_passes=sum(not r['metrics']['fell'] and abs(r['metrics']['actual_drop_mm']-40)<=6 and abs(r['metrics']['return_height_error_mm'])<=5
                and r['metrics']['post_stop']['planar_speed_mean_mm_s']<=10 and r['metrics']['braking_net_displacement_mm']<=80 for r in stops),
            stop_details=[dict(delay=r['case']['delay'],fell=r['metrics']['fell'],drop_mm=r['metrics']['actual_drop_mm'],return_mm=r['metrics']['return_height_error_mm'],
                residual_speed_mm_s=r['metrics']['post_stop']['planar_speed_mean_mm_s'],limit_rad=r['metrics']['joint_limit_violation_max_rad']) for r in stops],
            flat_speed=next(r['metrics']['body_vx_mean_m_s'] for r in rows if r['case']['terrain']=='flat' and r['case']['scenario']=='straight'),
            limit_max_rad=max(r['metrics']['joint_limit_violation_max_rad'] for r in rows))
    fresh_rows=read('offline_probe/evaluation/rehearsal_fresh/matrix.json')
    assert read('offline_probe/evaluation/rehearsal_fresh/complete.json')['trials']==13
    valid=[r for r in fresh_rows if r['metrics'] is not None]
    fresh=dict(attempted=13,executed=len(valid),calibration_failures=13-len(valid),
        stairs_passed=sum(r['metrics']['terrain_traversed'] for r in valid if r['case']['terrain']=='steps_10'),
        stairs_planned=9,falls=sum(r['metrics']['fell'] for r in valid),
        stop_details=[dict(delay=r['case']['delay'],fell=r['metrics']['fell'],drop_mm=r['metrics']['actual_drop_mm'],
            residual_speed_mm_s=r['metrics']['post_stop']['planar_speed_mean_mm_s']) for r in valid if r['case']['program']=='resume'])
    rough={}
    for label in ('source','control','corrective'):
        rows=read(f'rough_probe/{label}/matrix.json');rough[label]={}
        for terrain in ('roughgrid_8mm','slope_3','slope_-3'):
            selected=[r['metrics'] for r in rows if r['case']['terrain']==terrain]
            rough[label][terrain]=dict(passed=sum(r['terrain_traversed'] for r in selected),
                falls=sum(r['fell'] for r in selected),vx=float(np.mean([r['body_vx_mean_m_s'] for r in selected])),
                max_limit_rad=max(r['joint_limit_violation_max_rad'] for r in selected))
    result=dict(status='COMPLETE',paired=summary,posthoc_development=development,fresh_rehearsal_confirmation=fresh,rough_readiness=rough,
        formal_new_trials=117,posthoc_development_trials=39,supplemental_stopped=read('evaluation/fitted/stopped.json'),
        collection_trials=27,rough_trials=27,fresh_supplemental_attempts=13,ppo_training_transitions=4939776,offline_optimizer_steps=2000,
        production_replaced=False,rough_training_started=False)
    (OUT/'final_summary.json').write_text(json.dumps(result,indent=2))
    lines=['# 纠正示范推进结果：越阶出现进展，完整能力仍未通过', '',
        '完成两组各512环境×201次PPO更新，共4,939,776条训练转移；两次成功短训另计。新采集27条接管轨迹，其中2500帧卡阶纠正、1050帧制动停稳进入训练。另从已有轨迹提取4050帧蹲起回放，没有新增这部分物理采集。',
        '', '## 固定配方对照', '',
        '两组从同一model_16250出发，seed47、实际LR3e-5、相同物理/奖励/课程和监督预算。每轮4×1536监督样本，其中512历史样本（固定128站立、384行走）、256停止纠正相同；control用768成功台阶示范，corrective用256成功示范＋512卡阶纠正。',
        '', '|指标|原策略|成功示范对照|卡阶纠正组|','|---|---:|---:|---:|']
    for title,field,fmt in [('平地实速，指令0.55m/s','flat_vx','.3f'),('平地相机朝向RMS，°','flat_yaw','.2f'),('平地相机俯仰RMS，°','flat_pitch','.2f')]:
        lines.append('|'+title+'|'+'|'.join(format(summary[a][field],fmt) for a in ('source','control','corrective'))+'|')
    for title,field,total in [('独立低速台阶穿越','holdout_traversals',9),('停止后实际速度达标','stop_passes',9),('蹲起深度与回升达标','crouch_passes',21)]:
        lines.append('|'+title+'|'+'|'.join(f"{summary[a][field]}/{total}" for a in ('source','control','corrective'))+'|')
    lines += ['', '新模型各54条确认均未跌倒。独立低速台阶使用未参与训练的初态201–203，分别覆盖5/10/15ms；原45条件保留初态1–3。旧示范库包含1–3，不能称所有回归初态完全未见过。两个新模型均未通过全部门槛，未晋级。',
        '', '对照组5/10ms停止后的实际速度约43–44/32–33mm/s；纠正组约34–36/18–25mm/s，均超10mm/s。纠正组相机俯仰2.27°也超2°门槛。两组最大实际关节越限分别约0.0383/0.0479rad，超0.02rad容差。[完整逐条件结果](TABLES.md)',
        '', '## 追加的示范拟合与旧动作回放试验', '',
        '配对训练后，纠正组在训练数据上的腿部目标RMSE仍约0.094rad（5.4°）；对照组反而约0.088rad。加入纠正数据并没有保证这些动作已被充分拟合。[训练目标拟合诊断](training_label_fit.json)',
        '', '随后在固定模型上做1000次纯示范Adam更新，暂停PPO并冻结归一化，学习率1e-4。恢复示范原始动作MSE从0.00651降至0.000114。此试验同时改变更新方式、优化器、学习率与归一化更新，不能把全部差异归因于某一个PPO奖励项。',
        '', '|相同开发初态61–63上的诊断|低速越阶|整段蹲起走停通过|13条中跌倒|平地实速m/s|','|---|---:|---:|---:|---:|']
    for label,title in [('before','拟合前纠正组'),('fitted','纯示范拟合后'),('rehearsal','补全蹲起回放后')]:
        r=development[label];lines.append(f"|{title}|{r['stairs_passed']}/9|{r['complete_stop_passes']}/3|{r['falls']}|{r['flat_speed']:.3f}|")
    lines += ['', '第一轮纯拟合使同组开发案例从3/9提高到7/9，但15ms下两次台阶跌倒，蹲起走停又有两次跌倒。已启动的补充确认因此提前停止，保留18条完成记录及停止原因，不将其计为完整确认。[停止记录](evaluation/fitted/stopped.json)',
        '', '排查发现，第一轮四个示范库都只有零身体高度指令，缺少完整蹲起回放。利用已有采集记录中学生执行的2–11秒蹲起/回升片段，重验动作标签和姿态，补入4050帧；再做1000次纯示范更新，每批新增256个姿态样本，总批量1792。此阶段新增的是已有轨迹回放，未修改物理或重新采集。[姿态数据验证](posture_dataset.json)',
        '', '这些追加结果属于事后开发诊断，初态61–63已用于前后比较，不是新的盲测结论；不与正式初态201–203直接混算。离线诊断检查点保留源critic和环境计数、清空旧PPO优化器动量，不能描述成新增PPO迭代；如续训，需明确重新建立优化器和检验critic。',
        '', f"补全姿态回放后另用初态401–403复核：13次计划中实际启动{fresh['executed']}次，{fresh['calibration_failures']}次在候选执行前因静止IMU校准角运动超限而失败；9个台阶计划中通过{fresh['stairs_passed']}个，已启动试验中跌倒{fresh['falls']}次。校准失败单独保留，既不算策略跌倒，也不排除后声称全部通过；未放宽校准阈值。[新初态记录](offline_probe/evaluation/rehearsal_fresh/matrix.json)",
        '', '## 凹凸地面与下一阶段', '',
        '|2–8mm不规则地面，指令0.2m/s|通过|实际平均速度m/s|','|---|---:|---:|']
    for label,title in [('source','原策略'),('control','成功示范对照'),('corrective','卡阶纠正组')]:
        r=rough[label]['roughgrid_8mm'];lines.append(f"|{title}|{r['passed']}/3|{r['vx']:.3f}|")
    lines += ['', '该地形检查以及±3°坡面使用初态211–213、名义10ms，三个模型共27条，无地形专用训练。它说明已有动作对部分不平地面出现迁移，不能替代多延迟、多地形训练验收。',
        '', '后续课程已准备并通过753个格子的物理射线高度核对：50%平地、20%上阶、10%下阶、20%小幅起伏；起伏从0–2mm逐步到0–6mm。当前完整越阶/走停/限位门槛未通过，因此未启动该课程正式训练。[数据与地形准备](DATA_AND_NEXT_STAGE.md)',
        '', '下一步应保留完整蹲起、走停和台阶回放，明确示范拟合与RL更新的先后和强度，从最新模型在5/15ms下实际访问的失败状态继续采集纠正，而非只重复旧学生状态。先复查关节越限、停止残余运动及静止校准启动。不要再仅靠延长同一弱监督配方或不断加难地形。跑跳、完全折腿及自由起身继续等待相应机械和接触模型条件。',
        '', '## 验证与交付', '',
        '实际学习率、4020/4824次PPO/含监督优化器增量、归一化官方导出和所有检查点有限性通过；数据动作标签精确复算，原始动作历史连续。两项配额/配置测试通过。两次64×5短训成功后才启动正式训练。初版停止数据检查忽略减速斜坡及航向限速而失败，已按原运行时代码修正，未重写轨迹；失败短训记录保留。',
        '', '首次离线导出比较在CPU/CUDA混合精度下断言失败，随后直接对已保存模型做CPU FP32核对，405条观测最大误差4.17e-7；没有重复训练或放宽2e-5容差。补姿态回放后官方导出也通过。数值保护保持，正式两组没有非有限数中断；旧MuJoCo Warp足部滚动摩擦警告仍未修复，不把本次无中断等同于根因已解决。',
        '', '保持v07、19×S288、1.0982148kg、正延迟/正阻尼、81→19及头部IMU职责。未加入视觉/gyro策略输入、未移植ERPO。实际舵机行程、全自碰撞、柔性和电热/电池参数仍有待实测。',
        '', '- [正式三组标记视频：原策略、成功示范对照、纠正组](videos/comparison.mp4)',
        '- [纯示范拟合后的10ms成功案例](videos/fitted/simulation.mp4)',
        '- [同一诊断初态15ms跌倒案例](videos/fitted_15ms/simulation.mp4)',
        '- [补全蹲起回放后的10ms案例](videos/rehearsal/simulation.mp4)',
        '- [补全回放后的5ms案例](videos/rehearsal_5ms/simulation.mp4)',
        '- [全部结构化结果](final_summary.json)',
        '- [完整哈希与源码/模型归档](artifact_manifest.json)',
        '', '生产microdinosaur_p2.onnx未替换，SHA256仍为0804114efd1e457ec2c297d709c0464fd0c2bfe251cbb36db7de74543857b0e3。没有推送或操作实物。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(result['posthoc_development'],indent=2))


if __name__=='__main__':main()
