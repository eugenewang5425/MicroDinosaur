"""Substep acceptance: actual running flight or actual fallen-to-standing."""
from pathlib import Path
from dataclasses import asdict
import argparse,json
import numpy as np
import mujoco
import onnxruntime as ort
from hardware_sim import HardwareCase
from evaluate_run_jump import MotionExperiment,score,render
from imu_owned_head import configure_owned
from build_contact_motion_bank import (prepare_fall,constrain_recovery_head,
    FALL_DIRECTIONS,FALL_PUSH_N)
from run_heading_stable_start import CALIBRATOR
from heading_sim import WARMUP_SECONDS
from evaluate_policy import sha
from mjlab_microduck.tasks import recovery_bounds
import getup_acceptance_margin as margin

OUT=Path(__file__).parent/'20260914_contact_motion'

STANCE_JOINTS=('left_hip_pitch','left_knee','left_ankle','neck_pitch','head_yaw',
               'right_hip_pitch','right_knee','right_ankle','tail_pitch')


def _sole_tilt(model, data, geom_id):
    """脚底平面与地面的夹角(度)。用最低 20% 顶点拟合平面。"""
    mid=model.geom_dataid[geom_id]; a=model.mesh_vertadr[mid]; n=model.mesh_vertnum[mid]
    v=model.mesh_vert[a:a+n]@data.geom_xmat[geom_id].reshape(3,3).T+data.geom_xpos[geom_id]
    lo=v[v[:,2]<np.quantile(v[:,2],.2)]
    if len(lo)<3: return None
    _,_,vt=np.linalg.svd(lo-lo.mean(0)); nrm=vt[2]
    if nrm[2]<0: nrm=-nrm
    return float(np.degrees(np.arccos(np.clip(nrm[2],-1,1))))


def final_attitude_metrics(sim, qpos_frame):
    """末态姿态:逐关节偏差 / 脚底平贴 / 朝向保持。判据真源在 getup_acceptance_margin。

    为什么必须单独量:实测策略的终态是膝 −86°、颈 −87°、尾 −47°、脚底与地面夹角 30°、
    机身偏航 +130°,而十二条原判据(只看根高/倾角/速度/受力)对此**全部通过** ——
    因为蜷缩躯干的根高 116mm 与 v7 正确站姿 114mm 几乎相同。没有这三条就看不见姿势。
    """
    model=sim.model
    d=mujoco.MjData(model)
    d.qpos[:]=qpos_frame
    mujoco.mj_forward(model,d)
    home=np.asarray(sim.home,dtype=float)
    dev={}
    for n in STANCE_JOINTS:
        i=sim.names.index(n)
        dev[n]=float(np.rad2deg(qpos_frame[sim.jadr[i]]-home[i]))
    rot=d.xmat[sim.body].reshape(3,3)
    sole={}
    for side in ('left','right'):
        gid=model.geom('robot/'+side+'_foot_collision').id
        sole[side]=_sole_tilt(model,d,gid)
    return dict(stance_dev_deg=dev,
                stance_max_dev_deg=float(max(abs(v) for v in dev.values())),
                sole_tilt_deg=sole,
                sole_tilt_max_deg=float(max(v for v in sole.values() if v is not None)),
                final_yaw_deg=float(np.degrees(np.arctan2(rot[1,0],rot[0,0]))))


STANCE_MAX_DEV_DEG = 15.0
SOLE_TILT_MAX_DEG = 8.0
YAW_DRIFT_MAX_DEG = 30.0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--skill',choices=['run','getup'],required=True)
    p.add_argument('--policy',type=Path,required=True);p.add_argument('--out',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[921,922,923]);p.add_argument('--speeds',type=float,nargs='+',default=[.45,.6])
    p.add_argument('--delays',type=int,nargs='+',default=[10]);p.add_argument('--baseline-friction',action='store_true')
    p.add_argument('--torsional-friction',type=float,help='Explicit foot torsional friction in metres; default is selected_contact.json')
    p.add_argument('--video',action='store_true')
    # 倒地夹具:是否抵消推力的绕竖轴力矩。**默认关**(与 prepare_fall 同默认)。
    # 试过:开了之后 left 变成鼻子朝下落地被拒收、right 漂移 114.5°->157.7°,
    # 因为横向倒地的偏航不全是施加力矩造成的。留开关备查,不做默认。
    p.add_argument('--fall-yaw-neutral',action='store_true',help='抵消推力的竖轴力矩(试过,会让摔法变形)')
    p.add_argument('--no-sitfold',action='store_true',
                   help='back 案例不装坐撑示范(复现纯策略旧行为做对照)')
    a=p.parse_args()
    dest=OUT/a.out;dest.mkdir(parents=True,exist_ok=False);results=[]
    selected=json.loads((OUT/'selected_contact.json').read_text());plant=OUT/selected.get('plant_directory','plant')
    if a.baseline_friction and a.torsional_friction is not None:
        p.error('--baseline-friction and --torsional-friction are mutually exclusive')
    torsion=.01 if a.baseline_friction else (a.torsional_friction if a.torsional_friction is not None else selected['friction'][1])
    cases=[(f'v{v:g}',v,None) for v in a.speeds] if a.skill=='run' else [
        (k,0,v) for k,v in FALL_DIRECTIONS.items()]
    for label,speed,direction in cases:
      for delay in a.delays:
       for seed in a.seeds:
        key=f'{label}_c{delay}_s{seed}';result=dict(key=key,skill=a.skill,seed=seed,speed=speed,policy_sha256=sha(a.policy),plant_sha256=sha(plant/'nominal.mjb'))
        # selected_contact.json is a mutable selection file; record it so a result can
        # still be attributed after later study steps re-point plant_directory.
        result['selection_sha256']=sha(OUT/'selected_contact.json');result['plant_directory']=plant.name
        case=HardwareCase(motor_curve=True,voltage=12.,physics_dt=.00125,command_ms=delay,
            foot_torsional_friction_m=torsion)
        result['hardware_case']=asdict(case)
        e=configure_owned(MotionExperiment(a.policy if a.skill=='run' else CALIBRATOR,'run',speed,case,
            plant=plant,operating_envelope=True),(1,))
        if a.skill=='getup':
            # 与训练同源:getup 的头部偏航跟随躯干(head_imu_action.yaw_follows_trunk)
            e.yaw_follows_trunk=True
            constrain_recovery_head(e)
            previous_target=e.sim.transform_target
            # 坐撑示范(单一真源 recovery_bounds;与训练侧 SitFoldResidualAction 同构):
            # back 案例的目标 = 折腿时间表 + ±0.015rad·tanh(raw) 有界残差。残差从
            # target=home+raw·scale 精确恢复 raw,与训练侧的 actions 同源。
            # 训练侧示范环境绕过站定包络的理由(phi≈0.90 会把 ±86° 折腿夹没)在这里
            # 同样成立,所以示范路径不走 apply_band_numpy。
            # cases 解包是 (name, 0, 向量):方位名在 label,向量在 direction
            import os as _os
            _res=float(_os.environ.get('SITFOLD_RESIDUAL',recovery_bounds.SITFOLD_RESIDUAL_RAD))
            sitfold=e.sim.home is not None and label=='back' and not a.no_sitfold
            sit_legs=[i for i,n in enumerate(e.sim.names) if n.startswith(('left_','right_'))]
            result_sitfold={'enabled':bool(sitfold)}
            def bounded_recovery(target,obs):
                limits=e.sim.model.jnt_range[e.sim.jids]
                if sitfold:
                    raw=(target-e.sim.home)/e.sim.scale
                    t=max(0.,e.sim.data.time-e.getup_t0)
                    ref=recovery_bounds.sitfold_reference(np.array([t]),e.sitfold_g0,
                        e.sim.names,e.sim.home)[0]
                    # 与训练侧同门:残差只在站定后(t>5s)生效,折腿期纯脚本
                    if t>recovery_bounds.SITFOLD_RESIDUAL_AFTER_S:
                        ref[sit_legs]+=_res*np.tanh(raw[sit_legs])
                    ref=recovery_bounds.apply_numpy(ref,e.sim.names)
                    ref=np.clip(ref,limits[:,0]+.07,limits[:,1]-.07)
                    return previous_target(ref,obs)
                # Same operating envelope as training. The ankle/jaw bounds already
                # applied by MotionExperiment(operating_envelope=True) are re-applied
                # idempotently; the neck/battery clamp is what used to be missing, so
                # the evaluator accepted poses the CAD clearance gate rejects.
                target=recovery_bounds.apply_numpy(target,e.sim.names)
                target=np.clip(target,limits[:,0]+.07,limits[:,1]-.07)
                # 站定后的操作包络(与训练侧同源)。不加这一句就会重演今晚那个漂移:
                # 训练里关节被夹到 HOME±半宽、评估里没有,于是"测到的姿态"根本不是
                # 训练时的姿态 —— 我第一次测第⑬轮就是这样误判"包络没生效"的。
                rot=e.sim.data.xmat[e.sim.body].reshape(3,3)
                phi=recovery_bounds.phi_numpy(rot,e.sim.data.qpos[2])
                target=recovery_bounds.apply_band_numpy(target,e.sim.names,e.sim.home,phi)
                return previous_target(target,obs)
            # **注意:这里不装 bounded_recovery。** 倒地准备是测试夹具的动作,不该受
            # 被试策略的操作包络约束 —— 实测踩过:先装包络再 prepare_fall,于是
            # "把尾巴命令到 +70° 让路"被站位包络夹回 ±15°,尾巴又撑住,back 案例
            # 的非足承重从 53% 掉到 20%,被合格判定拒掉。包络在 rollout 前才装。
        try:
            if a.skill=='run':
                metrics,heading=e.run('stand','imu',seed,10.,True);score(e,metrics)
                result.update(metrics=metrics,passed=bool(metrics['running_with_flight'] and metrics['settled_final_second']))
            else:
                calibration=e.reset(seed)
                # 倒之前的站立朝向:世界系基准。与"落地朝向"是两个不同的基准,见下面 drift。
                _rs=e.sim.data.xmat[e.sim.body].reshape(3,3)
                result['start_yaw_deg']=float(np.degrees(np.arctan2(_rs[1,0],_rs[0,0])))
                pre,pre_frames=prepare_fall(e,direction,yaw_neutral=a.fall_yaw_neutral)
                _r0=e.sim.data.xmat[e.sim.body].reshape(3,3)
                result['fallen_yaw_deg']=float(np.degrees(np.arctan2(_r0[1,0],_r0[0,0])))
                # 合格判定收紧:除倾角/静止/非足支地/穿透外,新增"根高必须明显低于站姿"。
                # 旧判定只查倾角>25°,于是"站着后仰 32°、双脚全踩地、根高 138mm"这种
                # 根本没倒的状态也能过关(实测就是旧 front 案例混进四方位的原因)。
                # 合格判定的关键增量:**非足部位必须真的承担体重**,不是"倾角够大"就行。
                # 实测区分度:旧的假 front 案例(站着后仰32°、双脚全踩地)非足只 1.91N = 体重的 18%,
                # 而真倒地的四例是 53%(坐撑) / 68%(俯卧) / 100%(侧躺)。用 35% 作门。
                body_weight=float(np.sum(e.sim.model.body_mass)*9.81)
                nonfoot_frac=float(pre[-1,5])/max(1e-6,body_weight)
                qualified=bool(pre[-1,2]>25 and pre[-1,3]<.05 and pre[-1,4]<.5 and pre[-1,5]>1
                    and pre[-1,6]<.002 and nonfoot_frac>=.35)
                result['fallen_root_z_mm']=float(pre[-1,1])*1000
                result['yaw_neutral_fall']=a.fall_yaw_neutral
                result['nonfoot_weight_fraction']=nonfoot_frac
                result['body_weight_N']=body_weight
                result.update(start_tilt_deg=float(pre[-1,2]),fallen_start_valid=qualified,calibration=calibration,
                    start_class='lying_over_75deg' if pre[-1,2]>75 else 'body_supported_25_to_75deg')
                if not qualified:raise ValueError('Physical push did not produce a valid resting fallen start')
                e.rows=[];e.qpos_frames=[];start=e.sim.data.time-WARMUP_SECONDS
                e.getup_t0=e.sim.data.time
                # 三段式参考的起点:倒地态的真实关节角(与 try_sit_reference 的 g0 同源)
                e.sitfold_g0=e.sim.data.qpos[e.sim.jadr].copy()
                result['sitfold_script']=result_sitfold
                # 现在才装被试策略的操作包络(倒地准备已完成)
                e.sim.transform_target=bounded_recovery
                e.sim.session=ort.InferenceSession(str(a.policy),providers=['CPUExecutionProvider'])
                for tick in range(600):e.poll();e.sim.step(np.zeros(18))
                trace=np.asarray(e.rows);trace[:,0]-=start;e.rows=trace.tolist();tail=trace[trace[:,0]>=10]
                # 判据唯一真源在 getup_acceptance_margin;这里不再手写掩码,否则
                # "评估通过"与"margin 报告"会各自演化(neck 界就这样漂过一次)。
                good=margin.good_mask(tail)
                binding,binding_frac=margin.binding(margin.hold_frac(tail))
                result.update(passed=bool(good.mean()>=.95),standing_final_fraction=float(good.mean()),
                    binding_criterion=binding,binding_fraction=float(binding_frac),
                    hold_frac=margin.hold_frac(tail),
                    final_tilt_deg=float(tail[-1,8]),final_height_mm=float(tail[-1,1]*1000))
            trace=np.asarray(e.rows)
            result.update(status='COMPLETE',penetration_max_mm=float(trace[:,16].max()*1000),
                joint_limit_excess_deg=float(np.rad2deg(trace[:,13].max())),ankle_max_deg=float(np.rad2deg(trace[:,18].max())),
                final_speed_m_s=float(np.linalg.norm(trace[-800:,9:11],axis=1).mean()))
            result['passed'] &= bool(result['penetration_max_mm']<=2 and result['joint_limit_excess_deg']<=.5 and result['ankle_max_deg']<55)
            if a.skill=='getup':
                result.update(head_pitch_min_deg=float(np.rad2deg(trace[:,19].min())),head_pitch_max_deg=float(np.rad2deg(trace[:,19].max())))
                result['passed'] &= bool(result['head_pitch_min_deg']>=-2 and result['head_pitch_max_deg']<=29)
                # Head/battery CAD clearance, measured on the same 0.04 s pose grid the
                # CAD gate samples. The offending bodies are ancestor and descendant so
                # MuJoCo can never generate the contact; the neck envelope is the
                # enforceable proxy and this is where it is actually checked.
                frames=np.asarray(e.qpos_frames)
                neck=np.rad2deg(frames[:,e.sim.jadr[e.sim.names.index('neck_pitch')]])
                # 末态姿态三条(旧十二条看不见):站姿/脚底平贴/朝向保持
                att=final_attitude_metrics(e.sim,frames[-1])
                start_yaw=np.degrees(np.arctan2(np.sin(0.),np.cos(0.)))  # 见下:用落地帧
                rot0=np.array(e.sim.model.body_quat[0])
                drift=abs(((att['final_yaw_deg']-result.get('fallen_yaw_deg',0.)+180))%360-180)
                # 世界系基准:从"倒之前站着朝哪"到末态。用户看图说的"站着头朝后"是这一条。
                # 两个基准含义不同、数值也差很多(实测左 113.4° vs 41.2°、右 114.5° vs 147.3°),
                # 门目前仍然用落地基准(不动验收线),但两个都记,别让基准的选择藏在默认值里。
                world=abs(((att['final_yaw_deg']-result.get('start_yaw_deg',0.)+180))%360-180)
                result.update(**att, yaw_drift_deg=float(drift), yaw_drift_world_deg=float(world),
                    stance_limit_deg=STANCE_MAX_DEV_DEG, sole_limit_deg=SOLE_TILT_MAX_DEG,
                    yaw_limit_deg=YAW_DRIFT_MAX_DEG)
                result['passed'] &= bool(att['stance_max_dev_deg']<=STANCE_MAX_DEV_DEG
                                         and att['sole_tilt_max_deg']<=SOLE_TILT_MAX_DEG
                                         and drift<=YAW_DRIFT_MAX_DEG)
                limit=recovery_bounds.NECK_BATTERY_TERMINATE_DEG
                result.update(neck_pitch_min_deg=float(neck.min()),neck_pitch_max_deg=float(neck.max()),
                    neck_battery_limit_deg=limit,
                    geometry_status='Measured: neck_pitch within the CAD head/battery clearance bound'
                        if neck.max()<=limit else 'FAILED: neck_pitch exceeded the CAD head/battery clearance bound')
                result['passed'] &= bool(neck.max()<=limit)
            physical=e.sim.trace[-len(trace):]
            np.savez_compressed(dest/(key+'.npz'),physics=trace,qpos=np.asarray(e.qpos_frames),frames_dt=.04,
                actual_joint_velocity=np.array([r[0] for r in physical]),actual_joint_torque=np.array([r[1] for r in physical]))
            if a.video and seed==a.seeds[0] and delay==a.delays[0]:
                e.skill=a.skill;render(e,dest/(key+'.mp4'))
        except Exception as exc:result.update(status='REJECTED',error=repr(exc),passed=False)
        results.append(result);(dest/(key+'.json')).write_text(json.dumps(result,indent=2))
        (dest/'summary.json').write_text(json.dumps(results,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k!='metrics'},ensure_ascii=True),flush=True)
