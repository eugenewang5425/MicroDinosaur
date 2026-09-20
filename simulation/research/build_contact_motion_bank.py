"""Measured reset states; external pushes occur only before get-up episodes."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json
import numpy as np
import mujoco
from hardware_sim import HardwareCase
from head_attitude_sim import HeadExperiment
from head_attitude import HeadConfig
from imu_owned_head import configure_owned
from run_heading_stable_start import CALIBRATOR
from imu_heading import matrix
from evaluate_policy import sha

OUT=Path(__file__).parent/'20260914_contact_motion'

def constrain_recovery_head(e):
    # Operating envelope from solid CAD checks; physical joint limits unchanged.
    e.kinematics.limits=e.kinematics.limits.copy()
    # 与训练侧 RecoveryAction 同源且同步:操作界要留在验收界(head_pitch<=29°)之内,
    # 否则实测顶到 29.04° 每个 checkpoint 都贴边。见 contact_motion_cfg.RecoveryAction。
    e.kinematics.limits[1]=[np.deg2rad(5)-.035,np.deg2rad(26.5)]
    return e

def experiment(seed,head_envelope=False):
    selected=json.loads((OUT/'selected_contact.json').read_text())
    e=configure_owned(HeadExperiment(OUT/selected.get('plant_directory','plant'),CALIBRATOR,
        HardwareCase(motor_curve=True,voltage=12.,physics_dt=.00125),
        head_config=HeadConfig(max_measurement_age_s=.04)),(1,))
    s=e.sim
    if head_envelope:constrain_recovery_head(e)
    def target(goal,obs):
        goal=goal.copy();goal[s.names.index('jaw_hinge')]=.04
        return e.transform_target(goal,obs)
    s.transform_target=target
    calibration=e.reset(seed)
    return e,calibration

def snapshot(e,calibration,label,seed):
    s=e.sim;body=s.data.site_xmat[s.model.site('robot/imu').id].reshape(3,3)
    _,sensor=e.stream.latest
    return dict(label=label,seed=seed,root_qpos=s.data.qpos[:7].tolist(),root_qvel=s.data.qvel[:6].tolist(),
        joint_pos=s.data.qpos[s.jadr].tolist(),joint_vel=s.data.qvel[s.vadr].tolist(),
        raw_action=s.raw.tolist(),applied=s.applied.tolist(),head_rotation=matrix(e.head_est.q).tolist(),
        head_accel=e.head_est.accel_filtered.tolist(),head_bias=e.head_est.bias.tolist(),
        alignment=e.alignment_rotation.tolist(),packet=np.r_[sensor[2],sensor[3],sensor[1]].tolist(),
        heading_world_zero=float(np.arctan2(body[1,0],body[0,0])),calibration=calibration)

# 落地方向表(**唯一真源**)。2026-09-16 实测校正:
#   旧表 front=[-1,0,0]/back=[1,0,0] 是反的 —— front 物理上是"往后推",而它被尾巴顶住从不倒地,
#   于是那个"四方位"案例其实不是倒地。left/right 的向量本来是对的(推哪边倒哪边)。
#   并实测:后推要**让尾巴让路**(命令 tail_pitch 抬起)才能坐到屁股着地,否则尾巴当支架撑成后仰32°。
FALL_DIRECTIONS={'left':[0,1,0],'right':[0,-1,0],'front':[1,0,0],'back':[-1,0,0]}
FALL_TAIL_RELAX_DEG={'back':70.}
FALL_PUSH_N={'left':5.,'right':5.,'front':4.,'back':3.}
FALL_RAMP_S=2.5
FALL_HOLD_S=2.0
FALL_SETTLE_S=2.5
FALL_TAIL_ACTIVATE_S=5.


def _fall_target(e,jads,names,tail_deg):
    """后推时把尾巴命令到让路角度(否则它当防后倒支架),其余关节保持标定站姿。"""
    home=np.asarray(e.sim.home,dtype=float)
    if tail_deg is None:return None
    g=home.copy();g[names.index('tail_pitch')]=np.deg2rad(tail_deg)
    return g


def prepare_fall(e,direction,force=None,capture_tilts=False,ramp_s=None,hold_s=None,
                 settle_s=None,tail_relax_deg=None,push_body=None,yaw_neutral=False):
    """准静态倾斜诱导倒地。

    2026-09-16 两处修正(实测驱动):
      1) "推一把(0.8s 冲击)"产生的是混乱翻滚 —— 八个组合里七个落侧面,而且前推与左推落在同一侧。
         改成"力在 ramp_s 内线性升到峰值、保持 hold_s、放手、静置 settle_s"后才方向可控:
         前推 4N -> 俯卧(腹朝下)、左推 5N -> 左侧、右推 5N -> 右侧。
      2) 后推必须让尾巴让路(tail_relax_deg),否则尾巴承担 2.8N 把机器人撑成后仰 32° 的"坐撑"
         而不是倒地;让路后屁股着地(躯干最低 0.0mm)、尾巴受力 0.0N。
    """
    s=e.sim;d=s.data;m=s.model;initial=s.applied.copy();start=d.time
    # 施加点默认仍是嘴尖(s.head),但那是**最坏**的施加点:jaw_soft 的 CoM 在整机
    # 质心前方 +101.8mm,横向力 F 于是附带 τz=0.1018·F 的持续偏航力矩(5N -> 0.51 N·m,
    # 挂 4.5 秒)。实测后果:四个方位的落地朝向完全不对称 —— 左 +154.6°、右 −32.8°、
    # 前 −3.9°、后 −0.2°;而验收的"朝向保持"是拿**落地朝向**当基准的,于是这条判据
    # 一半在量测试夹具自己拧了多少,而不是被试会不会转身。push_body 用来换成
    # 躯干质心附近的施加点(robt/trunk_base 的 x 偏移 −15.9mm,τz/F 降 6.4 倍)。
    #   yaw_neutral(**默认关,试过且失败,别开**):试图把推力的绕竖轴力矩逐个 tick
    #   抵消掉。动机是实测 jaw_soft 的 CoM 在整机质心 +101.8mm,5N 横向力附带 0.509 N·m
    #   的 τz 挂 4.5 秒,落地朝向因此左 +154.6°/右 −32.8°/前 −3.9°/后 −0.2°,四个方位
    #   基准朝向不统一。但实测抵消的后果是**摔法变了**:left 变成鼻子朝下落地
    #   (evaluator 直接报 'Camera optical axis is vertical; yaw is undefined' 拒收),
    #   right 的漂移从 114.5° 涨到 157.7°。结论:横向倒地时的偏航不全是施加力矩造成的,
    #   腿的蹬地也在拧,只掐施加力矩等于换了一种摔法而不是消掉了拧。保留开关备查。
    if push_body is None:
        push_body=s.head
    elif isinstance(push_body,str):
        push_body=m.body(push_body).id
    # direction 可以是方位名或推力图向量;两者都归一到一个名字再查表
    if isinstance(direction,str):
        dname=direction
    else:
        vec=np.asarray(direction,dtype=float)
        dname=next((k for k,v in FALL_DIRECTIONS.items() if np.allclose(v,vec)),None)
    force=FALL_PUSH_N.get(dname,3.) if force is None else force
    ramp_s=FALL_RAMP_S if ramp_s is None else ramp_s
    hold_s=FALL_HOLD_S if hold_s is None else hold_s
    settle_s=FALL_SETTLE_S if settle_s is None else settle_s
    tail_relax_deg=FALL_TAIL_RELAX_DEG.get(dname,None) if tail_relax_deg is None else tail_relax_deg
    class Hold:
        def get_inputs(self):return [SimpleNamespace(name='obs')]
        def run(self,*args):return [((initial-s.home)/s.scale).astype(np.float32)[None]]
    s.session=Hold()
    prev_target=getattr(s,'transform_target',None)
    goal=_fall_target(e,None,s.names,tail_relax_deg)
    if goal is not None and prev_target is not None:
        limits=s.model.jnt_range[s.jids]
        def scripted(target,obs,_g=goal,_p=prev_target,_l=limits):
            gg=np.clip(_g,_l[:,0]+.07,_l[:,1]-.07)
            return _p(gg,obs)
        s.transform_target=scripted
    try:
      rows=[];frames=[];push=np.asarray(FALL_DIRECTIONS[dname] if dname else direction,dtype=float)*force
      n_ramp=round(ramp_s/s.dt);n_hold=round(hold_s/s.dt);n_total=n_ramp+n_hold+round(settle_s/s.dt)
      e.lean_snapshots=[];thresholds=[20,35,50]
      for tick in range(n_total):
        t=tick*s.dt
        w=min(1.,t/ramp_s) if tick<n_ramp+n_hold else 0.
        d.xfrc_applied[:]=0
        d.xfrc_applied[push_body,:3]=push*w
        if yaw_neutral:
            # xfrc_applied 的力作用在 body 的 CoM(xipos)、力矩也在世界系;要抵消的是
            # 相对**整机**质心的竖轴力矩 (r×F)_z, r = xipos - subtree_com[0]。
            r=d.xipos[push_body]-d.subtree_com[0]
            d.xfrc_applied[push_body,5]=-(r[0]*push[1]*w-r[1]*push[0]*w)
        e.poll();s.step(np.zeros(18))
        rot=d.xmat[s.body].reshape(3,3)
        penalty=0.;nonfoot=0.;w=np.zeros(6)
        floor=m.geom('terrain').id
        for i,c in enumerate(d.contact):
            if floor in (c.geom1,c.geom2):
                penalty=max(penalty,-c.dist)
                other=c.geom2 if c.geom1==floor else c.geom1
                mujoco.mj_contactForce(m,d,i,w)
                if other not in s.foot_geoms:nonfoot+=max(0,w[0])
        rows.append([d.time-start,d.qpos[2],np.rad2deg(np.arccos(np.clip(rot[2,2],-1,1))),
            np.linalg.norm(d.qvel[:3]),np.linalg.norm(d.qvel[3:6]),nonfoot,penalty])
        if capture_tilts and thresholds and rows[-1][2]>=thresholds[0]:
            threshold=thresholds.pop(0)
            e.lean_snapshots.append(snapshot(e,{},f'lean_{threshold}',0))
        if tick%2==1:frames.append(d.qpos.copy())
    finally:
        d.xfrc_applied[:]=0
        if goal is not None and prev_target is not None:s.transform_target=prev_target
    return np.array(rows),np.array(frames)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['standing','getup'],default='standing');a=p.parse_args()
    selected=json.loads((OUT/'selected_contact.json').read_text());suffix=selected.get('bank_suffix','')
    target=OUT/(a.kind+'_reset_bank'+suffix+'.json');assert not target.exists()
    records=[];audit=[]
    for seed in (1,2,3):
        e,cal=experiment(seed);records.append(snapshot(e,cal,'standing',seed))
    if a.kind=='getup':
        dest=OUT/('fall_preparation'+suffix);dest.mkdir(exist_ok=False)
        for name,direction in [('left',[0,1,0]),('right',[0,-1,0]),('front',[-1,0,0]),('back',[1,0,0])]:
            e,cal=experiment(2)
            rows,frames=prepare_fall(e,direction)
            row=dict(label=name,final_tilt=float(rows[-1,2]),final_speed=float(rows[-1,3]),
                final_omega=float(rows[-1,4]),nonfoot_N=float(rows[-1,5]),penetration_mm=float(rows[:,6].max()*1000),
                external_force_N=3.,external_force_duration_s=.8,learning_episode_external_force=False)
            row['accepted']=bool(rows[-1,2]>55 and rows[-1,3]<.05 and rows[-1,4]<.5 and rows[-1,5]>1 and rows[-1,6]<.002)
            np.savez_compressed(dest/(name+'.npz'),trace=rows,qpos=frames,frames_dt=.04)
            if row['accepted']:
                record=snapshot(e,cal,name,2)
                # Equal sampling of four fallen starts plus upright retention.
                records.extend([record.copy() for _ in range(3)])
            audit.append(row);print(json.dumps(row),flush=True)
        (dest/'audit.json').write_text(json.dumps(audit,indent=2))
        assert any(r['accepted'] for r in audit),'No physically resting fallen start; do not train invalid resets'
    result=dict(source_policy=str(CALIBRATOR),source_sha256=sha(CALIBRATOR),calibration_seconds=6.,
        sample_count=50,plant_sha256=sha(OUT/selected.get('plant_directory','plant')/'nominal.mjb'),action_names=e.sim.names,records=records,
        contract='Measured nominal 12V states; initial-state distribution only; no force or pose writes inside learned episodes',
        self_collision_status='No whole-machine self-collision release; sampled candidate CAD verification required')
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(target,len(records),flush=True)
