"""Slow foot-supported fold and stand return through the current S288 pipeline."""
from pathlib import Path
import argparse, json, sys, time
from dataclasses import asdict
import mujoco
import numpy as np
from hardware_sim import HardwareCase
from head_attitude_sim import HeadExperiment
from head_attitude import HeadConfig
from imu_owned_head import configure_owned
from run_heading_stable_start import CALIBRATOR
from heading_sim import WARMUP_SECONDS
from evaluate_policy import sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_compact_fold'
COLUMNS=['time','root_z','tilt_deg','speed','omega','left_N','right_N',
         'nonfoot_N','mouth_N','penetration_m','limit_excess_rad','com_z']


def run(plant, depth, case, seed, folder):
    key=f'd{depth}_c{case.command_ms}_p{case.position_ms}_v{case.voltage:g}_dt{case.physics_dt:g}_s{seed}'
    target=folder/(key+'.json');assert not target.exists(),key
    result=dict(key=key,depth_mm=depth,case=asdict(case),seed=seed,
                model_sha256=sha(plant/'nominal.mjb'),kind='scripted feasibility, not learned policy')
    e=configure_owned(HeadExperiment(plant,CALIBRATOR,case,
        head_config=HeadConfig(max_measurement_age_s=.04)),(1,))
    s=e.sim;m=s.model;d=s.data
    try:
        result['calibration']=e.reset(seed)
    except ValueError as exc:
        result.update(status='CALIBRATION_REJECTED',error=str(exc),passed=False)
        target.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result),flush=True);return result
    reference=next(r for r in json.loads((ROOT/'20260914_fold_recovery/fold_geometry.json').read_text()) if r['depth_mm']==depth)
    # Preserve the calibrated standing servo offsets and transport histories.
    initial=s.applied.copy();delta=np.asarray(reference['target'])-s.home
    legs=[i for i,n in enumerate(s.names) if n.startswith(('left_','right_'))]
    def trajectory(nominal,obs):
        t=d.time-WARMUP_SECONDS
        fraction=np.clip((t-1)/3,0,1) if t<7 else 1-np.clip((t-7)/3,0,1)
        fraction=fraction*fraction*(3-2*fraction)
        goal=initial.copy();goal[legs]+=fraction*delta[legs]
        return e.transform_target(goal,obs)
    s.transform_target=trajectory
    rows=[];qpos=[];tau=[];velocity=[];floor=m.geom('terrain').id
    previous=s.substep_callback
    def record():
        previous();forces=np.zeros(4);wrench=np.zeros(6);penetration=0.
        for i,c in enumerate(d.contact):
            if c.efc_address<0: continue
            mujoco.mj_contactForce(m,d,i,wrench)
            normal=max(0.,wrench[0]);pair=(c.geom1,c.geom2)
            if floor in pair:
                other=c.geom2 if c.geom1==floor else c.geom1
                if other in s.foot_geoms: forces[s.foot_geoms.index(other)]+=normal
                else: forces[2]+=normal
                penetration=max(penetration,-c.dist)
            elif all(m.geom(g).name.startswith('robot/mouth_') for g in pair):forces[3]+=normal
        rot=d.xmat[s.body].reshape(3,3)
        q=d.qpos[s.jadr];limits=m.jnt_range[s.jids]
        excess=np.maximum(limits[:,0]-q,q-limits[:,1]).clip(0).max()
        rows.append([d.time-WARMUP_SECONDS,d.qpos[2],
            np.degrees(np.arccos(np.clip(rot[2,2],-1,1))),np.linalg.norm(d.qvel[:3]),
            np.linalg.norm(d.qvel[3:6]),*forces,penetration,excess,d.subtree_com[s.body,2]])
        tau.append(d.actuator_force[s.aids].copy());velocity.append(d.qvel[s.vadr].copy())
        if len(rows)%round(.04/case.physics_dt)==0:qpos.append(d.qpos.copy())
    s.substep_callback=record
    for _ in range(round(13/s.dt)):
        e.poll();s.step(np.zeros(18))
        if not np.isfinite(d.qpos).all():raise ValueError('Nonfinite state')
    a=np.array(rows);hold=a[(a[:,0]>=4)&(a[:,0]<7)]
    final=a[a[:,0]>=10];base=a[(a[:,0]>=.5)&(a[:,0]<1),1].mean()
    stable=lambda x:(x[:,2]<10)&(x[:,3]<.03)&(x[:,4]<.3)&(x[:,5]>.5)&(x[:,6]>.5)&(x[:,7]<.2)
    hold_good=float(stable(hold).mean());final_good=float((stable(final)&(abs(final[:,1]-base)<.005)).mean())
    depth_actual=float((base-hold[:,1].mean())*1000)
    # Predeclared intended pose requires feet support, not a body-rest shortcut.
    passed=bool(hold_good>=.95 and final_good>=.95 and depth_actual>=depth-5
        and a[:,9].max()<=.002 and a[:,10].max()<=np.deg2rad(.5))
    result.update(status='COMPLETE',passed=passed,standing_root_mm=float(base*1000),
        actual_depth_mm=depth_actual,hold_root_mm=float(hold[:,1].mean()*1000),
        hold_tilt_max_deg=float(hold[:,2].max()),hold_stable_fraction=hold_good,
        hold_bilateral_support_fraction=float(((hold[:,5]>.5)&(hold[:,6]>.5)).mean()),
        hold_body_support_max_N=float(hold[:,7].max()),return_stable_fraction=final_good,
        final_tilt_max_deg=float(final[:,2].max()),floor_penetration_max_mm=float(a[:,9].max()*1000),
        joint_limit_excess_max_deg=float(np.rad2deg(a[:,10].max())),
        mouth_force_max_N=float(a[:,8].max()),hold_knee_deg=np.rad2deg(np.mean(np.array(qpos)[100:175][:,s.jadr[[s.names.index('left_knee'),s.names.index('right_knee')]]],axis=0)).tolist())
    np.savez_compressed(folder/(key+'.npz'),trace=a,columns=np.array(COLUMNS),qpos=np.array(qpos),
        torque=np.array(tau),joint_velocity=np.array(velocity),frames_dt=.04)
    target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
    return result


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--plant',default='plant');p.add_argument('--out',default='nominal')
    p.add_argument('--depths',type=int,nargs='+',default=[30,40,50]);p.add_argument('--delays',type=int,nargs='+',default=[10])
    p.add_argument('--seeds',type=int,nargs='+',default=[601]);p.add_argument('--voltage',type=float,default=12.)
    p.add_argument('--position',type=int,default=20);p.add_argument('--dt',type=float,default=.00125)
    args=p.parse_args();folder=OUT/args.out;folder.mkdir(exist_ok=False)
    plan=dict(arguments=vars(args),gate=dict(hold_seconds=3,return_seconds=3,stable_fraction=.95,
        tilt_deg=10,speed_m_s=.03,angular_speed_rad_s=.3,each_foot_load_N=.5,
        body_support_max_N=.2,depth_tolerance_mm=5,penetration_max_mm=2,joint_excess_deg=.5),
        sources={n:sha(ROOT/n) for n in ['try_compact_fold.py','compact_contact_cfg.py','hardware_sim.py','head_attitude_sim.py']},
        started_unix=time.time(),constant_battery_position=True)
    (folder/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    results=[]
    for delay in args.delays:
        for seed in args.seeds:
            for depth in args.depths:
                case=HardwareCase(physics_dt=args.dt,motor_curve=True,voltage=args.voltage,
                    command_ms=delay,position_ms=args.position,velocity_ms=20)
                results.append(run(OUT/args.plant,depth,case,seed,folder))
    (folder/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()
