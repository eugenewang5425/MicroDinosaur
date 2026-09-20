"""Matched calibrated CPU tests; command phases enter before the body PI gate."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from terrain_skill_eval import TerrainSkillExperiment
from hardware_sim import HardwareCase
from heading_sim import WARMUP_SECONDS
from evaluate_policy import sha

ROOT=Path(__file__).parent
OUT=ROOT/'20260914_transition_refine'


class TransitionExperiment(TerrainSkillExperiment):
    def __init__(self,*args,program='hold',**kwargs):
        self.program=program
        super().__init__(*args,**kwargs)

    def reset(self,*args,**kwargs):
        self.shaped_user=np.zeros(18);self.user_rows=[]
        return super().reset(*args,**kwargs)

    def user_command(self,t,scenario):
        wanted=np.zeros(18)
        if self.program=='resume' and 11<=t<17:wanted[0]=.35
        self.shaped_user[:3]+=np.clip(wanted[:3]-self.shaped_user[:3],-.03,.03)
        self.user_rows.append(np.r_[t,self.shaped_user])
        return self.shaped_user.copy()


def phase_stats(full,start,end,reference_yaw=0.):
    rows=full[(full[:,0]-WARMUP_SECONDS>=start)&(full[:,0]-WARMUP_SECONDS<end)]
    dt=np.diff(rows[:,0]);planar=np.diff(rows[:,1:3],axis=0)/dt[:,None]
    yaw=np.unwrap(rows[:,5])-reference_yaw
    yaw-=2*np.pi*round(yaw[0]/(2*np.pi))
    rms=lambda v:float(np.sqrt(np.mean(v*v)))
    return dict(net_drift_mm_s=float(np.linalg.norm(rows[-1,1:3]-rows[0,1:3])/(rows[-1,0]-rows[0,0])*1000),
        planar_speed_mean_mm_s=float(np.linalg.norm(planar,axis=1).mean()*1000),
        body_vx_mean_m_s=float(rows[:,13].mean()),height_mean_mm=float(rows[:,3].mean()*1000),
        camera_yaw_rms_deg=float(np.rad2deg(rms(yaw))),
        camera_yaw_rate_rms_deg_s=float(np.rad2deg(rms(rows[:,7]))),
        camera_pitch_mean_deg=float(np.rad2deg(rows[:,6].mean())),
        camera_pitch_rms_deg=float(np.rad2deg(rms(rows[:,6]))),
        camera_pitch_std_deg=float(np.rad2deg(rows[:,6].std())))


def score(e,m):
    full=np.asarray(e.sim.gaze_trace)
    initial=phase_stats(full,.5,1.8)
    zero=full[np.argmin(abs(full[:,0]-WARMUP_SECONDS)),4]
    phases={name:phase_stats(full,start,end,zero) for name,start,end in
        [('initial',.5,1.8),('crouch',4.2,6.5),('returned',9.5,10.8),('late',m['seconds']-3,m['seconds'])]}
    if m['seconds']>=20:phases['walk_window']=phase_stats(full,13,16,zero)
    depth=int(e.posture.removeprefix('crouch')) if e.posture!='none' else 0
    drop=initial['height_mean_mm']-phases['crouch']['height_mean_mm']
    error=phases['returned']['height_mean_mm']-initial['height_mean_mm']
    success=not m['fell'] and abs(drop-depth)<=6 and abs(error)<=5 and phases['late']['net_drift_mm_s']<=10
    m.update(phases=phases,actual_drop_mm=drop,return_height_error_mm=error,
        transition_success=bool(success))
    return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--label',required=True);p.add_argument('--suite',choices=('transitions','confirmation'),required=True)
    a=p.parse_args();dest=OUT/'evaluation'/a.label/a.suite;dest.mkdir(parents=True,exist_ok=False)
    cases=[dict(program=program,depth=40,lag=lag,seconds=20,terrain='flat',scenario='stand')
        for program in ('hold','resume') for lag in (5,10,15)] if a.suite=='transitions' else [
        dict(program='legacy',depth=0,lag=10,seconds=12,terrain='flat',scenario='stand'),
        dict(program='legacy',depth=0,lag=10,seconds=12,terrain='flat',scenario='straight'),
        dict(program='legacy',depth=0,lag=10,seconds=12,terrain='steps_10',scenario='straight'),
        dict(program='hold',depth=30,lag=10,seconds=20,terrain='flat',scenario='stand')]
    source_names=['evaluate_transition_refine.py','heading_sim.py','head_attitude_sim.py','head_attitude.py',
        'terrain_skill_eval.py','hardware_sim.py','evaluate_policy.py','evaluate_gaze_ablation.py','run_heading_stable_start.py','imu_heading.py']
    sources={name:sha(ROOT/name) for name in source_names}
    records=[]
    for case in cases:
        for seed in (1,2,3):
            name=f"{case['program']}_{case['terrain']}_{case['scenario']}_d{case['depth']}_lag{case['lag']}_s{seed}"
            hardware=HardwareCase(physics_dt=.00125,command_ms=case['lag'])
            kwargs=dict(terrain=case['terrain'],posture=f"crouch{case['depth']}" if case['depth'] else 'none',hardware_case=hardware)
            e=TerrainSkillExperiment(a.policy,**kwargs) if case['program']=='legacy' else TransitionExperiment(a.policy,program=case['program'],**kwargs)
            m,trace=e.run(case['scenario'],'imu',seed,case['seconds'],True)
            score(e,m)
            record=dict(status='COMPLETE',name=name,case=case,seed=seed,policy_sha256=sha(a.policy),
                hardware=asdict(hardware),source_hashes=sources,metrics=m)
            arrays=dict(trace=trace,gaze=np.asarray(e.sim.gaze_trace),commands=np.asarray(e.command_rows),
                head_controls=np.asarray(e.head_rows),head_physics=np.asarray(e.head_physics))
            if hasattr(e,'user_rows'):arrays['user_commands']=np.asarray(e.user_rows)
            np.savez_compressed(dest/(name+'.npz'),**arrays)
            (dest/(name+'.json')).write_text(json.dumps(record,indent=2),encoding='utf-8');records.append(record)
            print(json.dumps(dict(name=name,fell=m['fell'],drop=m['actual_drop_mm'],pitch=m['camera_pitch_rms_deg'],
                vx=m['body_vx_mean_m_s'],late_drift=m['phases']['late']['net_drift_mm_s'],success=m['transition_success'])),flush=True)
    (dest/'matrix.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':main()
