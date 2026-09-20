"""Same frozen yaw-IMU runtime for before/after terrain learning confirmation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from imu_owned_head import configure_owned
from terrain_skill_eval import TerrainSkillExperiment
from evaluate_transition_refine import TransitionExperiment,score,phase_stats
from hardware_sim import HardwareCase
from evaluate_policy import sha
from terrain_lane_cfg import OUT

ROOT=Path(__file__).parent


def cases():
    rows=[dict(terrain='flat',scenario=s,program='legacy',depth=0,delay=10,speed=None,seconds=12) for s in ('stand','straight')]
    rows += [dict(terrain='steps_10',scenario='straight',program='legacy',depth=0,delay=10,speed=v,seconds=12) for v in (.2,.35,.55)]
    rows += [dict(terrain='downsteps_10',scenario='straight',program='legacy',depth=0,delay=10,speed=.35,seconds=12)]
    rows += [dict(terrain='flat',scenario='stand',program='hold',depth=30,delay=10,speed=None,seconds=20)]
    rows += [dict(terrain='flat',scenario='stand',program=p,depth=40,delay=d,speed=None,seconds=20) for p in ('hold','resume') for d in (5,10,15)]
    rows += [dict(terrain='flat',scenario=s,program='legacy',depth=0,delay=10,speed=None,seconds=12) for s in ('left_then_hold','right_then_hold')]
    return rows


def key(case,seed):return f"{case['terrain']}_{case['program']}_{case['scenario']}_d{case['depth']}_lag{case['delay']}_v{case['speed']}_s{seed}"


def experiment(policy,case):
    kwargs=dict(terrain=case['terrain'],posture=f"crouch{case['depth']}" if case['depth'] else 'none',
        hardware_case=HardwareCase(physics_dt=.00125,command_ms=case['delay']),walking_speed=case['speed'])
    e=TerrainSkillExperiment(policy,**kwargs) if case['program']=='legacy' else TransitionExperiment(policy,program=case['program'],**kwargs)
    return configure_owned(e,(1,))


def evaluate(e,case,seed):
    m,t=e.run(case['scenario'],'imu',seed,case['seconds'],True);score(e,m)
    if case['program']=='resume':
        f=np.asarray(e.sim.gaze_trace);m['post_stop']=phase_stats(f,18,20)
        m['braking_net_displacement_mm']=phase_stats(f,17,18)['net_drift_mm_s']*(1-e.sim.model.opt.timestep)
    m['terrain_traversed']=bool(not m['fell'] and m['forward_displacement_m']>=.8) if case['terrain']!='flat' else None
    return m,t


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--label',required=True)
    a=p.parse_args();dest=OUT/'evaluation'/a.label;dest.mkdir(parents=True,exist_ok=False)
    names=['evaluate_owned_terrain.py','imu_owned_head.py','heading_sim.py','head_attitude_sim.py','head_attitude.py',
        'evaluate_transition_refine.py','terrain_skill_eval.py','hardware_sim.py','evaluate_policy.py','evaluate_gaze_ablation.py','imu_heading.py','run_heading_stable_start.py']
    hashes={n:sha(ROOT/n) for n in names};records=[]
    for case in cases():
        for seed in (1,2,3):
            name=key(case,seed);e=experiment(a.policy,case);m,t=evaluate(e,case,seed)
            r=dict(status='COMPLETE',name=name,case=case,seed=seed,metrics=m,policy_sha256=sha(a.policy),
                runtime='yaw_owned',source_hashes=hashes);records.append(r)
            (dest/(name+'.json')).write_text(json.dumps(r,indent=2))
            arrays=dict(trace=t,gaze=np.asarray(e.sim.gaze_trace),commands=np.asarray(e.command_rows),
                head_controls=np.asarray(e.head_rows),head_physics=np.asarray(e.head_physics))
            if hasattr(e,'user_rows'):arrays['user_commands']=np.asarray(e.user_rows)
            np.savez_compressed(dest/(name+'.npz'),**arrays)
            print(json.dumps(dict(name=name,fell=m['fell'],traversed=m['terrain_traversed'],vx=m['body_vx_mean_m_s'],
                yaw=m['camera_heading_error_rms_deg'],pitch=m['camera_pitch_rms_deg'])),flush=True)
    (dest/'matrix.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
