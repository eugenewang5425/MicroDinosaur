"""CPU feet-first acceptance: deep crouch, simultaneous soles clear, landing separate."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
from evaluate_jump_refine import RefineExperiment
from evaluate_run_jump import score
from hardware_sim import HardwareCase
from imu_owned_head import configure_owned
from probe_tuck_jump import feet_metrics
from foot_flight_cfg import OUT
from evaluate_policy import sha


class FeetExperiment(RefineExperiment):
    def __init__(self,policy,case,return_to_base,depth):
        self.depth=depth
        super().__init__(policy,case,return_to_base)
    def user_command(self,t,scenario):
        command=super().user_command(t,scenario)
        command[9]=-self.depth if t<1.8 else (.015 if t<2.4 else 0.)
        return command


def jobs():
    for depth in [.03,.05]:
        for delay in [5,10,15]:
            for seed in [601,602,603]:
                for ret in [False,True]:
                    yield depth,HardwareCase(physics_dt=.00125,motor_curve=True,voltage=12.,command_ms=delay),seed,ret
    stress=[HardwareCase(name='voltage11.1',physics_dt=.00125,motor_curve=True,voltage=11.1),
        HardwareCase(name='voltage9.9',physics_dt=.00125,motor_curve=True,voltage=9.9),
        HardwareCase(name='fine_dt',physics_dt=.000625,motor_curve=True,voltage=12.),
        HardwareCase(name='effort80',physics_dt=.00125,motor_curve=True,voltage=12.,effort_fraction=.8),
        HardwareCase(name='feedback40',physics_dt=.00125,motor_curve=True,voltage=12.,position_ms=40),
        HardwareCase(name='friction0.6',physics_dt=.00125,motor_curve=True,voltage=12.,ground_friction=.6)]
    for case in stress:
        for ret in [False,True]:yield .05,case,604,ret


def assess(e,m):
    a=np.asarray(e.rows);score(e,m)
    launch=a[(a[:,0]>=1.8)&(a[:,0]<2.8)]
    m.update(feet_metrics(launch,e.sim.model.opt.timestep))
    pre=a[(a[:,0]>=1.3)&(a[:,0]<1.8)]
    depth=float(a[0,1]-np.mean(pre[:,1]));support=float(((pre[:,5]>.1)&(pre[:,6]>.1)).mean())
    m.update(requested_crouch_m=e.depth,actual_crouch_m=depth,crouch_both_supported_fraction=support,
        crouch_tilt_deg=float(pre[:,8].max()),maximum_mesh_penetration_m=float(max(0.,-a[:,14:16].min())),
        deep_crouch_achieved=bool(depth>=.045 and support>.95 and pre[:,8].max()<10),
        feet_and_landing_success=bool(m['feet_goal'] and m['settled_final_second'] and not m['motion_failed']))
    m['complete_deep_tuck_jump']=bool(m['feet_and_landing_success'] and m['deep_crouch_achieved'])
    return m


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--quick',action='store_true');args=p.parse_args()
    dest=OUT/'evaluation'/args.label;dest.mkdir(parents=True,exist_ok=False);rows=[]
    for depth,case,seed,ret in jobs():
        if args.quick and not(seed==601 and case.command_ms==10):continue
        name=f'c{round(depth*1000)}_d{case.command_ms}_s{seed}_{case.name}_return{int(ret)}'
        e=configure_owned(FeetExperiment(args.policy,case,ret,depth),(1,))
        row=dict(name=name,policy=str(args.policy),policy_sha256=sha(args.policy),hardware=asdict(case),seed=seed,
            requested_depth_m=depth,return_to_base=ret,preparation_s=1.8,extension_m=.015,extension_s=.6)
        try:
            m,heading=e.run('stand','imu',seed,6.,True);assess(e,m)
            if ret:
                reference=np.load(dest/(name.replace('return1','return0')+'.npz'))['physics'];actual=np.asarray(e.rows)
                mask=actual[:,0]<2.8-1e-7;error=float(abs(actual[mask]-reference[mask]).max());assert error==0
                row['pre_return_max_error']=error
            row.update(status='COMPLETE',metrics=m,handoff=e.handoff)
            np.savez_compressed(dest/(name+'.npz'),physics=np.asarray(e.rows),extra=np.asarray(e.extra),qpos=np.asarray(e.qpos_frames),
                heading=heading,head_physics=np.asarray(e.head_physics))
            print(json.dumps(dict(name=name,actual_crouch_mm=depth and m['actual_crouch_m']*1000,gap_mm=m['both_foot_peak_gap_m']*1000,
                continuous5_ms=m['continuous_5mm_s']*1000,feet=m['feet_goal'],complete=m['feet_and_landing_success'],limit=m['joint_limit_excess_rad'])),flush=True)
        except Exception as exc:row.update(status='FAILED',error=repr(exc));print(json.dumps(row),flush=True)
        rows.append(row);(dest/(name+'.json')).write_text(json.dumps(row,indent=2),encoding='utf-8')
        (dest/'matrix.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':main()
