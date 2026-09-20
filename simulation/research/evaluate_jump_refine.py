"""Paired CPU evaluation on unchanged plant: isolated specialist and continuous v7 return."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

from evaluate_run_jump import MotionExperiment,score,render
from evaluate_policy import sha
from jump_limit_probe import Probe,Recipe,additional_metrics
from jump_refine_cfg import OUT
from hardware_sim import HardwareCase
from imu_owned_head import configure_owned
from run_heading_stable_start import CALIBRATOR


class RefineExperiment(Probe):
    def __init__(self,policy,case,return_to_base):
        self.return_to_base=return_to_base
        super().__init__(case,Recipe(extension=.01))
        self.jump_session=ort.InferenceSession(str(policy),providers=['CPUExecutionProvider'])
        self.sim.session=self.jump_session
        if return_to_base:self.return_time=2.8

    def user_command(self,t,scenario):
        if self.return_to_base:return super().user_command(t,scenario)
        command=MotionExperiment.user_command(self,t,scenario)
        command[9]=-.03 if t<1.5 else (.01 if t<2.2 else 0.)
        return command


def jobs():
    for curve in [False,True]:
        for delay in [5,10,15]:
            for seed in [401,402,403]:
                for return_to_base in [False,True]:
                    yield HardwareCase(physics_dt=.00125,command_ms=delay,motor_curve=curve,voltage=12.),seed,return_to_base
    stress=[HardwareCase(name='voltage11.1',physics_dt=.00125,motor_curve=True,voltage=11.1),
            HardwareCase(name='voltage9.9',physics_dt=.00125,motor_curve=True,voltage=9.9),
            HardwareCase(name='fine_dt',physics_dt=.000625,motor_curve=True,voltage=12.),
            HardwareCase(name='effort80',physics_dt=.00125,motor_curve=True,voltage=12.,effort_fraction=.8),
            HardwareCase(name='feedback40',physics_dt=.00125,motor_curve=True,voltage=12.,position_ms=40),
            HardwareCase(name='friction0.6',physics_dt=.00125,motor_curve=True,voltage=12.,ground_friction=.6)]
    for case in stress:
        for return_to_base in [False,True]:yield case,404,return_to_base


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--label',required=True);p.add_argument('--quick',action='store_true');p.add_argument('--video',action='store_true')
    args=p.parse_args();dest=OUT/'evaluation'/args.label;dest.mkdir(parents=True,exist_ok=False)
    rows=[]
    for case,seed,use_return in jobs():
        if args.quick and not (seed==401 and case.command_ms==10):continue
        name=f'curve{int(case.motor_curve)}_d{case.command_ms}_s{seed}_{case.name}_return{int(use_return)}'
        e=configure_owned(RefineExperiment(args.policy,case,use_return),(1,))
        r=dict(name=name,policy=str(args.policy),policy_sha256=sha(args.policy),hardware=asdict(case),seed=seed,
               return_to_base=use_return,recipe=asdict(e.recipe),return_policy_sha256=sha(CALIBRATOR) if use_return else None)
        try:
            m,heading=e.run('stand','imu',seed,6.,True);score(e,m);additional_metrics(e,m)
            r.update(status='COMPLETE',metrics=m,handoff=e.handoff)
            if use_return:
                before=dest/(name.replace('_return1','_return0')+'.npz')
                a=np.asarray(e.rows);b=np.load(before)['physics'];prefix=a[:,0]<2.8-1e-7
                error=float(np.max(abs(a[prefix]-b[prefix])));assert error==0
                r['pre_return_trace_max_error']=error
            np.savez_compressed(dest/(name+'.npz'),physics=np.asarray(e.rows),extra=np.asarray(e.extra),
                                qpos=np.asarray(e.qpos_frames),heading=heading)
            if args.video and use_return and seed==401 and case.command_ms==10:
                try:render(e,dest/(name+'.mp4'));r['video_status']='COMPLETE'
                except Exception as exc:r.update(video_status='FAILED',video_error=repr(exc))
            print(json.dumps(dict(name=name,height_mm=1000*m['max_qualified_com_height_m'],flight_ms=1000*m['max_clear_flight_s'],
                jump=m['jump_success'],limit=m['joint_limit_excess_rad'],speed=m['final_second_speed_m_s'])),flush=True)
        except Exception as exc:
            r.update(status='FAILED',error=repr(exc));print(json.dumps(r),flush=True)
        rows.append(r)
        (dest/(name+'.json')).write_text(json.dumps(r,indent=2),encoding='utf-8')
        (dest/'matrix.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':main()
