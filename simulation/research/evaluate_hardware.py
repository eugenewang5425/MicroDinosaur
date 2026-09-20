"""Hardware-grounded protocol checks and explicitly hypothetical sensitivities."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from evaluate_policy import run_trial,sha
from hardware_sim import HardwareCase,HardwareSim


def cases():
    return [
        HardwareCase(name='legacy_reference',protocol=False),
        HardwareCase(),
        HardwareCase(name='curve_12v6',motor_curve=True),
        HardwareCase(name='curve_11v1',motor_curve=True,voltage=11.1),
        HardwareCase(name='curve_9v9',motor_curve=True,voltage=9.9),
        HardwareCase(name='effort_80pct',effort_fraction=.8),
        HardwareCase(name='kd_low',kd_scale=.62),
        HardwareCase(name='kd_high',kd_scale=1.25),
        HardwareCase(name='joint_friction_high',joint_friction_scale=1.3),
        HardwareCase(name='rotor_inertia_half',armature_scale=.5),
        HardwareCase(name='rotor_inertia_150pct',armature_scale=1.5),
        HardwareCase(name='floor_mu_06',ground_friction=.6),
        HardwareCase(name='floor_mu_12',ground_friction=1.2),
        HardwareCase(name='slope_up_5deg',slope_deg=-5),
        HardwareCase(name='slope_down_5deg',slope_deg=5),
        HardwareCase(name='head_mass_110pct',head_mass_scale=1.1),
        HardwareCase(name='body_gyro_bias_positive',gyro_bias_deg_s=.5),
        HardwareCase(name='body_gyro_bias_negative',gyro_bias_deg_s=-.5),
        HardwareCase(name='imu_poll_40ms',imu_period_ms=40),
        HardwareCase(name='bus_feedback_loss_2pct',feedback_drop_probability=.02),
        HardwareCase(name='leg_bus_stall_40ms',bus_stall_ms=40),
        HardwareCase(name='lateral_impulse_02Ns',push_force_n=2),
        HardwareCase(name='combined_hypothesis',motor_curve=True,voltage=11.1,effort_fraction=.85,
            ground_friction=.7,head_mass_scale=1.05,gyro_bias_deg_s=.3,feedback_drop_probability=.02),
        HardwareCase(name='protocol_dt_2p5ms',physics_dt=.0025),
        HardwareCase(name='protocol_dt_1p25ms',physics_dt=.00125),
        HardwareCase(name='curve_dt_2p5ms',motor_curve=True,physics_dt=.0025),
        HardwareCase(name='curve_dt_1p25ms',motor_curve=True,physics_dt=.00125),
        HardwareCase(name='protocol_dt_0p625ms',physics_dt=.000625),
        HardwareCase(name='curve_dt_0p625ms',motor_curve=True,physics_dt=.000625),
    ]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plant',required=True)
    p.add_argument('--policy',action='append',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--cases',default='all',help='Comma-separated case names, or all')
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    selected=[c for c in cases() if args.cases=='all' or c.name in args.cases.split(',')]
    assert selected
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    model_hash=sha(Path(args.plant)/'nominal.mjb')
    implementation_hash=sha(Path(__file__).with_name('hardware_sim.py'))
    started=time.time();records=[]
    for spec in args.policy:
        label,onnx=spec.split('=',1)
        for case in selected:
            dest=out/f'{label}__{case.name}.json'
            if dest.exists():
                if not args.resume:raise FileExistsError(dest)
                old=json.loads(dest.read_text())
                assert old['case']==asdict(case) and old['onnx_sha256']==sha(Path(onnx))
                assert old['plant_sha256']==model_hash and old['implementation_sha256']==implementation_hash
                assert len(old['trials'])==3*args.seeds
                records.append(old);continue
            sim=HardwareSim(args.plant,onnx,case)
            trials=[]
            for name,vx,tail,seconds in [('stand',0.,0.,5),('forward',.55,0.,6),('tail_up',.35,1.57,6)]:
                cmd=np.zeros(18);cmd[0]=vx;cmd[16]=tail
                for seed in range(args.seeds):
                    trial=run_trial(sim,cmd,seed,seed>0,seconds=seconds)
                    trial.update(task=name,physics=sim.physics_metrics(seconds))
                    trials.append(trial)
            record={'policy':label,'onnx_sha256':sha(Path(onnx)),'plant_sha256':model_hash,
                'implementation_sha256':implementation_hash,'case':asdict(case),
                'action_names':sim.names,'actual_total_mass_kg':float(sim.model.body_mass.sum()),
                'trials':trials,'wall_seconds':time.time()-started,
                'electrical_current_or_temperature_identified':False}
            dest.write_text(json.dumps(record,indent=2),encoding='utf-8');records.append(record)
            print(json.dumps({'policy':label,'case':case.name,'falls':sum(t['fell'] for t in trials),
                'elapsed_s':round(time.time()-started)}),flush=True)
    (out/'matrix.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':main()
