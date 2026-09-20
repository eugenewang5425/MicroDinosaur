"""Paired friction hypotheses; retain failed calibrations in the denominator."""
from pathlib import Path
import argparse,json,time
from hardware_sim import HardwareCase
from try_squat_reference import run
from evaluate_policy import sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_contact_motion'
PLANT=ROOT/'20260914_squat_specialist/plant'
POLICY=Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_squat_specialist/20260914_residual_train_512x101/candidate.onnx')
# Deliberately bounded around the inherited contact, no large rolling lock.
PROFILES={'baseline':(1.,.01,.000001),'slide_12':(1.2,.01,.000001),
    'torsion_15mm':(1.,.015,.000001),'roll_01mm':(1.,.01,.0001),
    'roll_05mm':(1.,.01,.0005),'combined':(1.1,.015,.0001)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profiles',nargs='+',default=list(PROFILES))
    p.add_argument('--seeds',nargs='+',type=int,default=[901,902,903]);p.add_argument('--out',default='scan')
    p.add_argument('--dt',type=float,default=.00125)
    p.add_argument('--delays',type=int,nargs='+',default=[10])
    p.add_argument('--voltage',type=float,default=12.);p.add_argument('--position',type=int,default=20)
    p.add_argument('--contact-timeconst',type=float);a=p.parse_args()
    folder=OUT/a.out;folder.mkdir(parents=True,exist_ok=False)
    (folder/'plan.json').write_text(json.dumps(dict(profiles={k:PROFILES[k] for k in a.profiles},
        args=vars(a),plant_sha=sha(PLANT/'nominal.mjb'),policy_sha=sha(POLICY),
        source_sha={n:sha(ROOT/n) for n in ['contact_motion_scan.py','try_squat_reference.py','hardware_sim.py']},
        measured=False,selection='Require squat gates, head high-frequency jitter and contact slip improvement; validate other motion separately'),indent=2))
    rows=[]
    for name in a.profiles:
        dest=folder/name;dest.mkdir()
        mu,tor,roll=PROFILES[name]
        for delay in a.delays:
         for seed in a.seeds:
            start=time.time()
            case=HardwareCase(name=name,motor_curve=True,voltage=a.voltage,physics_dt=a.dt,
                command_ms=delay,position_ms=a.position,
                ground_friction=mu,foot_torsional_friction_m=tor,foot_rolling_friction_m=roll,
                foot_contact_timeconst_s=a.contact_timeconst,body_contact_timeconst_s=a.contact_timeconst)
            result=run(PLANT,25,case,seed,dest,POLICY,residual=True)
            result['wall_s']=time.time()-start;rows.append(result)
            (folder/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
