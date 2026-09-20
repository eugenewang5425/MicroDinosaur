"""Predeclared independent squat completion and friction/delay checks."""
from pathlib import Path
from dataclasses import asdict
import argparse,json
from hardware_sim import HardwareCase
from squat_plant import OUT
from try_squat_reference import run
from evaluate_policy import sha

p=argparse.ArgumentParser();p.add_argument('--onnx',type=Path)
p.add_argument('--out',required=True);p.add_argument('--residual',action='store_true')
p.add_argument('--confirmation',action='store_true');a=p.parse_args()
assert a.onnx is not None or a.residual,'A policy or explicit zero-residual baseline is required'
folder=OUT/a.out;folder.mkdir(exist_ok=False)
cases=[]
for seed in (701,702,703):
    for delay in (5,10,15):
        cases.append((seed,HardwareCase(physics_dt=.00125,motor_curve=True,voltage=12.,command_ms=delay)))
for mu in (.6,1.3):
    cases.append((704,HardwareCase(physics_dt=.00125,motor_curve=True,voltage=12.,ground_friction=mu)))
cases.append((705,HardwareCase(physics_dt=.00125,motor_curve=True,voltage=11.1,position_ms=40)))
cases.append((706,HardwareCase(physics_dt=.000625,motor_curve=True,voltage=12.)))
if a.confirmation:
    cases=[(s,HardwareCase(physics_dt=.00125,motor_curve=True,voltage=12.)) for s in (801,802,803)]
plan=dict(policy_sha256=sha(a.onnx) if a.onnx else None,plant_sha256=sha(OUT/'plant/nominal.mjb'),
    action_mode='bounded residual' if a.residual else 'raw position',
    depth_mm=25,hold_s=3,return_check_s=3,criterion='>=95% stable bilateral support during hold and return; depth within 5mm; tilt<10deg; ankle<55deg; no body support; physical limits; penetration<=2mm',
    initial_rejections_remain_in_denominator=True,policy_runs_entire_13s_after_v7_calibration=True,
    cases=[dict(seed=s,hardware=asdict(c)) for s,c in cases])
(folder/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
results=[]
for seed,case in cases:
    results.append(run(OUT/'plant',25,case,seed,folder,str(a.onnx) if a.onnx else None,residual=a.residual))
    (folder/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('PASS',sum(r['passed'] for r in results),'/',len(cases),flush=True)
