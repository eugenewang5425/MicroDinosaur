"""Slow dynamic probes of folded-leg candidates; keep legacy studies frozen."""
from pathlib import Path
import argparse,json
import numpy as np
import try_squat_reference as reference
from hardware_sim import HardwareCase
ROOT=Path(__file__).parent;OUT=ROOT/'20260914_requested_fold'
p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[961])
p.add_argument('--delays',nargs='+',type=int,default=[10]);p.add_argument('--out',default='dynamic')
p.add_argument('--ankle-target',type=float,default=55);p.add_argument('--knees',type=int,nargs='+',default=[70,75]);a=p.parse_args()
dest=OUT/a.out;dest.mkdir(exist_ok=False)
rows=json.loads((OUT/'kinematic_candidates.json').read_text())
poses=json.loads((ROOT/'20260914_squat_specialist/pose_references.json').read_text())
names=json.loads((ROOT/'20260914_contact_motion/preferred_plant/contract.json').read_text())['action_names']
ankles=[names.index(n) for n in ('left_ankle','right_ankle')]
# Only the file lookup is redirected, not the old files or physical parameters.
reference.OUT=dest
mapped=[]
for knee,depth in [(70,35),(75,38)]:
    if knee not in a.knees:continue
    row=next(r for r in rows if r['requested_knee_deg']==knee)
    target=np.array(row['target'])
    # Back away from the 55 degree task guard; retain the physical limits.
    target[ankles]=np.clip(target[ankles],-np.deg2rad(a.ankle_target),np.deg2rad(a.ankle_target))
    mapped.append(dict(depth_mm=depth,target=target.tolist(),requested_knee_deg=knee))
(dest/'pose_references.json').write_text(json.dumps([r for r in poses if r['depth_mm']==25]+mapped,indent=2))
(dest/'plan.json').write_text(json.dumps(vars(a),indent=2))
results=[]
for row in mapped:
  for delay in a.delays:
    for seed in a.seeds:
        case=HardwareCase(motor_curve=True,voltage=12,command_ms=delay,position_ms=20,velocity_ms=20,
            physics_dt=.00125)
        r=reference.run(ROOT/'20260914_contact_motion/preferred_plant',row['depth_mm'],case,seed,dest)
        r['requested_knee_deg']=row['requested_knee_deg']
        r['status_note']='Unlearned dynamic probe; collision audit still required. Photo-pose acceptance remains unfulfilled.'
        if r['status']=='COMPLETE':
            r['fold_tracking_gate']=bool(max(abs(np.abs(r['hold_knee_deg'])-row['requested_knee_deg']))<5)
        else:r['fold_tracking_gate']=False
        (dest/(r['key']+'.json')).write_text(json.dumps(r,indent=2));results.append(r)
        (dest/'summary.json').write_text(json.dumps(results,indent=2))
