"""Head-envelope recovery curriculum with full provenance of each reset."""
import json
import numpy as np
from build_contact_motion_bank import OUT,experiment,prepare_fall,snapshot
selected=json.loads((OUT/'selected_contact.json').read_text())
base=json.loads((OUT/('standing_reset_bank'+selected['bank_suffix']+'.json')).read_text())
dest=OUT/'safe_fall_preparation';dest.mkdir(exist_ok=False)
audit=[]
for name,direction in [('left',[0,1,0]),('right',[0,-1,0]),('front',[-1,0,0]),('back',[1,0,0])]:
    e,cal=experiment(2,head_envelope=True);trace,frames=prepare_fall(e,direction,capture_tilts=True)
    q=e.sim.data.qpos[e.sim.jadr];limits=e.sim.model.jnt_range[e.sim.jids]
    head=q[e.sim.names.index('head_pitch')]
    valid=bool(trace[-1,2]>55 and trace[-1,3]<.05 and trace[-1,4]<.5 and trace[-1,6]<.002 and np.rad2deg(head)<29
        and max(np.maximum(limits[:,0]-q,q-limits[:,1]))<.005)
    if valid:
        r=snapshot(e,cal,name,2);base['records'].extend([r.copy() for _ in range(3)])
    for r in e.lean_snapshots:
        head=r['joint_pos'][e.sim.names.index('head_pitch')]
        if not np.deg2rad(-2)<head<np.deg2rad(29):continue
        r.update(label=name+'_'+r['label'],calibration=cal,seed=2);base['records'].append(r)
    np.savez_compressed(dest/(name+'.npz'),trace=trace,qpos=frames,frames_dt=.04)
    audit.append(dict(label=name,accepted=valid,final_tilt_deg=float(trace[-1,2]),head_pitch_deg=float(np.rad2deg(head)),
        max_preparation_penetration_mm=float(trace[:,6].max()*1000),external_push_only_before_episode=True))
assert any(r['accepted'] for r in audit)
base.update(head_pitch_command_range_deg=[0,27],head_pitch_failure_range_deg=[-2,29],
    curriculum='Standing + physically generated leaning + resting fallen. No force or pose writes after learned episode begins.')
name='getup_reset_bank_normal005_headsafe.json';target=OUT/name;assert not target.exists()
target.write_text(json.dumps(base,indent=2));(dest/'audit.json').write_text(json.dumps(audit,indent=2))
selected['getup_bank']=name;(OUT/'selected_contact.json').write_text(json.dumps(selected,indent=2))
print(len(base['records']),json.dumps(audit),flush=True)
