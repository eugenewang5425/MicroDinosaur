"""A friction hypothesis must still permit commanded left/right turns."""
from pathlib import Path
import json
import numpy as np
from evaluate_run_jump import MotionExperiment
from imu_owned_head import configure_owned
from hardware_sim import HardwareCase
OUT=Path(__file__).parent/'20260914_contact_motion'
POLICY=Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_run_specialist/20260914_train_512x301/candidate.onnx')
class Turn(MotionExperiment):
    def user_command(self,t,scenario):
        command=np.zeros(18);command[0]=.35
        command[2]=(.45 if scenario=='left' else -.45) if 2<=t<4 else 0
        return command
if __name__=='__main__':
    dest=OUT/'turning';dest.mkdir(exist_ok=False);rows=[]
    for torsion in (.01,.015):
      for side in ('left','right'):
       for seed in (951,952):
        key=f'{side}_tor{torsion}_s{seed}'
        e=configure_owned(Turn(POLICY,'run',.35,HardwareCase(motor_curve=True,voltage=12.,physics_dt=.00125,
            foot_torsional_friction_m=torsion),plant=OUT/'normal_contact_plant',operating_envelope=True),(1,))
        r=dict(key=key,torsion=torsion,seed=seed,direction=side)
        try:
            metrics,trace=e.run(side,'imu',seed,10.,True)
            r.update(status='COMPLETE',metrics=metrics)
            np.savez_compressed(dest/(key+'.npz'),physics=np.asarray(e.rows),heading=trace,qpos=np.asarray(e.qpos_frames))
        except Exception as exc:r.update(status='REJECTED',error=repr(exc))
        rows.append(r);(dest/'summary.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps(dict(key=key,status=r['status'],error_deg=r.get('metrics',{}).get('heading_error_final_deg'))),flush=True)
