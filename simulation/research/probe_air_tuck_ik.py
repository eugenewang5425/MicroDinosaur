"""Learned push + IK feet retraction: physics-only teacher feasibility, never teleports the robot."""
import argparse
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
import onnxruntime as ort
from scipy.optimize import least_squares
from evaluate_foot_flight import FeetExperiment,assess
from run_heading_stable_start import CALIBRATOR
from hardware_sim import HardwareCase
from head_attitude import rotation_vector
from imu_owned_head import configure_owned
from foot_flight_cfg import OUT

POLICY=OUT.parent/'20260914_jump_refine/candidate.onnx'


def retract(e,lift,policy_target=None):
    m=e.sim.model;d=mujoco.MjData(m);d.qpos[:]=e.sim.data.qpos;mujoco.mj_forward(m,d)
    ids=np.array([i for i,n in enumerate(e.sim.names) if n.startswith(('left_','right_'))])
    adr=np.asarray(e.sim.jadr)[ids];limits=m.jnt_range[np.asarray(e.sim.jids)[ids]]
    if policy_target is not None:
        d.qpos[np.asarray(e.sim.jadr)]=policy_target
        d.qpos[adr]=np.clip(d.qpos[adr],limits[:,0]+.03,limits[:,1]-.03)
        mujoco.mj_forward(m,d)
    sites=[m.site('robot/'+s+'_foot').id for s in ['left','right']]
    position=d.site_xpos[sites].copy();position[:,2]+=lift
    rotation=d.site_xmat[sites].reshape(2,3,3).copy();q=d.qpos[adr].copy()
    def residual(x):
        d.qpos[adr]=x;mujoco.mj_forward(m,d);R=d.site_xmat[sites].reshape(2,3,3)
        return np.r_[(d.site_xpos[sites]-position).ravel()/.05,
            rotation_vector(rotation[0].T@R[0]),rotation_vector(rotation[1].T@R[1]),(x-q)*.01]
    solved=least_squares(residual,np.clip(q,limits[:,0]+.03,limits[:,1]-.03),bounds=(limits[:,0]+.03,limits[:,1]-.03),max_nfev=60)
    residual(solved.x)
    return ids,solved.x,float(np.linalg.norm(d.site_xpos[sites]-position,axis=1).max())


class Session:
    def __init__(self,e,policy,stand):self.e=e;self.policy=policy;self.stand=stand
    def get_inputs(self):return self.policy.get_inputs()
    def run(self,outputs,feed):
        e=self.e;t=e.current_t;action=(self.stand if t>=2.8 else self.policy).run(outputs,feed)[0].copy()
        if not e.active:return [action]
        if 1.8<=t<2.4:
            if e.trigger is None and e.rows:
                recent=np.asarray(e.rows)[-4:]
                if np.all(recent[:,4]<.05) and np.min(recent[-1,14:16])>0:
                    target=e.sim.home+action[0]*e.sim.scale if e.target_reference else None
                    ids,q,error=retract(e,e.lift,target);e.target=(ids,q);e.trigger=t;e.ik_error=error
            if e.trigger is not None:
                age=t-e.trigger
                blend=np.clip((e.hold+.04-age)/.04,0,1)
                ids,q=e.target;target=e.sim.home+action[0]*e.sim.scale
                target[ids]+=blend*(q-target[ids]);action[0]=(target-e.sim.home)/e.sim.scale
        e.obs.append(next(iter(feed.values()))[0].copy());e.actions.append(action[0].copy())
        return [action]


class Experiment(FeetExperiment):
    def __init__(self,depth,lift,hold,seed=501,case=None,target_reference=False):
        super().__init__(POLICY,case or HardwareCase(physics_dt=.00125,motor_curve=True,voltage=12.),False,depth)
        self.current_t=0.;self.trigger=None;self.target=None;self.ik_error=None;self.lift=lift;self.hold=hold;self.obs=[];self.actions=[];self.target_reference=target_reference
        stand=ort.InferenceSession(str(CALIBRATOR),providers=['CPUExecutionProvider'])
        self.jump_session=Session(self,self.jump_session,stand);self.sim.session=self.jump_session
    def user_command(self,t,scenario):self.current_t=t;return super().user_command(t,scenario)


def main():
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--target-reference',action='store_true');args=p.parse_args()
    dest=OUT/('air_ik_target_probes' if args.target_reference else 'air_ik_probes');dest.mkdir(exist_ok=False)
    (dest/'source_snapshot.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8');records=[]
    for depth in [.03,.05]:
        for lift in [.01,.02,.03]:
            for hold in [.04,.08]:
                e=configure_owned(Experiment(depth,lift,hold,target_reference=args.target_reference),(1,));i=len(records)
                row=dict(index=i,depth=depth,lift_m=lift,hold_s=hold,seed=501,oracle_ik_teacher=True,not_trained_actor=True,target_reference=args.target_reference)
                try:
                    m,h=e.run('stand','imu',501,6.,True);assess(e,m);row.update(status='COMPLETE',metrics=m,trigger_s=e.trigger,ik_error_m=e.ik_error)
                    np.savez_compressed(dest/f'{i:03d}.npz',physics=np.asarray(e.rows),qpos=np.asarray(e.qpos_frames),extra=np.asarray(e.extra),
                        heading=h,head_physics=np.asarray(e.head_physics),obs=np.asarray(e.obs),actions=np.asarray(e.actions))
                    print(json.dumps(dict(index=i,depth=depth,lift=lift,hold=hold,gap_mm=m['both_foot_peak_gap_m']*1000,
                        air5_ms=m['continuous_5mm_s']*1000,feet=m['feet_goal'],complete=m['feet_and_landing_success'],limit=m['joint_limit_excess_rad'])),flush=True)
                except Exception as exc:row.update(status='FAILED',error=repr(exc));print(json.dumps(row),flush=True)
                records.append(row);(dest/f'{i:03d}.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
                (dest/'matrix.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':main()
