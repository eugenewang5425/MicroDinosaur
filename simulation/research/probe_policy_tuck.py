"""Feasibility teacher: stable deep crouch, learned push, bounded airborne leg tuck."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
import onnxruntime as ort
from evaluate_foot_flight import FeetExperiment,assess
from evaluate_jump_return import JUMP
from run_heading_stable_start import CALIBRATOR
from hardware_sim import HardwareCase
from imu_owned_head import configure_owned
from foot_flight_cfg import OUT


class MixedSession:
    def __init__(self,e,policy,stand):self.e=e;self.policy=policy;self.stand=stand
    def get_inputs(self):return self.policy.get_inputs()
    def run(self,outputs,feed):
        e=self.e;t=e.current_t
        base=self.stand if t<1.8 or t>=2.8 else self.policy
        action=base.run(outputs,feed)[0].copy()
        if not e.active:return [action]
        target=e.sim.home+action[0]*e.sim.scale
        if t<1.8:
            f=np.clip(t/1.3,0,1);f=f*f*(3-2*f)
            ref=e.sim.home+(e.fold-e.sim.home)*f
            target[e.legs]=ref[e.legs]
        elif t<2.3:
            if e.tuck_time is None and e.rows:
                recent=np.asarray(e.rows)[-4:]
                if np.all(recent[:,4]<.05) and np.min(recent[-1,14:16])>0:e.tuck_time=t
            if e.tuck_time is not None:
                age=t-e.tuck_time
                blend=e.strength*np.clip(age/.04,0,1)*np.clip((e.hold+.04-age)/.04,0,1)
                target[e.legs]+=blend*(e.tuck[e.legs]-target[e.legs])
        action[0]=(target-e.sim.home)/e.sim.scale
        e.demo_obs.append(next(iter(feed.values()))[0].copy());e.demo_actions.append(action[0].copy())
        return [action]


class MixedExperiment(FeetExperiment):
    def __init__(self,case,depth,strength,hold):
        super().__init__(JUMP,case,False,depth)
        targets=json.loads((OUT/'deep_targets.json').read_text());table={round(r['depth_m'],6):np.array(r['target']) for r in targets['rows']}
        self.fold=table[depth];self.tuck=table[.05];self.current_t=0.;self.strength=strength;self.hold=hold
        self.tuck_time=None;self.demo_obs=[];self.demo_actions=[]
        self.legs=np.array([i for i,n in enumerate(self.sim.names) if n.startswith(('left_','right_'))])
        stand=ort.InferenceSession(str(CALIBRATOR),providers=['CPUExecutionProvider'])
        self.jump_session=MixedSession(self,self.jump_session,stand);self.sim.session=self.jump_session
    def user_command(self,t,scenario):
        self.current_t=t
        return super().user_command(t,scenario)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    dest=OUT/'hybrid_probes';dest.mkdir(exist_ok=False)
    (dest/'source_snapshot.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    records=[]
    for depth in [.03,.04,.05]:
        for strength in [0.,.35,.7]:
            e=configure_owned(MixedExperiment(HardwareCase(physics_dt=.00125,motor_curve=True,voltage=12.),depth,strength,.08),(1,))
            i=len(records);row=dict(index=i,depth=depth,tuck_strength=strength,seed=501,hardware=asdict(e.sim.case),
                controller='Script crouch, original learned jump, oracle-triggered tuck and v7 at 2.8s; not a single trained actor')
            try:
                m,heading=e.run('stand','imu',501,6.,True);assess(e,m)
                row.update(status='COMPLETE',metrics=m,tuck_trigger_s=e.tuck_time)
                np.savez_compressed(dest/f'{i:03d}.npz',physics=np.asarray(e.rows),qpos=np.asarray(e.qpos_frames),extra=np.asarray(e.extra),
                    heading=heading,head_physics=np.asarray(e.head_physics),obs=np.asarray(e.demo_obs),actions=np.asarray(e.demo_actions))
                print(json.dumps(dict(index=i,depth=depth,tuck=strength,gap_mm=m['both_foot_peak_gap_m']*1000,air5_ms=m['continuous_5mm_s']*1000,
                    feet=m['feet_goal'],complete=m['complete_deep_tuck_jump'],limit=m['joint_limit_excess_rad'],settled=m['settled_final_second'])),flush=True)
            except Exception as exc:row.update(status='FAILED',error=repr(exc));print(json.dumps(row),flush=True)
            records.append(row);(dest/f'{i:03d}.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
            (dest/'matrix.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':main()
