"""Bounded teacher probe: hips/ankles follow delayed knee progress through the IK curve."""
import json
from pathlib import Path
import sys
import numpy as np
from evaluate_foot_flight import assess
from probe_tuck_jump import TuckExperiment,Recipe,OUT
from hardware_sim import HardwareCase
from imu_owned_head import configure_owned


class SynchronizedSession:
    def __init__(self,e,base,extension,push,lead,table):
        self.e=e;self.base=base;self.extension=extension;self.push=push;self.lead=lead
        self.curve=np.asarray([r['target'] for r in table['rows'] if -.01<=r['depth_m']<=.05])
    def get_inputs(self):return self.base.get_inputs()
    def run(self,outputs,feed):
        e=self.e;action=self.base.run(outputs,feed)[0].copy()
        if not e.active:return [action]
        t=e.current_time;phase=t-1.8
        target=e.sim.home+action[0]*e.sim.scale
        if t<1.8:
            f=np.clip(t/1.3,0,1);f=f*f*(3-2*f)
            target[e.legs]=(e.sim.home+(e.crouch-e.sim.home)*f)[e.legs]
        elif phase<self.push+.12:
            measured=e.sim.home+next(iter(feed.values()))[0,6:25]
            retract=phase>=self.push
            goal=e.tuck if retract else self.extension
            for side,sign in [('left',1),('right',-1)]:
                ids=[e.sim.names.index(side+'_'+n) for n in ['hip_yaw','hip_roll','hip_pitch','knee','ankle']]
                knee=ids[3];x=self.curve[:,knee]*sign;order=np.argsort(x)
                progress=measured[knee]*sign+(self.lead if retract else -self.lead)
                for j in ids:target[j]=np.interp(progress,x[order],self.curve[order,j])
                target[knee]=goal[knee]
        action[0]=(target-e.sim.home)/e.sim.scale
        e.demo_obs.append(next(iter(feed.values()))[0].copy());e.demo_actions.append(action[0].copy())
        return [action]


def main():
    sys.stdout.reconfigure(encoding='utf-8');dest=OUT/'synchronized_probes';dest.mkdir(exist_ok=False)
    (dest/'source_snapshot.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    table=json.loads((OUT/'deep_targets.json').read_text());targets={round(r['depth_m'],6):np.array(r['target']) for r in table['rows']};records=[]
    for extend in [0.,.01]:
        for push in [.20,.30,.40]:
            for lead in [.05,.20]:
                e=configure_owned(TuckExperiment(HardwareCase(physics_dt=.00125,motor_curve=True,voltage=12.),Recipe(depth=.05),table),(1,))
                e.depth=.05;e.jump_session=SynchronizedSession(e,e.jump_session.base,targets[-extend],push,lead,table);e.sim.session=e.jump_session
                i=len(records);row=dict(index=i,depth_m=.05,extension_m=extend,push_s=push,knee_lead_rad=lead,seed=501,
                    controller='Scripted knee-synchronized targets using delayed observation, inherited S288 limits and head IMU; not trained actor')
                try:
                    m,h=e.run('stand','imu',501,6.,True);assess(e,m);row.update(status='COMPLETE',metrics=m)
                    np.savez_compressed(dest/f'{i:03d}.npz',physics=np.asarray(e.rows),extra=np.asarray(e.extra),qpos=np.asarray(e.qpos_frames),
                        heading=h,obs=np.asarray(e.demo_obs),actions=np.asarray(e.demo_actions))
                    print(json.dumps(dict(index=i,extension=extend,push=push,lead=lead,gap_mm=m['both_foot_peak_gap_m']*1000,
                        air5_ms=m['continuous_5mm_s']*1000,complete=m['complete_deep_tuck_jump'],limit=m['joint_limit_excess_rad'],fell=m['fell'])),flush=True)
                except Exception as exc:row.update(status='FAILED',error=repr(exc));print(json.dumps(row),flush=True)
                records.append(row);(dest/f'{i:03d}.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
                (dest/'matrix.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':main()
