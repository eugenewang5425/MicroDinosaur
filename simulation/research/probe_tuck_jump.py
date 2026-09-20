"""Bounded deep-crouch / push / airborne tuck demonstrations on the frozen plant."""
import argparse
from dataclasses import asdict,dataclass
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
from scipy.optimize import least_squares

from evaluate_jump_refine import RefineExperiment
from evaluate_run_jump import score
from hardware_sim import HardwareCase
from head_attitude import rotation_vector
from imu_owned_head import configure_owned
from run_heading_stable_start import CALIBRATOR

OUT=Path(__file__).parent/'20260914_foot_flight'


def feet_metrics(a,dt):
    a=np.asarray(a);gap=np.min(a[:,14:16],axis=1)
    safe=(a[:,4]<.05)&(a[:,8]<30)&(a[:,13]<=.02)
    segments={}
    for threshold in [.002,.005,.010]:
        mask=safe&(gap>threshold)
        edges=np.diff(np.r_[False,mask,False].astype(int));runs=[]
        for i,j in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            runs.append(dict(start_s=float(a[i,0]),end_s=float(a[j-1,0]+dt),duration_s=(j-i)*dt,
                peak_both_gap_m=float(gap[i:j].max())))
        segments[str(threshold)]=runs
    events=segments['0.005']
    return dict(both_foot_peak_gap_m=float(gap.max()),
        continuous_2mm_s=max((r['duration_s'] for r in segments['0.002']),default=0.),
        continuous_5mm_s=max((r['duration_s'] for r in events),default=0.),
        continuous_10mm_s=max((r['duration_s'] for r in segments['0.01']),default=0.),
        feet_goal=any(r['duration_s']>=.06-1e-8 and r['peak_both_gap_m']>=.010 for r in events),
        feet_segments=segments)


def solve_targets(sim):
    m=sim.model;d=mujoco.MjData(m);d.qpos[:]=sim.data.qpos;d.qpos[sim.jadr]=sim.home
    # The existing contract defines the HOME free-joint placement.
    mujoco.mj_forward(m,d);initial=d.qpos.copy()
    feet=[m.site('robot/'+s+'_foot').id for s in ['left','right']]
    position=d.site_xpos[feet].copy();rotation=d.site_xmat[feet].reshape(2,3,3).copy()
    legs=np.array([i for i,n in enumerate(sim.names) if n.startswith(('left_','right_'))])
    addresses=np.asarray(sim.jadr)[legs]
    limits=m.jnt_range[np.asarray(sim.jids)[legs]];rows=[]
    for depth in [-.02,-.01,0.,.03,.035,.04,.045,.05]:
        d.qpos[:]=initial;d.qpos[2]-=depth
        def residual(q):
            d.qpos[addresses]=q;mujoco.mj_forward(m,d)
            R=d.site_xmat[feet].reshape(2,3,3)
            return np.r_[(d.site_xpos[feet]-position).ravel()/.05,
                          rotation_vector(rotation[0].T@R[0]),rotation_vector(rotation[1].T@R[1])]
        result=least_squares(residual,sim.home[legs],bounds=(limits[:,0]+.05236,limits[:,1]-.05236),
            ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=350)
        residual(result.x);q=sim.home.copy();q[legs]=result.x
        rows.append(dict(depth_m=depth,target=q.tolist(),foot_error_mm=float(np.linalg.norm(d.site_xpos[feet]-position,axis=1).max()*1000),
            minimum_leg_margin_rad=float(np.minimum(result.x-limits[:,0],limits[:,1]-result.x).min())))
    return dict(action_names=sim.names,rows=rows)


@dataclass
class Recipe:
    depth:float=.045
    extension:float=.02
    preparation:float=1.8
    push_s:float=.12
    tuck_s:float=.06
    tuck_hold_s:float=.04
    untuck_s:float=.10
    tuck_depth:float=.05
    enable_tuck:bool=True
    timed_tuck_s:float=0.


class TeacherSession:
    def __init__(self,e,base):self.e=e;self.base=base
    def get_inputs(self):return self.base.get_inputs()
    def run(self,outputs,feed):
        result=self.base.run(outputs,feed)[0].copy();e=self.e
        if not e.active:return [result]
        t=e.current_time;r=e.script_recipe
        if t>=r.preparation+.6:
            e.demo_obs.append(next(iter(feed.values()))[0].copy());e.demo_actions.append(result[0].copy())
            return [result]
        if t<r.preparation:
            f=np.clip(t/1.3,0,1);f=f*f*(3-2*f);target=e.sim.home+(e.crouch-e.sim.home)*f
        else:
            if e.takeoff_time is None and e.rows and t<r.preparation+.6:
                recent=np.asarray(e.rows)[-4:]
                if np.all(recent[:,4]<.05) and np.min(recent[-1,14:16])>0:e.takeoff_time=t
            elapsed=t-r.preparation
            if r.timed_tuck_s>0 and elapsed>=r.timed_tuck_s and e.takeoff_time is None:
                e.takeoff_time=t
            target=e.crouch+(e.extend-e.crouch)*np.clip(elapsed/r.push_s,0,1)
            if r.enable_tuck and e.takeoff_time is not None:
                age=t-e.takeoff_time
                if age<r.tuck_s:target=e.extend+(e.tuck-e.extend)*np.clip(age/r.tuck_s,0,1)
                elif age<r.tuck_s+r.tuck_hold_s:target=e.tuck
                else:target=e.tuck+(e.sim.home-e.tuck)*np.clip((age-r.tuck_s-r.tuck_hold_s)/r.untuck_s,0,1)
            elif elapsed>.40:target=e.sim.home
        result[0,e.legs]=((target-e.sim.home)/e.sim.scale)[e.legs]
        e.demo_obs.append(next(iter(feed.values()))[0].copy());e.demo_actions.append(result[0].copy())
        return [result]


class TuckExperiment(RefineExperiment):
    def __init__(self,case,recipe,targets):
        super().__init__(CALIBRATOR,case,False)
        self.script_recipe=recipe;self.current_time=0.;self.takeoff_time=None;self.demo_obs=[];self.demo_actions=[]
        table={round(r['depth_m'],6):np.array(r['target']) for r in targets['rows']}
        self.crouch=table[recipe.depth];self.extend=table[-recipe.extension];self.tuck=table[recipe.tuck_depth]
        self.legs=np.array([i for i,n in enumerate(self.sim.names) if n.startswith(('left_','right_'))])
        self.jump_session=TeacherSession(self,self.jump_session);self.sim.session=self.jump_session
    def user_command(self,t,scenario):
        self.current_time=t;r=self.script_recipe;command=np.zeros(18)
        command[9]=-r.depth if t<r.preparation else (r.extension if t<r.preparation+.4 else 0.)
        return command


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['geometry','scan','timed'],required=True);args=p.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.stage=='geometry':
        e=RefineExperiment(CALIBRATOR,HardwareCase(physics_dt=.00125,motor_curve=True,voltage=12.),False)
        targets=solve_targets(e.sim);(OUT/'deep_targets.json').write_text(json.dumps(targets,indent=2),encoding='utf-8')
        print(json.dumps([{k:v for k,v in r.items() if k!='target'} for r in targets['rows']],indent=2));return
    targets=json.loads((OUT/'deep_targets.json').read_text());dest=OUT/('script_timed' if args.stage=='timed' else 'script_probes');dest.mkdir(exist_ok=False)
    (dest/'source_snapshot.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    recipes=[Recipe(depth=d,push_s=push,tuck_s=tuck) for d in [.04,.045,.05] for push in [.08,.16] for tuck in [.04,.08]]
    recipes += [Recipe(depth=d,enable_tuck=False) for d in [.04,.045,.05]]
    if args.stage=='timed':
        recipes=[Recipe(depth=depth,extension=extend,push_s=.08,tuck_s=.04,tuck_hold_s=.02,untuck_s=.10,timed_tuck_s=delay)
            for depth in [.045,.05] for extend in [0.,.01] for delay in [.16,.24,.32]]
    records=[]
    for i,recipe in enumerate(recipes):
        e=configure_owned(TuckExperiment(HardwareCase(physics_dt=.00125,motor_curve=True,voltage=12.),recipe,targets),(1,))
        row=dict(index=i,recipe=asdict(recipe),seed=501,hardware=asdict(e.sim.case),controller='scripted demonstration; '+('fixed tuck timing' if recipe.timed_tuck_s else 'oracle contact timing')+'; not a learned policy')
        try:
            m,heading=e.run('stand','imu',501,6.,True);score(e,m);m.update(feet_metrics(e.rows,e.sim.model.opt.timestep))
            a=np.asarray(e.rows);pre=a[(a[:,0]>=1.3)&(a[:,0]<1.8)]
            m.update(crouch_root_z_m=float(pre[:,1].mean()),crouch_tilt_deg=float(pre[:,8].max()),
                crouch_joint_excess_rad=float(pre[:,13].max()),crouch_both_supported_fraction=float(((pre[:,5]>.1)&(pre[:,6]>.1)).mean()))
            row.update(status='COMPLETE',metrics=m,tuck_trigger_s=e.takeoff_time)
            np.savez_compressed(dest/f'{i:03d}.npz',physics=a,extra=np.asarray(e.extra),qpos=np.asarray(e.qpos_frames),
                heading=heading,head_physics=np.asarray(e.head_physics),obs=np.asarray(e.demo_obs),actions=np.asarray(e.demo_actions))
            print(json.dumps(dict(index=i,depth=recipe.depth,gap_mm=m['both_foot_peak_gap_m']*1000,air5_ms=m['continuous_5mm_s']*1000,
                feet=m['feet_goal'],settled=m['settled_final_second'],limit=m['joint_limit_excess_rad'],fell=m['fell'])),flush=True)
        except Exception as exc:row.update(status='FAILED',error=repr(exc));print(json.dumps(row),flush=True)
        records.append(row);(dest/f'{i:03d}.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        (dest/'matrix.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':main()
