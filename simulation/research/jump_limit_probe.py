"""Frozen-policy command sweep: empirical ceiling, flight energetics, no training."""
import argparse
from dataclasses import dataclass,asdict
import itertools
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from evaluate_jump_return import ReturningJump,JUMP
from evaluate_run_jump import score
from evaluate_policy import sha
from hardware_sim import HardwareCase
from imu_owned_head import configure_owned

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_jump_limit'


@dataclass(frozen=True)
class Recipe:
    crouch:float=.03
    extension:float=.02
    preparation_s:float=1.5
    extension_s:float=.7


class Probe(ReturningJump):
    def __init__(self,case,recipe):
        self.recipe=recipe
        super().__init__(case)
        self.extra=[]
        previous=self.sim.substep_callback
        def capture():
            previous()
            if not self.active:return
            d=self.sim.data;m=self.sim.model
            mujoco.mj_energyPos(m,d);mujoco.mj_energyVel(m,d)
            self.extra.append(np.r_[d.energy,d.qpos[self.sim.jadr],
                                    d.qvel[self.sim.vadr],d.actuator_force[self.sim.aids]])
        self.sim.substep_callback=capture

    def reset(self,seed=0):
        self.extra=[]
        return super().reset(seed)

    def user_command(self,t,scenario):
        command=super().user_command(t,scenario)
        r=self.recipe
        command[9]=-r.crouch if t<r.preparation_s else (r.extension if t<r.preparation_s+r.extension_s else 0.)
        return command


def additional_metrics(e,m):
    a=np.asarray(e.rows);extra=np.asarray(e.extra);dt=e.sim.model.opt.timestep
    mass=float(e.sim.model.body_subtreemass[e.sim.body]);gravity=float(abs(e.sim.model.opt.gravity[2]))
    assert len(a)==len(extra) and np.isfinite(extra).all()
    valid=m['qualified_flight_segments']
    best=max(valid,key=lambda s:s['com_apex_above_takeoff_m'],default=None)
    m.update(max_qualified_com_height_m=best['com_apex_above_takeoff_m'] if best else 0.,
             maximum_both_foot_clearance_m=float(np.min(a[:,14:16],axis=1).max()),
             maximum_mesh_floor_penetration_m=float(max(0.,-a[:,14:16].min())),
             mass_kg=mass)
    if not best:return
    takeoff=int(np.argmin(abs(a[:,0]-best['support_free_start_s'])))
    end=takeoff+round(best['support_free_duration_s']/dt)
    end=min(end,len(a))
    # Start of the commanded push; report existing kinetic energy separately.
    launch=int(np.argmin(abs(a[:,0]-e.recipe.preparation_s)))
    if launch>=takeoff:return
    velocity=extra[:,21:40];tau=extra[:,40:59];power=tau*velocity
    positive=np.maximum(power[launch:takeoff],0).sum(axis=0)*dt
    negative=-np.minimum(power[launch:takeoff],0).sum(axis=0)*dt
    v=best['takeoff_com_velocity_m_s'];height=best['com_apex_above_takeoff_m']
    t=a[takeoff:end,0]-a[takeoff,0];z=a[takeoff:end,3]
    fit=np.polyfit(t,z,2) if len(t)>=3 else [np.nan]*3
    q=extra[launch:takeoff,2:21]
    joint_range=np.ptp(q,axis=0)
    armature=e.sim.model.dof_armature[e.sim.vadr]
    rotor_energy=.5*np.sum(armature*velocity[takeoff]**2)
    # This loose ideal allocation assumes actuator work could be redirected
    # before release. Internal kinetic energy cannot raise COM once airborne.
    available=extra[launch,1]+positive.sum()-(extra[takeoff,0]-extra[launch,0])
    m['energetics']=dict(takeoff_s=float(a[takeoff,0]),push_duration_s=float(a[takeoff,0]-a[launch,0]),
        com_takeoff_velocity_m_s=v,ballistic_height_m=v*v/(2*gravity),measured_com_height_m=height,
        vertical_takeoff_energy_j=.5*mass*v*v,vertical_impulse_from_rest_ns=mass*v,
        measured_apex_potential_gain_j=mass*gravity*height,
        fitted_airborne_acceleration_m_s2=float(2*fit[0]),
        kinetic_at_command_j=float(extra[launch,1]),kinetic_at_takeoff_j=float(extra[takeoff,1]),
        reflected_rotor_kinetic_at_takeoff_j=float(rotor_energy),
        gravity_potential_change_in_push_j=float(extra[takeoff,0]-extra[launch,0]),
        actuator_positive_work_push_j=float(positive.sum()),actuator_negative_work_push_j=float(negative.sum()),
        ideal_same_push_work_height_m=float(max(0.,available)/(mass*gravity)),
        ideal_work_bound_is_not_an_achievable_or_global_limit=True,
        actuator_work_by_joint={name:dict(positive_j=float(positive[i]),negative_j=float(negative[i]),
            excursion_rad=float(joint_range[i]),peak_abs_torque_nm=float(abs(tau[launch:takeoff,i]).max()),
            peak_abs_speed_rad_s=float(abs(velocity[launch:takeoff,i]).max())) for i,name in enumerate(e.sim.names)})


def run_case(dest,index,recipe,case,seed):
    name=f'{index:03d}_c{recipe.crouch:.3f}_e{recipe.extension:.3f}_p{recipe.preparation_s:.1f}_x{recipe.extension_s:.1f}_s{seed}_d{case.command_ms}_{case.name}'
    r=dict(name=name,recipe=asdict(recipe),hardware=asdict(case),seed=seed,policy_sha256=sha(JUMP))
    try:
        e=configure_owned(Probe(case,recipe),(1,))
        m,heading=e.run('stand','imu',seed,6.,True);score(e,m);additional_metrics(e,m)
        r.update(status='COMPLETE',metrics=m,handoff=e.handoff)
        np.savez_compressed(dest/(name+'.npz'),physics=np.asarray(e.rows),extra=np.asarray(e.extra),
                            qpos=np.asarray(e.qpos_frames),heading=heading)
        print(json.dumps(dict(name=name,height_mm=1000*m['max_qualified_com_height_m'],
                              success=m['jump_success'],excess=m['joint_limit_excess_rad'],
                              penetration_mm=1000*m['maximum_mesh_floor_penetration_m'])),flush=True)
    except Exception as exc:
        r.update(status='FAILED',error=repr(exc));print(json.dumps(r),flush=True)
    (dest/(name+'.json')).write_text(json.dumps(r,indent=2),encoding='utf-8')
    return r


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['scan','confirm'],required=True);args=p.parse_args()
    dest=OUT/args.stage;dest.mkdir(parents=True,exist_ok=False)
    jobs=[]
    if args.stage=='scan':
        jobs=[(Recipe(c,e),HardwareCase(physics_dt=.00125),101)
              for c,e in itertools.product([.02,.025,.03,.035,.04],[0.,.01,.02,.03,.04])]
    else:
        scanned=json.loads((OUT/'scan/matrix.json').read_text())
        good=[r for r in scanned if r['status']=='COMPLETE' and r['metrics']['jump_success']]
        assert good,'No nominal valid candidate in sweep'
        best=max(good,key=lambda r:r['metrics']['max_qualified_com_height_m'])
        recipe=Recipe(**best['recipe'])
        (OUT/'selected.json').write_text(json.dumps(best,indent=2),encoding='utf-8')
        jobs=[(recipe,HardwareCase(physics_dt=.00125,command_ms=delay),seed)
              for delay,seed in itertools.product([5,10,15],[301,302,303])]
        for c in [HardwareCase(name='fine_dt',physics_dt=.000625),
                  HardwareCase(name='derate80',physics_dt=.00125,effort_fraction=.8),
                  *[HardwareCase(name='curve'+str(v),physics_dt=.00125,motor_curve=True,voltage=v) for v in [12.6,11.1,9.9]]]:
            jobs.append((recipe,c,304))
    rows=[]
    for recipe,case,seed in jobs:
        rows.append(run_case(dest,len(rows),recipe,case,seed))
        (dest/'matrix.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    if args.stage=='scan':
        good=[r for r in rows if r['status']=='COMPLETE' and r['metrics']['jump_success']]
        leaders=sorted(good,key=lambda r:r['metrics']['max_qualified_com_height_m'],reverse=True)[:3]
        for leader in leaders:
            for preparation,extension in itertools.product([1.,2.],[.4,1.]):
                recipe=Recipe(**{**leader['recipe'],'preparation_s':preparation,'extension_s':extension})
                # Preserve fixed handoff at 2.8: no raised command after return.
                if preparation+extension>2.8:continue
                rows.append(run_case(dest,len(rows),recipe,HardwareCase(physics_dt=.00125),101))
                (dest/'matrix.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':main()
