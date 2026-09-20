"""Read-only paired replay: actual output torque, PD terms, power and jaw filtering."""
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
from evaluate_foot_flight import FeetExperiment
from hardware_sim import HardwareCase, torque_bounds
from imu_owned_head import configure_owned
from evaluate_policy import sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT / '20260914_jump_dynamics_review'
ARCHIVE = ROOT / '20260914_foot_flight/evaluation'


def model_audit(e):
    s = e.sim; m = s.model
    rows = []
    for name, aid, jid, vid in zip(s.names, s.aids, s.jids, s.vadr):
        rows.append(dict(name=name, gear=m.actuator_gear[aid].tolist(),
            kp=float(m.actuator_gainprm[aid, 0]), bias=m.actuator_biasprm[aid, :3].tolist(),
            force_limited=bool(m.actuator_forcelimited[aid]), base_force_range=s.base_force_ranges[aid].tolist(),
            ctrl_limited=bool(m.actuator_ctrllimited[aid]), ctrl_range=m.actuator_ctrlrange[aid].tolist(),
            joint_force_limited=bool(m.jnt_actfrclimited[jid]),
            joint_force_range=m.jnt_actfrcrange[jid].tolist(),
            passive_damping=float(m.dof_damping[vid]), frictionloss=float(m.dof_frictionloss[vid]),
            armature=float(m.dof_armature[vid])))
    groups = {}; bodies = {}
    for name in ['jaw_soft', 'jaw_hinge']:
        bid = m.body('robot/' + name).id
        gids = np.flatnonzero(m.geom_bodyid == bid)
        groups[name] = gids
        bodies[name] = dict(id=bid, parent=int(m.body_parentid[bid]), weld=int(m.body_weldid[bid]),
            parent_weld=int(m.body_weldid[m.body_parentid[bid]]),
            geoms=[dict(id=int(g), name=m.geom(g).name, type=int(m.geom_type[g]),
                group=int(m.geom_group[g]), contype=int(m.geom_contype[g]),
                conaffinity=int(m.geom_conaffinity[g])) for g in gids])
    compatible = [(int(a), int(b)) for a in groups['jaw_soft'] for b in groups['jaw_hinge']
        if (m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a])]
    return dict(mass_kg=float(m.body_subtreemass[s.body]), policy_dt=s.dt, physics_dt=float(m.opt.timestep),
        actuators=rows, jaw_bodies=bodies, jaw_mask_compatible_pairs=compatible,
        explicit_pairs=[dict(geom1=int(a), geom2=int(b)) for a,b in zip(m.pair_geom1,m.pair_geom2)],
        parent_filter_enabled=not bool(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT),
        s288_assumed_bounds_12v={str(v):list(map(float, torque_bounds(v, 12.))) for v in [0., 4., 8., 16.5]},
        active_curve_identified_from_hardware=False)


def summarize(e, physics, data):
    dt=e.sim.model.opt.timestep; t=physics[:,0]; vel=data['qvel_pre']; tau=data['joint_actuator_torque']
    power=tau*vel
    major=[i for i,n in enumerate(e.sim.names) if n.startswith(('left_', 'right_')) and n.endswith(('hip_pitch','knee','ankle'))]
    unsupported=(physics[:,4]<.05)&(t>=1.8)&(t<2.8)&(physics[:,8]<30)
    edges=np.diff(np.r_[False,unsupported,False].astype(int))
    runs=[(int(a),int(b)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if (b-a)*dt>=.02]
    takeoff=runs[0][0] if runs else None
    end=float(t[takeoff]) if takeoff is not None else 2.1
    metrics={}
    for label,start in [('push',1.8),('last40ms',max(1.8,end-.04))]:
        mask=(t>=start)&(t<end)
        p=power[mask][:,major]; v=vel[mask][:,major]; tr=tau[mask][:,major]
        metrics[label]=dict(start_s=start,end_s=end,ends_at_takeoff=takeoff is not None,
            positive_work_j=float(np.maximum(p,0).sum()*dt), negative_work_j=float(-np.minimum(p,0).sum()*dt),
            net_negative_power_time_fraction=float((p.sum(-1)<0).mean()),
            peak_joint_speed_rad_s=float(abs(v).max()), peak_joint_torque_nm=float(abs(tr).max()),
            mean_net_power_w=float(p.sum(-1).mean()),
            by_joint={e.sim.names[i]:dict(positive_work_j=float(np.maximum(power[mask,i],0).sum()*dt),
                negative_work_j=float(-np.minimum(power[mask,i],0).sum()*dt),
                mean_pd_position_torque_nm=float(data['p_term'][mask,i].mean()),
                mean_pd_damping_torque_nm=float(data['d_term'][mask,i].mean()),
                mean_actual_torque_nm=float(tau[mask,i].mean())) for i in major})
    if runs:
        a,b=runs[0];z=physics[a:b,3];v=data['com_velocity'][a,2]
        metrics['flight']=dict(takeoff_s=float(t[a]), support_free_duration_ms=(b-a)*dt*1000,
            direct_com_takeoff_velocity_m_s=float(v), com_rise_mm=float((z.max()-z[0])*1000),
            ballistic_from_velocity_mm=float(max(0.,v)**2/(2*9.81)*1000),
            peak_both_feet_mm=float(physics[a:b,14:16].min(-1).max()*1000))
    else:metrics['flight']=None
    metrics['pd_force_reconstruction_max_error_nm']=float(abs(data['predicted_clipped_torque']-data['actuator_force']).max())
    metrics['actuator_vs_joint_force_max_error_nm']=float(abs(tau-data['actuator_force']).max())
    metrics['jaw_contact_count_max']=int(data['jaw_contact_count'].max())
    metrics['jaw_range_rad']=[float(data['qpos_pre'][:,e.sim.names.index('jaw_hinge')].min()),
                              float(data['qpos_pre'][:,e.sim.names.index('jaw_hinge')].max())]
    return metrics


def replay(label,depth):
    name=f'c{depth}_d10_s601_s288_protocol_return0'
    source=ARCHIVE/label/(name+'.json'); record=json.loads(source.read_text())
    e=configure_owned(FeetExperiment(Path(record['policy']),HardwareCase(**record['hardware']),False,depth/1000),(1,))
    audit=model_audit(e); cache={}; captured=[]; before=e.sim.before_physics_step; after=e.sim.substep_callback
    s=e.sim; m=s.model
    upper=m.body('robot/jaw_soft').id; lower=m.body('robot/jaw_hinge').id
    def pre(target):
        result=before(target)
        cache['q']=s.data.qpos[s.jadr].copy();cache['v']=s.data.qvel[s.vadr].copy()
        cache['target']=result.copy()
        return result
    def post():
        after()
        if not e.active:return
        d=s.data
        p=m.actuator_gainprm[s.aids,0]*cache['target']+m.actuator_biasprm[s.aids,0]+m.actuator_biasprm[s.aids,1]*d.actuator_length[s.aids]
        damp=m.actuator_biasprm[s.aids,2]*d.actuator_velocity[s.aids]
        lo=m.actuator_forcerange[s.aids,0];hi=m.actuator_forcerange[s.aids,1]
        # mj_subtreeVel only refreshes a derived velocity field, never dynamics.
        mujoco.mj_subtreeVel(m,d)
        nc=sum({int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])}=={upper,lower} for c in d.contact)
        captured.append(dict(qpos_pre=cache['q'],qvel_pre=cache['v'],target=cache['target'],
            actuator_force=d.actuator_force[s.aids].copy(),joint_actuator_torque=d.qfrc_actuator[s.vadr].copy(),
            p_term=p,d_term=damp,predicted_clipped_torque=np.clip(p+damp,lo,hi),
            bounds=np.stack([lo,hi],-1).copy(),passive_force=d.qfrc_passive[s.vadr].copy(),
            constraint_force=d.qfrc_constraint[s.vadr].copy(),com_velocity=d.subtree_linvel[s.body].copy(),
            jaw_contact_count=nc))
    s.before_physics_step=pre;s.substep_callback=post
    e.run('stand','imu',601,6.,True)
    physics=np.asarray(e.rows); original=np.load(source.with_suffix('.npz'))['physics']
    error=float(abs(physics-original).max());assert error==0,error
    data={key:np.array([r[key] for r in captured]) for key in captured[0]}
    assert len(physics)==len(captured)
    metrics=summarize(e,physics,data)
    assert metrics['actuator_vs_joint_force_max_error_nm']<1e-10
    assert metrics['pd_force_reconstruction_max_error_nm']<1e-7
    dest=OUT/f'{label}_c{depth}'
    np.savez_compressed(dest.with_suffix('.npz'),physics=physics,**data)
    result=dict(source=str(source),source_sha256=sha(source),policy_sha256=record['policy_sha256'],
        model=audit,replay_max_error=error,metrics=metrics,
        interpretation='Observed negative actuator work is not proof of avoidable pre-braking; requires a phase-controlled intervention.')
    dest.with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(label=label,depth=depth,replay_error=error,metrics=metrics),ensure_ascii=False),flush=True)
    return result


def main():
    sys.stdout.reconfigure(encoding='utf-8');OUT.mkdir(exist_ok=True)
    results={f'{label}_c{depth}':replay(label,depth) for depth in [30,50] for label in ['previous','candidate']}
    (OUT/'audit.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()
