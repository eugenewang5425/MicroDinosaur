"""Check stepping and contact motion during commanded turns, at 50 Hz."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from terrain_skill_eval import TerrainSkillExperiment, OUT
from heading_sim import WARMUP_SECONDS
from evaluate_policy import sha


def run(policy, scenario, seed):
    e=TerrainSkillExperiment(policy); s=e.sim; rows=[]; original=s.step
    vertices={}
    for g in s.foot_geoms:
        assert s.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
        mid=s.model.geom_dataid[g]; start=s.model.mesh_vertadr[mid]
        vertices[g]=s.model.mesh_vert[start:start+s.model.mesh_vertnum[mid]].copy()
    def step(command,*args,**kwargs):
        original(command,*args,**kwargs)
        if not e.active: return
        height=[]; contact=[]; slip=[]; yaw=[]
        for g in s.foot_geoms:
            rotation=s.data.geom_xmat[g].reshape(3,3)
            pts=s.data.geom_xpos[g]+vertices[g]@rotation.T
            height.append(float(pts[:,2].min()))
            velocity=np.zeros(6)
            mujoco.mj_objectVelocity(s.model,s.data,mujoco.mjtObj.mjOBJ_GEOM,g,velocity,0)
            speeds=[]
            for index,c in enumerate(s.data.contact):
                if g not in (c.geom1,c.geom2) or c.efc_address<0: continue
                force=np.zeros(6); mujoco.mj_contactForce(s.model,s.data,index,force)
                if force[0]>.05:
                    v=velocity[3:]+np.cross(velocity[:3],c.pos-s.data.geom_xpos[g])
                    speeds.append(float(np.linalg.norm(v[:2])))
            contact.append(bool(speeds)); slip.append(float(np.mean(speeds)) if speeds else 0.)
            yaw.append(float(abs(velocity[2])) if speeds else 0.)
        rows.append([s.data.time-WARMUP_SECONDS,*height,*contact,*slip,*yaw])
    s.step=step
    metrics,_=e.run(scenario,'imu',seed,12.,True)
    a=np.array(rows); w=a[(a[:,0]>=2)&(a[:,0]<5)]
    m=dict(scenario=scenario,seed=seed,fell=metrics['fell'],
        body_turn_deg=metrics['reference_final_deg']+metrics['heading_error_final_deg'],
        reference_deg=metrics['reference_final_deg'],
        heading_error_final_deg=metrics['heading_error_final_deg'],
        camera_heading_error_rms_deg=metrics['camera_heading_error_rms_deg'],
        body_vx_mean_m_s=metrics['body_vx_mean_m_s'],sample_hz=50)
    m['swing_starts_during_turn']=[int(np.sum(np.diff(w[:,3+i])<0)) for i in range(2)]
    m['peak_sole_clearance_during_turn_mm']=(w[:,1:3].max(0)*1000).tolist()
    m['contact_slip_mean_during_turn_m_s']=[float(w[w[:,3+i]>0,5+i].mean()) for i in range(2)]
    m['contact_yaw_rate_mean_during_turn_deg_s']=[float(np.rad2deg(w[w[:,3+i]>0,7+i].mean())) for i in range(2)]
    return m,a


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',required=True,type=Path);p.add_argument('--label',required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();dest=OUT/'turns'/args.label;dest.mkdir(parents=True,exist_ok=args.resume)
    records=[]
    for scenario in ('stand','left_then_hold','right_then_hold'):
        for seed in (1,2,3):
            name=f'{scenario}__s{seed}'
            if (dest/(name+'.json')).exists():
                old=json.loads((dest/(name+'.json')).read_text())
                assert args.resume and old['policy_sha256']==sha(args.policy)
                records.append(old);continue
            m,a=run(args.policy,scenario,seed)
            r=dict(metrics=m,policy_sha256=sha(args.policy));records.append(r)
            (dest/(name+'.json')).write_text(json.dumps(r,indent=2),encoding='utf-8')
            np.savez_compressed(dest/(name+'.npz'),feet=a)
            print(json.dumps(m),flush=True)
    (dest/'matrix.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':main()
