"""Current-v07 bounded fold/recovery feasibility, independent of RL promotion."""
from collections import deque
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from mjlab.scene import Scene
from terrain_skill_cfg import build_config, XML
from head_attitude import rotation_vector
from imu_heading import from_rpy, matrix
from mjlab_microduck.s288_protocol import ROTOR_POSITION_OUTPUT_STEP
from evaluate_policy import sha

ROOT=Path(__file__).parent
OUT=ROOT/'20260914_fold_recovery'


def build_floor_model():
    dest=OUT/'mesh_floor.mjb'
    if dest.exists():return dest
    _,cfg=build_config('crouch',1);scene=Scene(cfg.env.scene,device='cpu');before=scene.compile()
    added=[]
    for g in list(scene.spec.geoms):
        if g.type!=mujoco.mjtGeom.mjGEOM_MESH or g.contype or g.conaffinity:continue
        if g.parent.name.endswith(('ankle_left','ankle_right')):continue
        name=g.parent.name+'_mesh_floor'
        scene.spec.body(g.parent.name).add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=g.meshname,pos=g.pos.copy(),quat=g.quat.copy(),contype=4,conaffinity=8,
            group=3,friction=[.6,.005,.0001],condim=3,rgba=[.2,.6,.5,.3])
        added.append(name)
    for g in scene.spec.geoms:
        if g.name.endswith('_mesh_floor'):continue
        if g.name=='terrain':g.contype=8;g.conaffinity=2|4
        elif g.contype or g.conaffinity:g.contype=2;g.conaffinity=2|8
    scene.spec.option.timestep=.00125
    scene.spec.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    scene.spec.option.iterations=30;scene.spec.option.ls_iterations=50
    model=scene.compile();np.testing.assert_array_equal(model.body_mass,before.body_mass)
    np.testing.assert_array_equal(model.body_inertia,before.body_inertia)
    buffer=np.empty(mujoco.mj_sizeModel(model),np.uint8);mujoco.mj_saveModel(model,buffer=buffer)
    dest.write_bytes(buffer.tobytes())
    (OUT/'model_audit.json').write_text(json.dumps(dict(xml_sha256=sha(XML),model_sha256=sha(dest),
        mujoco_version=mujoco.__version__,robot_mass_kg=float(model.body_mass.sum()),
        floor_meshes=added,mass_and_inertia_unchanged=True,full_self_collision_verified=False,
        body_floor_friction_assumed=.6,mesh_convex_hull_caveat=True),indent=2),encoding='utf-8')
    return dest


class Probe:
    def __init__(self,kind='mesh',command_ms=10,body_friction=.6):
        path=build_floor_model() if kind=='mesh' else ROOT/'20260914_terrain_skills/planned_motion_contacts/candidate_ground_contacts.mjb'
        self.model=mujoco.MjModel.from_binary_path('m.mjb',assets={'m.mjb':path.read_bytes()})
        self.data=mujoco.MjData(self.model);m=self.model
        self.contract=json.loads((ROOT/'20260913_handoff/native_v07/contract.json').read_text())
        self.names=self.contract['action_names'];self.home=np.array(self.contract['action_offset'][0])
        self.jids=np.array([m.joint('robot/'+n).id for n in self.names]);self.adr=m.jnt_qposadr[self.jids]
        self.vadr=m.jnt_dofadr[self.jids]
        self.aids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in self.jids])
        self.limits=m.jnt_range[self.jids].copy();self.leg=np.array([i for i,n in enumerate(self.names) if n.startswith(('left_','right_'))])
        self.feet=[m.site('robot/'+n).id for n in ('left_foot','right_foot')]
        self.trunk=m.body('robot/trunk_base').id;self.terrain=m.body('terrain').id
        self.command_ms=command_ms;self.vertices={};self.visual=[];self.floor_geoms=[]
        for g in range(m.ngeom):
            if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
                mid=m.geom_dataid[g];start=m.mesh_vertadr[mid]
                self.vertices[g]=m.mesh_vert[start:start+m.mesh_vertnum[mid]].copy()
            if m.geom_bodyid[g] and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g]==0:
                self.visual.append(g)
            if m.geom_bodyid[g]!=self.terrain and m.geom_contype[g] and (m.geom_conaffinity[g]&8):
                self.floor_geoms.append(g)
                if 'floor' in m.geom(g).name:m.geom_friction[g,0]=body_friction
        self.reset();self.initial=self.data.qpos.copy()
        self.foot_pos=self.data.site_xpos[self.feet].copy();self.foot_rot=self.data.site_xmat[self.feet].reshape(2,3,3).copy()
        self.pairs=[];self.home_dist={}
        for i,g in enumerate(self.visual):
            b=m.geom_bodyid[g]
            for h in self.visual[i+1:]:
                c=m.geom_bodyid[h]
                if b==c or m.body_parentid[b]==c or m.body_parentid[c]==b:continue
                self.pairs.append((g,h));self.home_dist[(g,h)]=mujoco.mj_geomDistance(m,self.data,g,h,.02,None)

    def reset(self,q=None,quat=None,place=False):
        m,d=self.model,self.data;mujoco.mj_resetData(m,d)
        d.qpos[2]=.117182;d.qpos[3:7]=[1,0,0,0] if quat is None else quat
        d.qpos[self.adr]=self.home if q is None else q;d.ctrl[self.aids]=d.qpos[self.adr]
        mujoco.mj_forward(m,d)
        if place:
            lowest=min(float(self.points(g)[:,2].min()) for g in self.floor_geoms)
            d.qpos[2]+=.002-lowest;mujoco.mj_forward(m,d)

    def points(self,g):
        m,d=self.model,self.data
        if g in self.vertices:v=self.vertices[g]
        else:v=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*m.geom_size[g]
        return v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]

    def knee_angles(self):
        result={}
        for side in ('left','right'):
            js=[self.model.joint('robot/'+side+'_'+n).id for n in ('hip_pitch','knee','ankle')]
            p=self.data.xanchor[js];u=p[0]-p[1];v=p[2]-p[1];axis=self.data.xaxis[js[1]]
            included=lambda a,b:float(np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1))))
            result[side]=dict(spatial_deg=included(u,v),projected_deg=included(u-axis*(u@axis),v-axis*(v@axis)))
        return result

    def collision_suspects(self):
        result=[]
        for g,h in self.pairs:
            dist=mujoco.mj_geomDistance(self.model,self.data,g,h,.02,None);base=self.home_dist[(g,h)]
            if dist<-.002 and dist<base-.002:
                result.append(dict(bodies=[self.model.body(self.model.geom_bodyid[x]).name for x in (g,h)],
                    penetration_mm=-dist*1000,home_penetration_mm=max(0,-base*1000)))
        return sorted(result,key=lambda r:-r['penetration_mm'])

    def solve_depth(self,depth):
        m,d=self.model,self.data;d.qpos[:]=self.initial;d.qpos[2]-=depth
        def residual(q):
            d.qpos[self.adr[self.leg]]=q;mujoco.mj_forward(m,d)
            r=d.site_xmat[self.feet].reshape(2,3,3)
            return np.r_[(d.site_xpos[self.feet]-self.foot_pos).ravel()/.05,
                rotation_vector(self.foot_rot[0].T@r[0]),rotation_vector(self.foot_rot[1].T@r[1])]
        solved=least_squares(residual,self.home[self.leg],bounds=(self.limits[self.leg,0]+np.deg2rad(3),self.limits[self.leg,1]-np.deg2rad(3)),
            max_nfev=500,xtol=1e-11,ftol=1e-11,gtol=1e-11)
        residual(solved.x);target=self.home.copy();target[self.leg]=solved.x
        footerror=np.linalg.norm(d.site_xpos[self.feet]-self.foot_pos,axis=1)*1000
        r=d.site_xmat[self.feet].reshape(2,3,3)
        angleerror=[np.degrees(np.linalg.norm(rotation_vector(self.foot_rot[i].T@r[i]))) for i in range(2)]
        foot_geoms=[m.geom('robot/'+n).id for n in ('left_foot_collision','right_foot_collision')]
        poly=ConvexHull(np.vstack([self.points(g)[:,:2] for g in foot_geoms]));com=d.subtree_com[self.trunk]
        margin=-max(poly.equations[:,:2]@com[:2]+poly.equations[:,2])
        limiting=[self.names[i] for i in self.leg if min(target[i]-self.limits[i,0],self.limits[i,1]-target[i])<np.deg2rad(3.2)]
        return dict(depth_mm=depth*1000,target=target.tolist(),foot_position_error_mm=footerror.tolist(),
            foot_orientation_error_deg=angleerror,knee_angles=self.knee_angles(),
            support_margin_mm=float(margin*1000),limiting_joints=limiting,
            collision_suspects=self.collision_suspects())

    def dynamics(self,label,start_q,start_quat,keyframes,seconds=10,seed=0):
        m,d=self.model,self.data;rng=np.random.default_rng(seed)
        self.reset(start_q,start_quat,place=True)
        d.qpos[0:2]+=rng.uniform(-.003,.003,2);d.qvel[:3]=rng.normal(0,.003,3) if seed else 0
        applied=start_q.copy();transmitted=applied.copy()
        queue=deque([applied.copy() for _ in range(round(self.command_ms/.00125/1000))])
        rows=[];qs=[];contacts=set();max_pen=0.;suspects={}
        ts=np.array([x[0] for x in keyframes]);values=np.array([x[1] for x in keyframes])
        for k in range(round(seconds/.00125)):
            t=k*.00125
            if k%16==0:
                goal=np.array([np.interp(t,ts,values[:,i]) for i in range(19)])
                goal=np.clip(goal,self.limits[:,0]+.03,self.limits[:,1]-.03)
                applied+=np.clip(.9*(goal-applied),-.08,.08)
                transmitted=np.round(applied/ROTOR_POSITION_OUTPUT_STEP)*ROTOR_POSITION_OUTPUT_STEP
            queue.append(transmitted.copy());d.ctrl[self.aids]=queue.popleft();mujoco.mj_step(m,d)
            if k%16:continue
            mujoco.mj_forward(m,d);rot=d.xmat[self.trunk].reshape(3,3)
            tilt=float(np.degrees(np.arccos(np.clip(rot[2,2],-1,1))))
            feet_force=np.zeros(2);body_force=0.;force=np.zeros(6)
            for i,c in enumerate(d.contact):
                if self.terrain not in (m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]):continue
                mujoco.mj_contactForce(m,d,i,force)
                if force[0]<=.05:continue
                other=c.geom1 if m.geom_bodyid[c.geom2]==self.terrain else c.geom2
                name=m.geom(other).name;contacts.add(name);max_pen=max(max_pen,-float(c.dist))
                if name=='robot/left_foot_collision':feet_force[0]+=force[0]
                elif name=='robot/right_foot_collision':feet_force[1]+=force[0]
                else:body_force+=force[0]
            speed=float(np.linalg.norm(d.qvel[:3]));omega=float(np.linalg.norm(d.qvel[3:6]))
            rows.append([t,d.qpos[2],tilt,speed,omega,*feet_force,body_force,
                float(abs(d.actuator_force[self.aids]).max()),float(abs(d.qvel[self.vadr]).max()),*d.subtree_com[self.trunk]])
            qs.append(d.qpos.copy())
            if k%160==0:
                for s in self.collision_suspects():
                    key=tuple(s['bodies'])
                    if key not in suspects or s['penetration_mm']>suspects[key]['penetration_mm']:suspects[key]=s
        a=np.array(rows);steady=a[a[:,0]>=seconds-2]
        upright=(steady[:,1]>.105)&(steady[:,2]<10)&(steady[:,3]<.03)&(steady[:,4]<.3)&(steady[:,5]>.5)&(steady[:,6]>.5)&(steady[:,7]<.2)
        result=dict(label=label,seed=seed,seconds=seconds,command_delay_ms=self.command_ms,
            final_height_mm=float(a[-1,1]*1000),final_tilt_deg=float(a[-1,2]),
            minimum_height_mm=float(a[:,1].min()*1000),peak_torque_nm=float(a[:,8].max()),
            peak_joint_speed_rad_s=float(a[:,9].max()),maximum_floor_penetration_mm=max_pen*1000,
            upright_steady_fraction=float(upright.mean()),raw_recovery_criterion=bool(upright.mean()>.95),
            head_or_jaw_floor_contact=any('jaw' in n for n in contacts),floor_contacts=sorted(contacts),
            collision_suspects=sorted(suspects.values(),key=lambda r:-r['penetration_mm']),
            full_collision_validated=False,controller='scripted position targets; no learned recovery or head-world lock')
        dest=OUT/'trials';dest.mkdir(exist_ok=True)
        stem=f'{label}__s{seed}__lag{self.command_ms}'
        (dest/(stem+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
        np.savez_compressed(dest/(stem+'.npz'),trace=a,qpos=np.array(qs))
        print(json.dumps({k:result[k] for k in ('label','seed','final_height_mm','final_tilt_deg','raw_recovery_criterion')}),flush=True)
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('fold','recovery'),default='fold');args=p.parse_args()
    OUT.mkdir(exist_ok=True);probe=Probe()
    if args.stage=='fold':
        geo=[probe.solve_depth(x/1000) for x in (0,10,20,30,40,50,60,70)]
        (OUT/'fold_geometry.json').write_text(json.dumps(geo,indent=2),encoding='utf-8')
        results=[]
        for row in geo:
            q=np.array(row['target']);depth=int(row['depth_mm'])
            for kind in ('mesh','box'):
                e=Probe(kind)
                results.append(e.dynamics(f'fold_{depth}mm_{kind}',e.home,from_rpy(0,0),
                    [(0,e.home),(1,e.home),(3,q),(5,q),(7,e.home),(10,e.home)]))
        (OUT/'fold_trials.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    else:
        geo=json.loads((OUT/'fold_geometry.json').read_text());fold=np.array(geo[4]['target'])
        results=[]
        for label,quat in [('near_prone',from_rpy(0,np.deg2rad(35))),('prone',from_rpy(0,np.pi/2)),
                           ('supine',from_rpy(0,-np.pi/2)),('left_side',from_rpy(np.pi/2,0)),('right_side',from_rpy(-np.pi/2,0))]:
            for seed in (0,1,2):
                e=Probe()
                results.append(e.dynamics(label+'_home',e.home,quat,[(0,e.home),(10,e.home)],seed=seed))
                results.append(e.dynamics(label+'_fold_extend',e.home,quat,
                    [(0,e.home),(1,e.home),(2.5,fold),(4,fold),(6,e.home),(10,e.home)],seed=seed))
        (OUT/'recovery_trials.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    (OUT/'source_sha256.json').write_text(json.dumps({str(Path(__file__).resolve()):sha(__file__),str(XML):sha(XML)},indent=2),encoding='utf-8')


if __name__=='__main__':main()
