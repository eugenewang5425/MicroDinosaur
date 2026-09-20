"""User-specified vertical doubled-back legs. Static diagnostics only."""
from pathlib import Path
import json,hashlib
import numpy as np
import mujoco
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).parent;OUT=ROOT/'20260915_vertical_fold'
PLANT=ROOT/'20260914_contact_motion/preferred_plant'
OUT.mkdir(exist_ok=True)
m=mujoco.MjModel.from_binary_path('x',assets={'x':(PLANT/'nominal.mjb').read_bytes()});d=mujoco.MjData(m)
c=json.loads((PLANT/'contract.json').read_text());names=c['action_names']
jids=np.array([m.joint('robot/'+n).id for n in names]);adr=m.jnt_qposadr[jids]
home=np.array(c['action_offset'][0]);d.qpos[:3]=[0,0,.117182];d.qpos[3:7]=[1,0,0,0];d.qpos[adr]=home
d.qpos[adr[names.index('jaw_hinge')]]=.04;mujoco.mj_forward(m,d);base=d.qpos.copy()
def setq(n,v):d.qpos[m.jnt_qposadr[m.joint('robot/'+n).id]]=v
def joints(side):return [m.joint('robot/'+side+'_'+n).id for n in ('hip_pitch','knee','ankle')]
def projected(side):
    h,k,a=joints(side);axis=d.xaxis[k];u=d.xanchor[h]-d.xanchor[k];v=d.xanchor[a]-d.xanchor[k]
    return u-axis*(u@axis),v-axis*(v@axis)
def included(side):
    u,v=projected(side)
    return float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(u,v)),u@v)))
required={};sweeps=[]
for side,sign in [('left',1),('right',-1)]:
    k=joints(side)[1];lo,hi=m.jnt_range[k]
    samples=[]
    for q in np.linspace(lo,hi,1201):
        d.qpos[:]=base;d.qpos[m.jnt_qposadr[k]]=q;mujoco.mj_forward(m,d)
        samples.append(dict(q_deg=float(np.rad2deg(q)),included_deg=included(side)))
    # A rigid upstream rotation cannot change this projected link angle.
    def loss(q):
        d.qpos[:]=base;d.qpos[m.jnt_qposadr[k]]=q;mujoco.mj_forward(m,d)
        u,v=projected(side);return float(1-u@v/(np.linalg.norm(u)*np.linalg.norm(v)))
    fit=minimize_scalar(loss,bounds=np.deg2rad([100,145] if sign>0 else [-145,-100]),method='bounded',
        options={'xatol':1e-13})
    required[side]=float(fit.x)
    sweeps.append(dict(side=side,joint_range_deg=np.rad2deg([lo,hi]).tolist(),
        minimum_allowed=min(samples,key=lambda s:s['included_deg']),
        requested_closed_q_deg=float(np.rad2deg(fit.x)),samples=samples))
(OUT/'knee_range_audit.json').write_text(json.dumps(sweeps,indent=2))
foot_geoms=[m.geom('robot/'+side+'_foot_collision').id for side in ('left','right')]
def vertices(g):
    mid=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
    return v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
rows=[]
for label,closed,level_trunk in [('range_limited_vertical_shin',False,False),
    ('requested_vertical_fold_DIAGNOSTIC',True,False),
    ('requested_vertical_fold_level_trunk_DIAGNOSTIC',True,True)]:
    d.qpos[:]=base
    for side,sign in [('left',1),('right',-1)]:
        setq(side+'_hip_yaw',0);setq(side+'_hip_roll',0)
        setq(side+'_hip_pitch',required[side] if level_trunk else sign*np.deg2rad(85))
        setq(side+'_knee',required[side] if closed else sign*np.deg2rad(90-0.0001))
        setq(side+'_ankle',0)
    mujoco.mj_forward(m,d)
    _,v=projected('left');up=-v
    pitch=float(np.arctan2(-up[0],up[2]))
    d.qpos[3:7]=[np.cos(pitch/2),0,np.sin(pitch/2),0]
    setq('neck_pitch',home[names.index('neck_pitch')]-pitch)
    # Set the sagittal foot long axis level while the shank is vertical.
    foot=m.site('robot/left_foot').id
    def foot_level(q):
        setq('left_ankle',q);mujoco.mj_forward(m,d)
        x=d.site_xmat[foot].reshape(3,3)[:,0]
        return float(x[2]**2+max(x[0],0)**2)
    ankle=minimize_scalar(foot_level,bounds=(-np.pi/2,np.pi/2),method='bounded').x
    setq('left_ankle',ankle);setq('right_ankle',-ankle);mujoco.mj_forward(m,d)
    foot_v=np.concatenate([vertices(g) for g in foot_geoms])
    d.qpos[2]-=float(foot_v[:,2].min());d.qpos[0]-=float(foot_v[:,0].mean()+.065)
    mujoco.mj_forward(m,d);legs=[]
    for side in ('left','right'):
        h,k,a=joints(side);u,v=projected(side)
        tilt=lambda x:float(np.degrees(np.arctan2(np.linalg.norm(x[:2]),-x[2])))
        legs.append(dict(side=side,hip_mm=(d.xanchor[h]*1000).tolist(),knee_mm=(d.xanchor[k]*1000).tolist(),
            ankle_mm=(d.xanchor[a]*1000).tolist(),shin_from_vertical_deg=tilt(v),
            thigh_from_vertical_deg=tilt(u),folded_included_deg=included(side),
            hip_above_ankle_mm=float((d.xanchor[h,2]-d.xanchor[a,2])*1000)))
    q=d.qpos[adr];excess=np.maximum(m.jnt_range[jids,0]-q,q-m.jnt_range[jids,1]).clip(0)
    body_penetrations=[]
    for g in range(m.ngeom):
        n=m.geom(g).name
        if n.endswith('_floor_proxy'):
            z=float(vertices(g)[:,2].min())
            if z<-.0001:body_penetrations.append(dict(geom=n,lowest_z_mm=z*1000))
    row=dict(name=label,qpos=d.qpos.tolist(),joint_deg=dict(zip(names,np.rad2deg(q).tolist())),
        legs=legs,body_pitch_deg=float(np.rad2deg(pitch)),root_height_mm=float(d.qpos[2]*1000),
        joint_limit_excess_deg=float(np.rad2deg(excess.max())),
        violated_joints=[names[i] for i in np.flatnonzero(excess>1e-6)],
        body_floor_proxy_penetrations=body_penetrations,
        task_shape=bool(all(l['shin_from_vertical_deg']<1 and l['thigh_from_vertical_deg']<1
            and abs(l['hip_above_ankle_mm'])<1 for l in legs)),
        dynamic_simulation=False,learned=False,physical_model_modified=False)
    rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='qpos'}),flush=True)
(OUT/'poses.json').write_text(json.dumps(rows,indent=2))
frames=[json.loads((ROOT/'20260914_squat_specialist/reference_geometry/cad_pose_frames.json').read_text())[0]]
for row in rows:
    d.qpos[:]=row['qpos'];mujoco.mj_forward(m,d);bodies={}
    for b in range(m.nbody):
        n=m.body(b).name
        if not n.startswith('robot/'):continue
        t=np.eye(4);t[:3,:3]=d.xmat[b].reshape(3,3);t[:3,3]=d.xpos[b]
        bodies[n.removeprefix('robot/')]=t.tolist()
    frames.append(dict(name=row['name'],bodies=bodies))
(OUT/'cad_pose_frames.json').write_text(json.dumps(frames))
(OUT/'contract.json').write_text(json.dumps(dict(
    user_target='Shins vertical; thighs folded down alongside shins; hips at foot/ankle height.',
    goal_complete=False,kinematic_diagnostic_only=True,
    plant=str(PLANT.resolve()),plant_sha256=hashlib.sha256((PLANT/'nominal.mjb').read_bytes()).hexdigest(),
    current_limits_unchanged=True,diagnostic_qpos_outside_limits_not_a_motion_policy=True,
    knee_angle_definition='Link vectors projected onto the knee hinge plane, removing axial mounting offsets.',
    interpretation='Hip height measured at hip pitch axis; foot reference is ankle axis, not ground z=0.',
    exact_s288_output_motion_range_measured=False),indent=2))
