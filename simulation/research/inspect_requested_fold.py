"""Kinematic photo-pose feasibility, not a policy or a physics success claim."""
from pathlib import Path
import json
import numpy as np
import mujoco
from scipy.optimize import least_squares
from head_attitude import rotation_vector

ROOT=Path(__file__).parent
OUT=ROOT/'20260914_requested_fold'
PLANT=ROOT/'20260914_contact_motion/preferred_plant'
OUT.mkdir(exist_ok=True)
m=mujoco.MjModel.from_binary_path('x',assets={'x':(PLANT/'nominal.mjb').read_bytes()})
d=mujoco.MjData(m)
c=json.loads((PLANT/'contract.json').read_text())
names=c['action_names'];home=np.array(c['action_offset'][0])
jids=[m.joint('robot/'+n).id for n in names];adr=m.jnt_qposadr[jids]
left=np.array([names.index('left_'+n) for n in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')])
right=np.array([names.index('right_'+n) for n in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')])
neck=names.index('neck_pitch');head=names.index('head_pitch')
d.qpos[:3]=[0,0,.117182];d.qpos[3:7]=[1,0,0,0];d.qpos[adr]=home;d.qpos[adr[names.index('jaw_hinge')]]=.04
mujoco.mj_forward(m,d);base=d.qpos.copy()
feet=[m.site('robot/'+n).id for n in ('left_foot','right_foot')]
pos=d.site_xpos[feet].copy();rot=d.site_xmat[feet].reshape(2,3,3).copy()
trunk=m.body('robot/trunk_base').id
com0=d.subtree_com[trunk].copy()
bounds0=np.r_[np.deg2rad([-20,-18,-85,-85,-55]),-.065,.025,np.deg2rad(-6)]
bounds1=np.r_[np.deg2rad([20,18,85,85,55]),.065,.12,np.deg2rad(30)]
def apply(x):
    d.qpos[:]=base;d.qpos[[0,2]]=x[5:7];p=x[7]
    d.qpos[3:7]=[np.cos(p/2),0,np.sin(p/2),0]
    d.qpos[adr[left]]=x[:5];d.qpos[adr[right]]=-x[:5]
    d.qpos[adr[neck]]=home[neck]-p
    mujoco.mj_forward(m,d)
def measure(x):
    apply(x)
    h,k,a=[d.xanchor[m.joint('robot/left_'+n).id] for n in ('hip_pitch','knee','ankle')]
    u=h-k;v=a-k
    included=np.degrees(np.arccos(np.clip(u@v/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1)))
    return dict(qpos=d.qpos.tolist(),target=d.qpos[adr].tolist(),
        root_z_mm=float(d.qpos[2]*1000),root_dx_mm=float(x[5]*1000),
        root_pitch_deg=float(np.rad2deg(x[7])),left_joint_deg=np.rad2deg(x[:5]).tolist(),
        knee_segment_included_deg=float(included),
        foot_error_mm=(np.linalg.norm(d.site_xpos[feet]-pos,axis=1)*1000).tolist(),
        foot_rotation_error_deg=[float(np.rad2deg(np.linalg.norm(rotation_vector(rot[i].T@d.site_xmat[f].reshape(3,3))))) for i,f in enumerate(feet)],
        com_shift_mm=((d.subtree_com[trunk]-com0)*1000).tolist())
rows=[];previous=np.r_[home[left],0,.117182,0]
for knee in (0,30,50,60,65,70,75,80,85):
    def residual(x):
        apply(x)
        return np.r_[(d.site_xpos[feet]-pos).ravel()/.0005,
            rotation_vector(rot[0].T@d.site_xmat[feet[0]].reshape(3,3))/.01,
            rotation_vector(rot[1].T@d.site_xmat[feet[1]].reshape(3,3))/.01,
            (x[3]-np.deg2rad(knee))/.01,
            (d.subtree_com[trunk,0]-com0[0])/.004,x[7]/4]
    guesses=[previous,np.r_[np.deg2rad([0,-5,10,knee,50]),-.01,.065,.2],
        np.r_[np.deg2rad([0,-5,-40,knee,40]),.02,.055,.35]]
    solutions=[least_squares(residual,np.clip(x,bounds0+1e-8,bounds1-1e-8),bounds=(bounds0,bounds1),
        max_nfev=250,ftol=1e-10,xtol=1e-10,gtol=1e-10) for x in guesses]
    sol=min(solutions,key=lambda s:np.linalg.norm(s.fun));previous=sol.x
    row=dict(requested_knee_deg=knee,**measure(sol.x))
    row['kinematic_gate']=bool(max(row['foot_error_mm'])<.5 and max(row['foot_rotation_error_deg'])<1
        and abs(row['left_joint_deg'][3]-knee)<1 and abs(row['com_shift_mm'][0])<5)
    row['geometry_validated']=False;rows.append(row)
    print(json.dumps({k:v for k,v in row.items() if k not in ('qpos','target')}),flush=True)
(OUT/'kinematic_candidates.json').write_text(json.dumps(rows,indent=2))
frames=[json.loads((ROOT/'20260914_squat_specialist/reference_geometry/cad_pose_frames.json').read_text())[0]]
for row in rows:
    d.qpos[:]=row['qpos'];mujoco.mj_forward(m,d);bodies={}
    for b in range(m.nbody):
        n=m.body(b).name
        if not n.startswith('robot/'):continue
        t=np.eye(4);t[:3,:3]=d.xmat[b].reshape(3,3);t[:3,3]=d.xpos[b]
        bodies[n.removeprefix('robot/')]=t.tolist()
    frames.append(dict(name=f"knee_{row['requested_knee_deg']}",bodies=bodies))
(OUT/'cad_pose_frames.json').write_text(json.dumps(frames))
