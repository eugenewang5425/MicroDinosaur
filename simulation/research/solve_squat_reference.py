"""Bounded compact poses with body fore-aft freedom and reduced ankle flexion."""
from pathlib import Path
import json
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from head_attitude import rotation_vector
OUT=Path(__file__).resolve().parent/'20260914_squat_specialist'
m=mujoco.MjModel.from_binary_path('x',assets={'x':(OUT/'plant/nominal.mjb').read_bytes()});d=mujoco.MjData(m)
c=json.loads((OUT/'plant/contract.json').read_text());names=c['action_names'];home=np.array(c['action_offset'][0])
jids=[m.joint('robot/'+n).id for n in names];adr=m.jnt_qposadr[jids]
legs=np.array([i for i,n in enumerate(names) if n.startswith(('left_','right_'))])
d.qpos[:3]=[0,0,.117182];d.qpos[3:7]=[1,0,0,0];d.qpos[adr]=home
home[names.index('jaw_hinge')]=.04;d.qpos[adr]=home
mujoco.mj_forward(m,d);root=d.qpos.copy()
feet=[m.site('robot/'+n).id for n in ['left_foot','right_foot']]
footpos=d.site_xpos[feet].copy();footrot=d.site_xmat[feet].reshape(2,3,3).copy()
trunk=m.body('robot/trunk_base').id;com0=d.subtree_com[trunk].copy()
lower=m.jnt_range[jids,0][legs]+np.deg2rad(5);upper=m.jnt_range[jids,1][legs]-np.deg2rad(5)
for name in ['left_ankle','right_ankle']:
    k=list(legs).index(names.index(name));lower[k]=np.deg2rad(-51);upper[k]=np.deg2rad(51)
initial=np.r_[home[legs],0,0];rows=[]
for depth in np.arange(0,45.1,5):
    def residual(x):
        d.qpos[:]=root;d.qpos[0]+=x[-2];d.qpos[2]-=depth/1000
        d.qpos[3:7]=[np.cos(x[-1]/2),0,np.sin(x[-1]/2),0]
        d.qpos[adr[legs]]=x[:-2]
        # Nominal neck/head correction seeds the same world-facing IMU task.
        d.qpos[adr[names.index('head_pitch')]]=home[names.index('head_pitch')]-x[-1]
        mujoco.mj_forward(m,d);r=d.site_xmat[feet].reshape(2,3,3)
        com=d.subtree_com[trunk]
        return np.r_[(d.site_xpos[feet]-footpos).ravel()/.001,
            rotation_vector(footrot[0].T@r[0])/.025,rotation_vector(footrot[1].T@r[1])/.025,
            x[-2]/.06,x[-1]/.12,(com[:2]-com0[:2])/.04]
    sol=least_squares(residual,initial,bounds=(np.r_[lower,-.035,-.12],np.r_[upper,.035,.12]),max_nfev=400,
                      ftol=1e-10,xtol=1e-10,gtol=1e-10)
    residual(sol.x);initial=sol.x
    rows.append(dict(depth_mm=float(depth),target=d.qpos[adr].tolist(),root_qpos=d.qpos[:7].tolist(),
        root_dx_mm=float(sol.x[-2]*1000),root_pitch_deg=float(np.rad2deg(sol.x[-1])),
        foot_error_mm=np.linalg.norm(d.site_xpos[feet]-footpos,axis=1).tolist(),
        com_shift_mm=((d.subtree_com[trunk]-com0)*1000).tolist(),
        ankle_deg=np.rad2deg(d.qpos[adr[[names.index('left_ankle'),names.index('right_ankle')]]]).tolist(),
        knee_deg=np.rad2deg(d.qpos[adr[[names.index('left_knee'),names.index('right_knee')]]]).tolist()))
    rows[-1]['foot_error_mm']=[v*1000 for v in rows[-1]['foot_error_mm']]
    print(json.dumps(rows[-1]),flush=True)
(OUT/'pose_references.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
