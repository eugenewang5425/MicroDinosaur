"""Measure the newly observed head-pitch casing / neck bracket interference."""
from pathlib import Path
import sys,json
import numpy as np
import mujoco
OUT=Path(__file__).parent/'20260914_contact_motion'
sys.path.insert(0,str(OUT.parent/'20260914_compact_fold/tools'))
import manifold3d as md
parts=json.loads((OUT/'preparation_cad/collision_parts_indexed.json').read_text())
data=np.load(OUT/'preparation_cad/collision_parts_indexed.npz')
names=['S288_head_pitch_CASE','neck_pitch__037'];solids=[];bodies=[]
m=mujoco.MjModel.from_binary_path('x',assets={'x':(OUT/'normal_contact_plant/nominal.mjb').read_bytes()});d=mujoco.MjData(m)
c=json.loads((OUT/'normal_contact_plant/contract.json').read_text())
for name,value in zip(c['action_names'],c['action_offset'][0]):d.qpos[m.jnt_qposadr[m.joint('robot/'+name).id]]=value
d.qpos[:7]=[0,0,.117182,1,0,0,0]
for name in names:
    r=next(r for r in parts if r['name']==name);k=r['key']
    solids.append(md.Manifold(md.Mesh((data[k+'_v']*1000).astype(np.float32),data[k+'_f'])))
    bodies.append(m.body('robot/'+r['body']).id)
rows=[]
for deg in (-60,-45,-30,-20,0,20,25,27,30,32,34,40,60):
    d.qpos[m.jnt_qposadr[m.joint('robot/head_pitch').id]]=np.deg2rad(deg);mujoco.mj_forward(m,d)
    positioned=[]
    for solid,b in zip(solids,bodies):
        t=np.c_[d.xmat[b].reshape(3,3),d.xpos[b]*1000];positioned.append(solid.transform(t))
    rows.append(dict(head_pitch_deg=deg,intersection_mm3=float((positioned[0]^positioned[1]).volume())))
(OUT/'head_pitch_bracket_scan.json').write_text(json.dumps(dict(parts=names,rows=rows),indent=2))
print(json.dumps(rows))
