"""Locate nominal CAD interference along selected one-joint paths."""
import sys,json
from pathlib import Path
import numpy as np
import mujoco
OUT=Path(__file__).resolve().parent/'20260914_compact_fold'
sys.path.insert(0,str(OUT/'tools'))
import manifold3d as md
rows=json.loads((OUT/'collision_parts_indexed.json').read_text());arrays=np.load(OUT/'collision_parts_indexed.npz')
solids={};bodies={}
for r in rows:
    k=r['key'];solids[r['name']]=md.Manifold(md.Mesh((arrays[k+'_v']*1000).astype(np.float32),arrays[k+'_f']))
    assert solids[r['name']].status()==md.Error.NoError
    bodies[r['name']]=r['body']
m=mujoco.MjModel.from_binary_path('x',assets={'x':(OUT/'floor_only_plant/nominal.mjb').read_bytes()});d=mujoco.MjData(m)
contract=json.loads((OUT/'floor_only_plant/contract.json').read_text());d.qpos[:3]=[0,0,.117182];d.qpos[3:7]=[1,0,0,0]
for n,q in zip(contract['action_names'],contract['action_offset'][0]):d.qpos[m.jnt_qposadr[m.joint('robot/'+n).id]]=q
home=d.qpos.copy()
cases=[('left_ankle','S288_left_ankle_CASE','ankle_left__032',np.arange(25,71,1)),
       ('right_ankle','S288_right_ankle_CASE','ankle_right__077',np.arange(-25,-71,-1)),
       ('jaw_hinge','V13_Microduck_Head_Pan','Rex_Jaw_Shell',np.arange(-2,2.01,.1)),
       ('left_hip_pitch','hip_l__018','S288_left_hip_pitch_OUT_1_M2x5_0',np.arange(-26,31,2))]
results=[]
for joint,first,second,angles in cases:
    samples=[]
    for angle in angles:
        d.qpos[:]=home;d.qpos[m.jnt_qposadr[m.joint('robot/'+joint).id]]=np.deg2rad(angle);mujoco.mj_forward(m,d)
        posed=[]
        for name in (first,second):
            b=m.body('robot/'+bodies[name]).id
            matrix=np.column_stack([d.xmat[b].reshape(3,3),d.xpos[b]*1000])
            posed.append(solids[name].transform(matrix))
        cross=posed[0]^posed[1];assert cross.status()==md.Error.NoError
        samples.append(dict(angle_deg=float(angle),intersection_mm3=float(cross.volume())))
    row=dict(joint=joint,parts=[first,second],samples=samples,
             first_above_0p02_mm3=next((s for s in samples if s['intersection_mm3']>.02),None))
    results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='samples'}),flush=True)
(OUT/'joint_interference_scan.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
