"""Solid CAD Boolean intersection volumes in mm^3, not AABB penetration."""
import sys,json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent/'20260914_compact_fold'
sys.path.insert(0,str(OUT/'tools'))
import manifold3d as md
rows=json.loads((OUT/'collision_parts_indexed.json').read_text())
data=np.load(OUT/'collision_parts_indexed.npz');solids={};bodies={}
for row in rows:
    k=row['key'];solid=md.Manifold(md.Mesh((data[k+'_v']*1000).astype(np.float32),data[k+'_f']))
    assert solid.status()==md.Error.NoError,(row['name'],solid.status())
    solids[row['name']]=solid;bodies[row['name']]=row['body']
pairs=[('S288_left_ankle_CASE','ankle_left__032'),('S288_right_ankle_CASE','ankle_right__077'),
       ('V13_Microduck_Head_Pan','Rex_Jaw_Shell'),('Rex_Skull_Shell','Rex_Jaw_Shell'),
       ('hip_l__018','S288_left_hip_pitch_OUT_1_M2x5_0')]
result=[]
for frame in json.loads((OUT/'cad_pose_frames.json').read_text()):
    positioned={}
    for name,solid in solids.items():
        transform=np.array(frame['bodies'][bodies[name]])[:3];transform[:,3]*=1000
        positioned[name]=solid.transform(transform)
    row=dict(frame=frame['name'],pairs=[])
    for a,b in pairs:
        intersection=positioned[a]^positioned[b]
        assert intersection.status()==md.Error.NoError
        row['pairs'].append(dict(parts=[a,b],intersection_mm3=float(intersection.volume())))
    result.append(row);print(json.dumps(row),flush=True)
(OUT/'solid_intersections.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
