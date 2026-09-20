"""Read current CAD attachment hierarchy before defining squat constraints."""
import sys,json
from pathlib import Path
import bpy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/20260914_squat_specialist'
OUT.mkdir(exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'design_source/current/MicroDinosaur_v1.blender'))
names=['hip_l__018','S288_left_hip_pitch_CASE','S288_left_hip_pitch_DISC_1',
       'S288_left_hip_pitch_OUT_1_M2x5_0','CTRL_left_hip_pitch','AXIS_left_hip_pitch',
       'BODY_hip_l','BODY_upper_leg_left','S288_left_ankle_CASE','ankle_left__032',
       'AXIS_left_ankle','CTRL_left_ankle']
rows=[]
for name in names:
    o=bpy.data.objects.get(name)
    if not o:continue
    ancestors=[];p=o
    while p is not None:
        ancestors.append(p.name);p=p.parent
    rows.append(dict(name=name,ancestors=ancestors,matrix=np.array(o.matrix_world).tolist(),
                     props={k:str(o[k]) for k in o.keys()},rotation_mode=o.rotation_mode,
                     rotation_euler=list(o.rotation_euler)))
(OUT/'attachments.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print(json.dumps(rows))
