"""Preserve CAD mesh topology for exact solid intersection checks."""
import sys,json
from pathlib import Path
import bpy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/20260914_squat_specialist'
if '--output-directory' in sys.argv:
    OUT=Path(sys.argv[sys.argv.index('--output-directory')+1]).resolve()
sys.path.insert(0,str(ROOT/'microdinosaur'))
import blend2mjcf as conv
conv.WITH_TAIL=conv.WITH_ARMS=conv.WITH_JAW=True
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'design_source/current/MicroDinosaur_v1.blender'))
blend=conv.BlendData({o.name:o for o in bpy.data.objects if o.type=='EMPTY'},[o for o in bpy.data.objects if o.type=='MESH'])
zero,home,theta,rel,mz=conv.unwind(blend)
parts=conv.body_parts(blend.empties,blend.meshes)
selected={'S288_left_ankle_CASE','ankle_left__032','S288_right_ankle_CASE','ankle_right__077',
          'V13_Microduck_Head_Pan','Rex_Jaw_Shell','Rex_Skull_Shell',
          'hip_l__018','S288_left_hip_pitch_OUT_1_M2x5_0'}
for frame in json.loads((OUT/'cad_intersections.json').read_text()):
    for row in frame['new_pairs']:selected.update(row['parts'])
rows=[];arrays={}
for body,objects in parts.items():
    for obj in objects:
        if obj.name not in selected:continue
        ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
        try:
            mesh.calc_loop_triangles()
            verts=np.array([v.co[:] for v in mesh.vertices],dtype=float)
            transform=np.linalg.inv(conv.M4(*home[body]))@np.array(ev.matrix_world)
            verts=verts@transform[:3,:3].T+transform[:3,3]
            key=f'p{len(rows)}';arrays[key+'_v']=verts
            arrays[key+'_f']=np.array([list(t.vertices) for t in mesh.loop_triangles],dtype=np.uint32)
            rows.append(dict(key=key,name=obj.name,body=body))
        finally:ev.to_mesh_clear()
np.savez_compressed(OUT/'collision_parts_indexed.npz',**arrays)
(OUT/'collision_parts_indexed.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print(json.dumps(rows))
