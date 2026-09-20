"""Read-only CAD surface-intersection audit at measured compact-fold poses.

New inter-body surface intersections relative to the CAD HOME are reported,
including adjacent bodies. This is sampled geometry, not continuous validation.
"""
import sys,json
from pathlib import Path
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/20260914_squat_specialist'
if '--output-directory' in sys.argv:
    OUT=Path(sys.argv[sys.argv.index('--output-directory')+1]).resolve()
sys.path.insert(0,str(ROOT/'microdinosaur'))
import blend2mjcf as conv
sys.path.insert(0,str(ROOT/'research'))
from cad_contact_geometry import triangles_world
conv.mesh_tris_world=triangles_world
conv.WITH_TAIL=conv.WITH_ARMS=conv.WITH_JAW=True
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'design_source/current/MicroDinosaur_v1.blender'))
blend=conv.BlendData({o.name:o for o in bpy.data.objects if o.type=='EMPTY'},[o for o in bpy.data.objects if o.type=='MESH'])
zero,home,theta,rel,mz=conv.unwind(blend)
parts=conv.body_parts(blend.empties,blend.meshes)
geometry=[]
for body,objects in parts.items():
    for obj in objects:
        if conv.classify_skip(list(obj.data.materials),obj.name):continue
        tri=conv.part_tris_body_zero(obj,*zero[body],mz[body])
        geometry.append((body,obj.name,tri.reshape(-1,3)))
poses=json.loads((OUT/'cad_pose_frames.json').read_text())
baseline={};result=[]
for frame in poses:
    items=[];trees={}
    for body,name,vertices in geometry:
        transform=np.array(frame['bodies'][body]);v=vertices@transform[:3,:3].T+transform[:3,3]
        items.append((body,name,v,v.min(0),v.max(0)))
    intersections={};fresh=[]
    for i,(body,name,vertices,lo,hi) in enumerate(items):
        for j in range(i+1,len(items)):
            other,othername,v2,lo2,hi2=items[j]
            if body==other:continue
            extent=np.minimum(hi,hi2)-np.maximum(lo,lo2)
            if extent.min()<.00005:continue
            for k in (i,j):
                if k not in trees:
                    v=items[k][2]
                    trees[k]=BVHTree.FromPolygons(v.tolist(),np.arange(len(v)).reshape(-1,3).tolist(),all_triangles=True,epsilon=1e-8)
            count=len(trees[i].overlap(trees[j]))
            if count==0:continue
            key=name+' | '+othername
            intersections[key]=count
            if frame['name']!='home' and baseline.get(key,0)==0:
                fresh.append(dict(parts=[name,othername],bodies=[body,other],
                    triangle_pairs=count,bbox_overlap_mm=(extent*1000).tolist()))
    if frame['name']=='home':baseline=intersections
    row=dict(name=frame['name'],intersecting_pairs=len(intersections),
        new_pairs=sorted(fresh,key=lambda r:-r['triangle_pairs']))
    result.append(row)
    (OUT/'cad_intersections.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(name=frame['name'],baseline_pairs=len(baseline),new_pairs=len(fresh))),flush=True)
