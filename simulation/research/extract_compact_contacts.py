"""Read current CAD only; extract jaw surfaces in the existing MJCF frames."""
import sys
from pathlib import Path
import json
import hashlib
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/20260914_compact_fold'
sys.path.insert(0, str(ROOT / 'microdinosaur'))
import blend2mjcf as conv
sys.path.insert(0, str(ROOT / 'research'))
from cad_contact_geometry import triangles_world
conv.mesh_tris_world = triangles_world

source = ROOT / 'design_source/current/MicroDinosaur_v1.blender'
bpy.ops.wm.open_mainfile(filepath=str(source))
conv.WITH_TAIL = conv.WITH_ARMS = conv.WITH_JAW = True
blend = conv.BlendData({o.name: o for o in bpy.data.objects if o.type == 'EMPTY'},
                       [o for o in bpy.data.objects if o.type == 'MESH'])
zero, home, theta, rel, mz = conv.unwind(blend)
parts = conv.body_parts(blend.empties, blend.meshes)
OUT.mkdir(exist_ok=True)
rows = []; arrays = {}
for body in ('jaw_soft', 'jaw_hinge'):
    for obj in parts[body]:
        if conv.classify_skip(list(obj.data.materials), obj.name):
            continue
        tri = conv.part_tris_body_zero(obj, *zero[body], mz[body])
        world = conv.mesh_tris_world(obj)
        key = f'p{len(rows)}'
        arrays[key] = tri
        rows.append(dict(key=key, body=body, name=obj.name, triangles=len(tri),
                         local_min=tri.reshape(-1, 3).min(0).tolist(),
                         local_max=tri.reshape(-1, 3).max(0).tolist(),
                         world_min=world.reshape(-1, 3).min(0).tolist(),
                         world_max=world.reshape(-1, 3).max(0).tolist()))
np.savez_compressed(OUT / 'jaw_parts.npz', **arrays)
(OUT / 'jaw_parts.json').write_text(json.dumps(dict(
    cad_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), parts=rows), indent=2), encoding='utf-8')
print(json.dumps([r for r in rows if any(k in r['name'].upper() for k in ('SHELL','MOUTH'))]))
trees = []
for name in ('Rex_Skull_Shell', 'Rex_Jaw_Shell'):
    tri = conv.mesh_tris_world(bpy.data.objects[name])
    trees.append(BVHTree.FromPolygons(tri.reshape(-1, 3).tolist(),
                 np.arange(len(tri)*3).reshape(-1, 3).tolist(), all_triangles=True))
print('HOME_SKULL_JAW_SURFACE_INTERSECTIONS', len(trees[0].overlap(trees[1])))
