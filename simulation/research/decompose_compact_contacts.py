"""Local, reproducible CAD shell decomposition; never fills the entire mouth."""
from pathlib import Path
import sys, json, time
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent
OUT = ROOT / '20260914_compact_fold'
sys.path.insert(0, str(OUT / 'tools'))
import coacd
import manifold3d as md
coacd.set_log_level('info')
parts = json.loads((OUT / 'collision_parts_indexed.json').read_text())
arrays = np.load(OUT / 'collision_parts_indexed.npz')
result = []; data = {}
parameters = dict(threshold=.0005, real_metric=True, resolution=2000,
                  mcts_iterations=40, mcts_max_depth=3, mcts_nodes=20,
                  seed=914, merge=True, decimate=False, preprocess_mode='auto',
                  preprocess_resolution=100)
for name in ('Rex_Jaw_Shell','V13_Microduck_Head_Pan','Rex_Skull_Shell'):
    row = next(r for r in parts if r['name'] == name)
    key=row['key']
    solid=md.Manifold(md.Mesh((arrays[key+'_v']*1000).astype(np.float32),arrays[key+'_f']))
    assert solid.status()==md.Error.NoError
    indexed=solid.to_mesh()
    mesh = trimesh.Trimesh(vertices=indexed.vert_properties[:,:3]/1000.,
                          faces=indexed.tri_verts, process=False)
    start = time.time()
    print(name, 'watertight', mesh.is_watertight, 'triangles', len(mesh.faces), flush=True)
    assert mesh.is_watertight
    parameters['preprocess_mode']='off'
    hulls = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), **parameters)
    for vertices, faces in hulls:
        key = f'h{len(result)}'
        data[key] = vertices
        result.append(dict(key=key, body=row['body'], source=name,
                           original_watertight=bool(mesh.is_watertight),
                           vertices=len(vertices), faces=len(faces)))
    print(name, 'hulls', len(hulls), 'seconds', time.time()-start, flush=True)
np.savez_compressed(OUT / 'jaw_hulls.npz', **data)
(OUT / 'jaw_hulls.json').write_text(json.dumps(dict(parts=result, parameters=parameters,
    coacd_version='1.0.9', cad_sha256=json.loads((OUT/'jaw_parts.json').read_text())['cad_sha256'],
    accuracy_note='0.5 mm is an algorithm threshold, not a certified maximum geometric error.'), indent=2), encoding='utf-8')
