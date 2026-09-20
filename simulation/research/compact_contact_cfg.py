"""Isolated flat-ground contact plant for compact folding / future specialists.

Use each CAD body's visual convex hull against the infinite flat plane only:
its lowest point equals the minimum of the source vertices, unlike an AABB.
These hulls must NOT be used as general self-collision or stairs geometry.
"""
from pathlib import Path
import json
import shutil
import mujoco
import numpy as np
from mjlab.scene import Scene
from terrain_skill_cfg import XML
from run_jump_cfg import build_config as original_config
from evaluate_policy import sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT / '20260914_compact_fold'


def robot_spec():
    spec = mujoco.MjSpec.from_file(str(XML))
    for geom in list(spec.geoms):
        if geom.type != mujoco.mjtGeom.mjGEOM_MESH or geom.contype or geom.conaffinity:
            continue
        body = geom.parent.name
        if body in ('ankle_left', 'ankle_right'):
            continue
        spec.body(body).add_geom(name=body+'_floor_proxy',
            type=mujoco.mjtGeom.mjGEOM_MESH, meshname=geom.meshname,
            pos=geom.pos.copy(), quat=geom.quat.copy(), density=0,
            contype=4, conaffinity=0, group=3, friction=[.6,.005,.0001],
            condim=3, rgba=[.2,.6,.5,.15])
    hull_path = OUT / 'jaw_hulls.json'
    if hull_path.exists():
        rows = json.loads(hull_path.read_text())['parts']
        data = np.load(OUT / 'jaw_hulls.npz')
        for row in rows:
            name = 'mouth_'+row['key']
            spec.add_mesh(name=name, uservert=data[row['key']].astype(float).ravel().tolist())
            spec.body(row['body']).add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name, density=0, contype=0, conaffinity=0, group=3,
                rgba=[.9,.4,.1,.15])
        # Explicit pairs are necessary: upper/lower jaw are adjacent bodies.
        for a in rows:
            if a['body'] != 'jaw_soft': continue
            for b in rows:
                if b['body'] != 'jaw_hinge': continue
                spec.add_pair(geomname1='mouth_'+a['key'], geomname2='mouth_'+b['key'],
                              condim=3, friction=[.6,.6,.005,.0001,.0001])
    return spec


def finalize_contacts(spec):
    # Scene hook executes AFTER Entity CollisionCfg disabled unmatched geoms.
    for geom in spec.geoms:
        if geom.name.endswith('_floor_proxy'):
            geom.contype=4; geom.conaffinity=0
        if geom.group == 0:
            geom.conaffinity |= 4


def build_config(skill='run', envs=64, seed=914, *, feasibility_only=False):
    if not feasibility_only:
        raise RuntimeError('Compact-fold CAD interference gate failed. This candidate is diagnostic only; review solid_intersections.json before training.')
    task, cfg = original_config(skill, envs, seed)
    cfg.env.scene.entities['robot'].spec_fn = robot_spec
    cfg.env.scene.spec_fn = finalize_contacts
    return task, cfg


def compile_plant(label='plant'):
    folder = OUT / label
    folder.mkdir(exist_ok=False)
    _, cfg = build_config(envs=1, feasibility_only=True)
    scene = Scene(cfg.env.scene, device='cpu')
    oldpath = ROOT / '20260914_run_jump/plant/nominal.mjb'
    old = mujoco.MjModel.from_binary_path('x.mjb', assets={'x.mjb':oldpath.read_bytes()})
    for key in ('timestep','integrator','solver','iterations','ls_iterations','cone','impratio','tolerance','ls_tolerance'):
        setattr(scene.spec.option,key,getattr(old.opt,key))
    m = scene.compile()
    invariant_fields = ['body_mass','body_inertia','body_ipos','jnt_range',
        'dof_damping','dof_armature','dof_frictionloss','actuator_gainprm',
        'actuator_biasprm','actuator_forcerange','actuator_gear']
    for field in invariant_fields:
        np.testing.assert_array_equal(getattr(m,field),getattr(old,field))
    proxies = [g for g in range(m.ngeom) if m.geom(g).name.endswith('_floor_proxy')]
    assert len(proxies)==18 and all(m.geom_contype[g]==4 for g in proxies)
    contract_path = ROOT / '20260914_run_jump/plant/contract.json'
    c = json.loads(contract_path.read_text())
    d=mujoco.MjData(m);d.qpos[:3]=[0,0,1];d.qpos[3:7]=[1,0,0,0]
    for n,v in zip(c['action_names'],c['action_offset'][0]):
        d.qpos[m.jnt_qposadr[m.joint('robot/'+n).id]]=v
    mujoco.mj_forward(m,d);home=d.qpos.copy();floor=m.geom('terrain').id
    controls=[]
    for gid in proxies:
        d.qpos[:]=home;mujoco.mj_forward(m,d)
        mid=m.geom_dataid[gid];start=m.mesh_vertadr[mid]
        vertices=m.mesh_vert[start:start+m.mesh_vertnum[mid]]
        minimum=(vertices@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid])[:,2].min()
        d.qpos[2]-=minimum+.003;mujoco.mj_forward(m,d)
        hits=[x for x in d.contact if {x.geom1,x.geom2}=={floor,gid}]
        assert hits, m.geom(gid).name
        controls.append(dict(name=m.geom(gid).name,forced_contacts=len(hits)))
    buffer=np.empty(mujoco.mj_sizeModel(m),np.uint8);mujoco.mj_saveModel(m,buffer=buffer)
    (folder/'nominal.mjb').write_bytes(buffer.tobytes())
    shutil.copy2(contract_path,folder/'contract.json')
    report=dict(original_sha256=sha(oldpath),model_sha256=sha(folder/'nominal.mjb'),
        invariant_fields=invariant_fields, mass_kg=float(m.body_mass.sum()),
        floor_contact_positive_controls=controls, explicit_mouth_pairs=int(m.npair),
        floor_geometry='CAD visual body hull, flat plane only; no AABB extension',
        full_self_collision=False, battery_moved=False, production_modified=False)
    (folder/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
    return folder


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--label',default='plant')
    compile_plant(p.parse_args().label)
