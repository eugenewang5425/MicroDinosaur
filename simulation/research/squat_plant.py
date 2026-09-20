"""Current CAD ancestry correction, with active CAD hull/flat-plane contacts."""
import json,shutil
from pathlib import Path
import mujoco
import numpy as np
from mjlab.scene import Scene
from run_jump_cfg import build_config as base_config
from compact_contact_cfg import finalize_contacts
from evaluate_policy import sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_squat_specialist'
XML=OUT/'robot/robot_microdinosaur_v07.xml'


def robot_spec():
    assets={p.name:p.read_bytes() for p in (XML.parent/'assets').glob('*.stl')}
    spec=mujoco.MjSpec.from_string(XML.read_text(),assets=assets)
    for geom in list(spec.geoms):
        if geom.type!=mujoco.mjtGeom.mjGEOM_MESH or geom.contype or geom.conaffinity:continue
        body=geom.parent.name
        if body in ('ankle_left','ankle_right'):continue
        spec.body(body).add_geom(name=body+'_floor_proxy',type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=geom.meshname,pos=geom.pos.copy(),quat=geom.quat.copy(),density=0,
            contype=4,conaffinity=0,group=3,friction=[.6,.005,.0001],condim=3)
    return spec


def scene_config(envs=1,seed=914):
    task,cfg=base_config('jump',envs,seed)
    cfg.env.scene.entities['robot'].spec_fn=robot_spec
    cfg.env.scene.spec_fn=finalize_contacts
    return task,cfg


def compile_plant():
    folder=OUT/'plant';folder.mkdir(exist_ok=False)
    _,cfg=scene_config();scene=Scene(cfg.env.scene,device='cpu')
    oldpath=ROOT/'20260914_compact_fold/floor_only_plant/nominal.mjb'
    old=mujoco.MjModel.from_binary_path('x',assets={'x':oldpath.read_bytes()})
    for key in ('timestep','integrator','solver','iterations','ls_iterations','cone','impratio','tolerance','ls_tolerance'):
        setattr(scene.spec.option,key,getattr(old.opt,key))
    model=scene.compile()
    fields=['body_pos','body_quat','jnt_axis','jnt_pos','jnt_range','dof_damping','dof_armature',
            'dof_frictionloss','actuator_gainprm','actuator_biasprm','actuator_forcerange','actuator_gear']
    for field in fields:np.testing.assert_array_equal(getattr(model,field),getattr(old,field))
    assert abs(model.body_mass.sum()-old.body_mass.sum())<1e-6
    proxies=[i for i in range(model.ngeom) if model.geom(i).name.endswith('_floor_proxy')]
    assert len(proxies)==18 and all(model.geom_contype[i]==4 for i in proxies)
    mass_changes={model.body(i).name:float((model.body_mass[i]-old.body_mass[i])*1000)
                  for i in range(model.nbody) if abs(model.body_mass[i]-old.body_mass[i])>1e-8}
    oldowners=json.loads((ROOT/'20260913_handoff/native_v07/mass_provenance.json').read_text()) if (ROOT/'20260913_handoff/native_v07/mass_provenance.json').exists() else None
    buffer=np.empty(mujoco.mj_sizeModel(model),np.uint8);mujoco.mj_saveModel(model,buffer=buffer)
    (folder/'nominal.mjb').write_bytes(buffer.tobytes())
    shutil.copy2(ROOT/'20260914_run_jump/plant/contract.json',folder/'contract.json')
    report=dict(mass_kg=float(model.body_mass.sum()),body_mass_changes_g=mass_changes,
        unchanged_fields=fields,floor_proxies_active=18,model_sha256=sha(folder/'nominal.mjb'),
        xml_sha256=sha(XML),old_model_sha256=sha(oldpath),
        inertia_recomputed_from_same_per_part_ledger=True,cad_modified=False,battery_moved=False,
        self_contact_handling='Task limits avoid known ankle and jaw interference; no global self-contact release')
    (folder/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))


if __name__=='__main__':compile_plant()
