"""Positive contact control for missing floor proxies; no production model edit."""
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
from mjlab.scene import Scene
from run_jump_cfg import build_config, robot_spec
from evaluate_policy import sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT / '20260914_jump_dynamics_review'


def proxy_flags(model):
    return [dict(name=model.geom(i).name,contype=int(model.geom_contype[i]),conaffinity=int(model.geom_conaffinity[i]))
            for i in range(model.ngeom) if model.geom(i).name.endswith('_floor_proxy')]


def forced_overlap(model, qpos):
    data=mujoco.MjData(model);data.qpos[:]=qpos
    mujoco.mj_forward(model,data)
    floor=model.geom('terrain').id;gid=model.geom('robot/jaw_hinge_floor_proxy').id
    # A deliberately intersecting pose tests contact generation only.
    # It is not a valid robot pose, jump, or demonstration.
    return [dict(distance_m=float(c.dist),geom1=int(c.geom1),geom2=int(c.geom2))
            for c in data.contact if {c.geom1,c.geom2}=={floor,gid}]


def main():
    sys.stdout.reconfigure(encoding='utf-8');OUT.mkdir(exist_ok=True)
    raw=robot_spec();raw_proxies=[dict(name=g.name,contype=g.contype,conaffinity=g.conaffinity)
        for g in raw.geoms if g.name.endswith('_floor_proxy')]
    _,cfg=build_config('jump',1)
    scene=Scene(cfg.env.scene,device='cpu')
    frozen_path=ROOT/'20260914_run_jump/plant/nominal.mjb'
    frozen=mujoco.MjModel.from_binary_path('x.mjb',assets={'x.mjb':frozen_path.read_bytes()})
    for key in ['timestep','integrator','solver','iterations','ls_iterations','cone','impratio','tolerance','ls_tolerance']:
        setattr(scene.spec.option,key,getattr(frozen.opt,key))
    baseline=scene.compile()
    # This candidate applies the intended masks AFTER entity collision editors.
    for geom in scene.spec.geoms:
        if geom.name.endswith('_floor_proxy'):
            geom.contype=4;geom.conaffinity=0
    fixed=scene.spec.compile()
    for field in ['body_mass','body_inertia','jnt_range','dof_damping','dof_armature',
                  'dof_frictionloss','actuator_gainprm','actuator_biasprm','actuator_forcerange',
                  'geom_size','geom_pos','geom_quat','geom_friction','geom_solref','geom_solimp']:
        assert np.array_equal(getattr(baseline,field),getattr(fixed,field)),field
        assert np.array_equal(getattr(baseline,field),getattr(frozen,field)),field
    contract=json.loads((ROOT/'20260914_run_jump/plant/contract.json').read_text())
    data=mujoco.MjData(fixed);data.qpos[:3]=[0,0,1];data.qpos[3:7]=[1,0,0,0]
    for name,q in zip(contract['action_names'],contract['action_offset'][0]):
        data.qpos[fixed.jnt_qposadr[fixed.joint('robot/'+name).id]]=q
    mujoco.mj_forward(fixed,data);gid=fixed.geom('robot/jaw_hinge_floor_proxy').id
    bottom=data.geom_xpos[gid,2]-abs(data.geom_xmat[gid].reshape(3,3)[2])@fixed.geom_size[gid]
    data.qpos[2]-=bottom+.003
    original=forced_overlap(baseline,data.qpos);candidate=forced_overlap(fixed,data.qpos)
    assert len(raw_proxies)==18 and all(r['contype']==4 for r in raw_proxies)
    assert all(r['contype']==0 and r['conaffinity']==0 for r in proxy_flags(baseline))
    assert all(r['contype']==4 and r['conaffinity']==0 for r in proxy_flags(fixed))
    assert not original and candidate
    buffer=np.empty(mujoco.mj_sizeModel(fixed),np.uint8);mujoco.mj_saveModel(fixed,buffer=buffer)
    target=OUT/'ground_contact_candidate.mjb';target.write_bytes(buffer.tobytes())
    report=dict(raw_spec_proxies=raw_proxies,compiled_original=proxy_flags(baseline),compiled_candidate=proxy_flags(fixed),
        cause='Inherited CollisionCfg only matches .*_collision; disable_other_geoms clears all named *_floor_proxy geoms.',
        positive_control=dict(kind='deliberate 3mm floor overlap; contact-generation diagnostic only',
            baseline_contacts=original,candidate_contacts=candidate),
        mass_inertia_actuators_limits_materials_unchanged=True,candidate_sha256=sha(target),
        original_plant_sha256=sha(frozen_path),physics_dt=float(fixed.opt.timestep),
        candidate_is_not_historical_jump_plant=True,
        upper_lower_jaw_self_contact_fixed=False,production_or_training_source_modified=False)
    (OUT/'contact_compile_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(proxies=18,baseline_active=0,candidate_active=18,
        baseline_contacts=len(original),candidate_contacts=len(candidate),jaw_self_contact_fixed=False),indent=2))


if __name__=='__main__':main()
