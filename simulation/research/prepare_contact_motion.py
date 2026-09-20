"""Freeze the bounded torsion candidate; never alter production geometry."""
from pathlib import Path
import json,shutil
import mujoco
import numpy as np
from evaluate_policy import sha

OUT=Path(__file__).parent/'20260914_contact_motion'
BASE=Path(__file__).parent/'20260914_squat_specialist/plant'
if __name__=='__main__':
    folder=OUT/'plant';folder.mkdir(exist_ok=False)
    m=mujoco.MjModel.from_binary_path('x',assets={'x':(BASE/'nominal.mjb').read_bytes()})
    profile=[1.,.015,.000001]
    feet=[m.geom('robot/'+side+'_foot_collision').id for side in ('left','right')]
    footprint=[]
    for gid in feet:
        m.geom_friction[gid]=profile
        mid=m.geom_dataid[gid];start=m.mesh_vertadr[mid]
        v=m.mesh_vert[start:start+m.mesh_vertnum[mid]]
        footprint.append(dict(name=m.geom(gid).name,mesh_extent_mm=(np.ptp(v,axis=0)*1000).tolist(),
            priority=int(m.geom_priority[gid]),condim=int(m.geom_condim[gid])))
    buffer=np.empty(mujoco.mj_sizeModel(m),np.uint8);mujoco.mj_saveModel(m,buffer=buffer)
    (folder/'nominal.mjb').write_bytes(buffer.tobytes())
    contract=json.loads((BASE/'contract.json').read_text())
    contract.update(xml=str(OUT.parent/'20260914_squat_specialist/robot/robot_microdinosaur_v07.xml'),
        mass_kg=float(m.body_mass.sum()),foot_friction=profile,friction_measured=False,
        model_sha256=sha(folder/'nominal.mjb'))
    (folder/'contract.json').write_text(json.dumps(contract,indent=2))
    (OUT/'selected_contact.json').write_text(json.dumps(dict(friction=profile,
        units=['dimensionless','m','m'],effective=[1.,1.,.015,.00001,.00001],
        status='PROVISIONAL_BOUNDED_TORSION_CANDIDATE',measured=False,footprint=footprint,
        rationale='Mean high-frequency head jitter improved in paired screening and fresh confirmation; individual seeds vary. Sliding and rolling increases were not supported.',
        base_plant_sha256=sha(BASE/'nominal.mjb'),plant_sha256=sha(folder/'nominal.mjb'),
        material='CAD assumes TPU95A; exact sole/floor material pair unmeasured'),indent=2))
    original=mujoco.MjModel.from_binary_path('x',assets={'x':(BASE/'nominal.mjb').read_bytes()})
    for field in ('body_mass','body_inertia','body_ipos','body_pos','body_quat','jnt_range',
        'dof_damping','dof_armature','dof_frictionloss','actuator_gainprm','actuator_biasprm','actuator_forcerange'):
        np.testing.assert_array_equal(getattr(m,field),getattr(original,field))
    difference=np.argwhere(m.geom_friction!=original.geom_friction)
    assert difference.tolist()==[[feet[0],1],[feet[1],1]]
    (folder/'invariants.json').write_text(json.dumps(dict(only_changed_array_entries=difference.tolist(),
        mass_kg=float(m.body_mass.sum()),production_unchanged=True),indent=2))
    print(json.dumps(footprint))
