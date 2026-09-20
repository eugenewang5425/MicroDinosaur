"""Bounded normal-contact hypothesis; preserve inherited effective friction."""
from pathlib import Path
import json
import mujoco
import numpy as np
from evaluate_policy import sha
OUT=Path(__file__).parent/'20260914_contact_motion'
if __name__=='__main__':
    record=json.loads((OUT/'selected_contact.json').read_text())
    (OUT/'selected_torsion_only.json').write_text(json.dumps(record,indent=2))
    dest=OUT/'normal_contact_plant';dest.mkdir(exist_ok=False)
    model=mujoco.MjModel.from_binary_path('x',assets={'x':(OUT/'plant/nominal.mjb').read_bytes()})
    for g in range(model.ngeom):
        n=model.geom(g).name
        if n.endswith(('_foot_collision','_floor_proxy')):
            model.geom_solref[g,0]=.005
            if n.endswith('_floor_proxy'):
                model.geom_priority[g]=1;model.geom_friction[g,0]=1.
    buffer=np.empty(mujoco.mj_sizeModel(model),np.uint8);mujoco.mj_saveModel(model,buffer=buffer)
    (dest/'nominal.mjb').write_bytes(buffer.tobytes())
    contract=json.loads((OUT/'plant/contract.json').read_text());contract.update(normal_contact_timeconst_s=.005,
        model_sha256=sha(dest/'nominal.mjb'))
    (dest/'contract.json').write_text(json.dumps(contract,indent=2))
    record.update(plant_directory='normal_contact_plant',bank_suffix='_normal005',normal_contact_timeconst_s=.005,
        contact_damping_ratio=1.,body_contact_sliding=1.,measured=False,
        reason_for_normal_contact_probe='Default 20ms normal response produced 17-19mm forced-fall penetration. 5ms candidate checked with unchanged control rate and positive damping.',
        plant_sha256=sha(dest/'nominal.mjb'))
    (OUT/'selected_contact.json').write_text(json.dumps(record,indent=2))
