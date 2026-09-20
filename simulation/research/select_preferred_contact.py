"""Keep nominal friction after the final normal-response interaction check."""
from pathlib import Path
import json
import numpy as np
import mujoco
from evaluate_policy import sha
OUT=Path(__file__).parent/'20260914_contact_motion'
if __name__=='__main__':
    selected=json.loads((OUT/'selected_contact.json').read_text())
    rows=[]
    for name in ('preferred_squat_delays','preferred_squat_low_voltage'):
        rows+=json.loads((OUT/name/'summary.json').read_text())
    assert len(rows)==11 and all(r['passed'] for r in rows)
    (OUT/'training_contact.json').write_text(json.dumps(selected,indent=2))
    old=OUT/selected['plant_directory'];dest=OUT/'preferred_plant';dest.mkdir(exist_ok=False)
    m=mujoco.MjModel.from_binary_path('x',assets={'x':(old/'nominal.mjb').read_bytes()})
    for side in ('left','right'):m.geom_friction[m.geom('robot/'+side+'_foot_collision').id,1]=.01
    buffer=np.empty(mujoco.mj_sizeModel(m),np.uint8);mujoco.mj_saveModel(m,buffer=buffer)
    (dest/'nominal.mjb').write_bytes(buffer.tobytes())
    contract=json.loads((old/'contract.json').read_text());contract.update(foot_friction=[1.,.01,.000001],model_sha256=sha(dest/'nominal.mjb'))
    (dest/'contract.json').write_text(json.dumps(contract,indent=2))
    selected.update(friction=[1.,.01,.000001],effective=[1.,1.,.01,.00001,.00001],
        status='PREFERRED_CONTACT_HYPOTHESIS_NOT_MEASURED',plant_directory='preferred_plant',
        bank_suffix='_preferred',plant_sha256=sha(dest/'nominal.mjb'),
        rationale='After normal response was corrected, raising torsion no longer showed clear benefit. Preserve original friction; nominal-response delay/voltage matrix passed 11/11. Training_contact.json retains the fixed 15mm-torsion recipe used in completed PPO runs.',
        recovery_bank_note='CAD-vetted physical initial-state distribution collected on the nearby 15mm-torsion hypothesis; it is not recalibration of each randomized model.')
    (OUT/'selected_contact.json').write_text(json.dumps(selected,indent=2))
    print(selected['plant_sha256'])
