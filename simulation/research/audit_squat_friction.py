"""Inspect actual contact coefficients; changing only the floor is insufficient."""
import json
from pathlib import Path
import mujoco
import numpy as np
from squat_plant import OUT
from evaluate_policy import sha

m=mujoco.MjModel.from_binary_path('x',assets={'x':(OUT/'plant/nominal.mjb').read_bytes()})
d=mujoco.MjData(m)
c=json.loads((OUT/'plant/contract.json').read_text())
d.qpos[:3]=[0,0,.114];d.qpos[3:7]=[1,0,0,0]
for name,angle in zip(c['action_names'],c['action_offset'][0]):d.qpos[m.jnt_qposadr[m.joint('robot/'+name).id]]=angle
floor=m.geom('terrain').id
feet=[m.geom('robot/'+side+'_foot_collision').id for side in ('left','right')]
geometry=[dict(name=m.geom(g).name,friction=m.geom_friction[g].tolist(),
    priority=int(m.geom_priority[g]),condim=int(m.geom_condim[g])) for g in [floor,*feet]]
def contact_values():
    mujoco.mj_forward(m,d)
    values=sorted({tuple(contact.friction.tolist()) for contact in d.contact
        if floor in (contact.geom1,contact.geom2) and any(g in (contact.geom1,contact.geom2) for g in feet)})
    assert values,'Test must generate actual foot contacts'
    return values
baseline=contact_values()
m.geom_friction[floor,0]=.2
floor_only=contact_values()
assert floor_only==baseline
cases=[]
for mu in (.6,1.,1.3):
    m.geom_friction[feet,0]=mu
    effective=contact_values()
    assert all(np.isclose(v[0],mu) and np.isclose(v[1],mu) for v in effective)
    cases.append(dict(requested_sliding_coefficient=mu,actual_contact=effective))
ledger=json.loads((OUT.parents[1]/'design_source/current/mass_estimate.json').read_text())
rows=ledger['rows']
if isinstance(rows,dict):rows=list(rows.values())
soles=[r for r in rows if 'sole_' in str(r.get('name',r.get('object','')))]
report=dict(plant_sha256=sha(OUT/'plant/nominal.mjb'),geometry=geometry,
    baseline_actual_contact=baseline,floor_only_mu02_actual_contact=floor_only,
    contact_coefficient_tests=cases,all_tests_passed=True,
    total_mass_kg=float(m.body_mass.sum()),nominal_weight_N=float(m.body_mass.sum()*9.81),
    measured_material_pair=None,sole_design_records=soles,
    important='Sliding coefficient is dimensionless; torsional and rolling parameters are separate. Contact solver clamps tiny coefficients, so actual contact values are recorded.',
    references=['https://mujoco.readthedocs.io/en/latest/modeling.html#contact-parameters',
                'https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-geom-friction'])
(OUT/'friction_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
