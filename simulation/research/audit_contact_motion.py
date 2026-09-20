from pathlib import Path
import json
import mujoco
import numpy as np
from mjlab.scene import Scene
from contact_motion_cfg import build_config,OUT

selected=json.loads((OUT/'selected_contact.json').read_text())
_,cfg=build_config('run',1)
scene=Scene(cfg.env.scene,device='cpu');gpu_nominal=scene.compile()
cpu=mujoco.MjModel.from_binary_path('x',assets={'x':(OUT/selected['plant_directory']/'nominal.mjb').read_bytes()})
fields=('body_mass','body_inertia','body_pos','body_quat','body_ipos','jnt_range',
    'dof_damping','dof_frictionloss','dof_armature','actuator_gainprm','actuator_biasprm',
    'geom_friction','geom_solref','geom_solimp','geom_priority','geom_contype','geom_conaffinity')
for field in fields:np.testing.assert_allclose(getattr(gpu_nominal,field),getattr(cpu,field),rtol=0,atol=1e-12,err_msg=field)
report=dict(matching_gpu_cpu_fields=fields,mass_kg=float(cpu.body_mass.sum()),
    feet=[dict(name=cpu.geom(i).name,friction=cpu.geom_friction[i].tolist(),solref=cpu.geom_solref[i].tolist(),
        condim=int(cpu.geom_condim[i]),priority=int(cpu.geom_priority[i])) for i in range(cpu.ngeom) if cpu.geom(i).name.endswith('foot_collision')],
    initial_distribution='Fresh sensor snapshots on this plant; no within-episode reset assistance',
    identified_hardware_parameters=False)
(OUT/'gpu_cpu_plant_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
