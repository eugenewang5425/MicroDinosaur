"""Freeze the nominal training plant and its policy interface for CPU evaluation."""
import argparse
import json
from pathlib import Path

import mjlab  # load task plugins before importing robot constants
import mujoco
import numpy as np
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True)
parser.add_argument('--xml')
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
cfg = load_env_cfg('Mjlab-Velocity-Flat-MicroDinosaur', play=False)
cfg.scene.num_envs = 1
if args.xml:
    cfg.scene.entities['robot'].spec_fn = lambda: mujoco.MjSpec.from_file(args.xml)
event_names = list(cfg.events)
cfg.events = {}  # compile nominal parameters, not one random sample
env = ManagerBasedRlEnv(cfg, device='cuda:0')
model = env.sim.mj_model
robot = env.scene['robot']
action = env.action_manager.get_term('joint_pos')
data = {
    'task': 'Mjlab-Velocity-Flat-MicroDinosaur',
    'xml': args.xml or 'zcode_v06_default',
    'mass_kg': float(model.body_mass.sum()),
    'physics_dt': env.physics_dt,
    'policy_dt': env.step_dt,
    'action_names': action.target_names,
    'action_scale': action.scale.tolist() if hasattr(action.scale, 'tolist') else action.scale,
    'action_offset': action.offset.tolist() if hasattr(action.offset, 'tolist') else action.offset,
    'action_filter': {'alpha_new': cfg.actions['joint_pos'].lp_alpha,
                      'max_delta_rad': cfg.actions['joint_pos'].max_delta},
    'observations': {g: {n: {'delay_lags': [t.delay_min_lag, t.delay_max_lag],
                            'delay_step_seconds': env.step_dt,
                            'function': t.func.__module__ + '.' + t.func.__name__}
                         for n, t in group.terms.items()}
                     for g, group in cfg.observations.items()},
    'disabled_events_for_nominal_capture': event_names,
    'joints': [], 'bodies': [], 'geoms': [],
}
for i in range(model.njnt):
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
        continue
    dof = model.jnt_dofadr[i]
    data['joints'].append({'name': model.joint(i).name,
                          'armature': float(model.dof_armature[dof]),
                          'frictionloss': float(model.dof_frictionloss[dof]),
                          'damping': float(model.dof_damping[dof]),
                          'range': model.jnt_range[i].tolist()})
for i in range(1, model.nbody):
    data['bodies'].append({'name': model.body(i).name, 'mass': float(model.body_mass[i])})
for i in range(model.ngeom):
    if model.geom_contype[i] or model.geom_conaffinity[i]:
        data['geoms'].append({'name': model.geom(i).name, 'condim': int(model.geom_condim[i]),
                              'priority': int(model.geom_priority[i]),
                              'friction': model.geom_friction[i].tolist()})
buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
mujoco.mj_saveModel(model, buffer=buffer)
(out / 'nominal.mjb').write_bytes(buffer.tobytes())
(out / 'contract.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
print(json.dumps({'mass_kg': data['mass_kg'], 'physics_dt': env.physics_dt,
                  'policy_dt': env.step_dt, 'actions': data['action_names'],
                  'first_joint': data['joints'][0], 'output': str(out)}, indent=2))
env.close()
