"""Numerical checks independent of training reward."""
import json
from pathlib import Path
import numpy as np
import mujoco
from evaluate_policy import Sim

root = Path(__file__).parent / '20260913_handoff'
frames = json.loads((root / 'cad_frames.json').read_text())
sim = Sim(root / 'native_v07', root / 'v7_reference.onnx')
old = Sim(root / 'native_v06', root / 'v7_reference.onnx')
cad_root = np.array(frames['FRAME_trunk_base'])
sim_root = sim.data.xpos[sim.body]
shift = sim_root - cad_root[:3, 3]
sensor = sim.model.site('robot/head_imu').id
expected = np.array(frames['FRAME_JY61P_HEAD_SENSOR'])
position_error = float(np.linalg.norm(sim.data.site_xpos[sensor] - (expected[:3, 3] + shift)))
rotation_error = float(np.max(np.abs(sim.data.site_xmat[sensor].reshape(3, 3) - expected[:3, :3])))
assert position_error < 1e-5, position_error
assert rotation_error < 1e-4, rotation_error
assert sim.names == old.names
assert np.allclose(sim.model.jnt_range[sim.jids], old.model.jnt_range[old.jids], atol=1e-6)
axis_error = float(np.max(np.abs(sim.data.xaxis[sim.jids] - old.data.xaxis[old.jids])))
anchor_error = float(np.max(np.abs(sim.data.xanchor[sim.jids] - old.data.xanchor[old.jids])))
assert axis_error < 1e-6 and anchor_error < 1e-6
body_errors = {}
for key, value in frames.items():
    if not key.startswith('BODY_'):
        continue
    name = 'robot/' + key[5:]
    body = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body < 0:
        continue
    matrix = np.array(value)
    err = float(np.linalg.norm(sim.data.xpos[body] - (matrix[:3, 3] + shift)))
    body_errors[name] = err
assert max(body_errors.values()) < 1e-5
ledger = json.loads((root.parents[1] / 'design_source/current/mass_estimate.json').read_text())
mass_error = float(abs(sim.model.body_mass.sum() - ledger['mass_estimate_g'] / 1000))
assert mass_error < 1e-6  # XML decimal rounding
expected_com = np.array(ledger['com_mm']) / 1000 + shift
actual_com = sim.data.subtree_com[sim.body]
com_error = float(np.linalg.norm(expected_com - actual_com))
# Hardware groups in the ledger use nominal COMs; the converter integrates
# every mesh's shape. Require sub-mm consistency, not bit identity.
assert com_error < 1e-3, (expected_com, actual_com, com_error)
result = {'status': 'PASS', 'actuated_joints': 19, 'joint_order_and_limits_unchanged': True,
          'head_imu_position_error_m': position_error,
          'head_imu_rotation_max_abs_error': rotation_error,
          'joint_axis_max_error': axis_error, 'joint_anchor_max_error_m': anchor_error,
          'body_home_position_max_error_m': max(body_errors.values()),
          'mass_error_kg': mass_error, 'com_error_m': com_error,
          'model_mass_kg': float(sim.model.body_mass.sum()),
          'legacy_mass_kg': float(old.model.body_mass.sum()),
          'cad_nominal_only': True}
(root / 'integration_checks.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
