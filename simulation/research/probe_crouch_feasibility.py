"""Current-CAD kinematic crouch feasibility; physical hold measured separately."""
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from hardware_sim import HardwareSim, HardwareCase
from head_attitude import rotation_vector
from evaluate_policy import sha

ROOT = Path(__file__).parent
OUT = ROOT/'20260914_terrain_skills'


def main():
    OUT.mkdir(exist_ok=True)
    sim = HardwareSim(ROOT/'20260913_handoff/native_v07', ROOT/'20260913_handoff/v7_reference.onnx', HardwareCase(physics_dt=.00125))
    leg = np.array([i for i, n in enumerate(sim.names) if n.startswith(('left_', 'right_'))])
    feet = [sim.model.site('robot/'+s).id for s in ('left_foot', 'right_foot')]
    initial = sim.data.qpos.copy(); pos = sim.data.site_xpos[feet].copy()
    rotations = sim.data.site_xmat[feet].reshape(2, 3, 3).copy()
    limits = sim.model.jnt_range[np.array(sim.jids)[leg]]
    rows = []
    for depth in (0., .01, .02, .03):
        sim.data.qpos[:] = initial; sim.data.qpos[2] -= depth
        def error(q):
            sim.data.qpos[sim.jadr[leg]] = q; mujoco.mj_forward(sim.model, sim.data)
            current = sim.data.site_xmat[feet].reshape(2, 3, 3)
            return np.r_[(sim.data.site_xpos[feet]-pos).ravel()/.05,
                rotation_vector(rotations[0].T@current[0]), rotation_vector(rotations[1].T@current[1])]
        solved = least_squares(error, initial[sim.jadr[leg]], bounds=(limits[:, 0]+.03, limits[:, 1]-.03),
            max_nfev=300, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        error(solved.x)
        target = sim.home.copy(); target[leg] = solved.x
        error_mm = float(np.max(np.linalg.norm(sim.data.site_xpos[feet]-pos, axis=1))*1000)
        # A hold check is not inverse dynamics certification or a learned skill.
        # Preserve gravity and measured-mass model; apply actual nominal servo PD.
        sim.data.qvel[:] = 0; sim.data.ctrl[sim.aids] = target
        samples = []; peaks = []
        for _ in range(2400):
            sim.data.ctrl[sim.aids] = target
            mujoco.mj_step(sim.model, sim.data)
            peaks.append(float(np.max(abs(sim.data.actuator_force[sim.aids]))))
            if len(peaks)%16 == 0:
                mujoco.mj_forward(sim.model, sim.data); samples.append(sim.sample())
        a = np.array(samples)
        rows.append(dict(depth_m=depth, ik_foot_position_error_mm=error_mm,
            target_joint_angles=target.tolist(), final_body_height_m=float(a[-1, 2]),
            max_body_tilt_deg=float(a[:, 6].max()), peak_torque_nm=max(peaks), fell=sim.fall_test(a),
            kinematically_feasible=error_mm < .5, holds_without_fall=not sim.fall_test(a)))
    report = dict(xml_sha256=sha('D:/microduck_rl/src/mjlab_microduck/robot/microdinosaur_v07/robot_microdinosaur_v07.xml'),
        results=rows, caution='Local geometry, existing partial collision model; static servo holds are not learned crouching')
    (OUT/'crouch_feasibility.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps([{k: v for k, v in r.items() if k != 'target_joint_angles'} for r in rows], indent=2))


if __name__ == '__main__': main()
