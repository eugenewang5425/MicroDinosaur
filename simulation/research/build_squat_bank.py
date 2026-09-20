"""Fresh sensor-calibrated resets for the corrected squat plant."""
import json
import numpy as np
from squat_plant import OUT
from hardware_sim import HardwareCase
from head_attitude_sim import HeadExperiment
from head_attitude import HeadConfig
from imu_owned_head import configure_owned
from run_heading_stable_start import CALIBRATOR
from imu_heading import matrix
from evaluate_policy import sha

records=[]
for seed in (1,2,3):
    case=HardwareCase(physics_dt=.00125,motor_curve=True,voltage=12.)
    e=configure_owned(HeadExperiment(OUT/'plant',CALIBRATOR,case,
        head_config=HeadConfig(max_measurement_age_s=.04)),(1,))
    s=e.sim
    def target(goal,obs):
        goal=goal.copy();goal[s.names.index('jaw_hinge')]=.04
        return e.transform_target(goal,obs)
    s.transform_target=target
    calibration=e.reset(seed)
    body=s.data.site_xmat[s.model.site('robot/imu').id].reshape(3,3)
    _,sensor=e.stream.latest
    records.append(dict(seed=seed,root_qpos=s.data.qpos[:7].tolist(),root_qvel=s.data.qvel[:6].tolist(),
        joint_pos=s.data.qpos[s.jadr].tolist(),joint_vel=s.data.qvel[s.vadr].tolist(),
        raw_action=s.raw.tolist(),applied=s.applied.tolist(),head_rotation=matrix(e.head_est.q).tolist(),
        head_accel=e.head_est.accel_filtered.tolist(),head_bias=e.head_est.bias.tolist(),
        alignment=e.alignment_rotation.tolist(),packet=np.r_[sensor[2],sensor[3],sensor[1]].tolist(),
        heading_world_zero=float(np.arctan2(body[1,0],body[0,0])),calibration=calibration))
    print(seed,records[-1]['root_qpos'][2],flush=True)
result=dict(source_policy=str(CALIBRATOR),source_sha256=sha(CALIBRATOR),calibration_seconds=6.,
    sample_count=50,motor_delay_ms=10,position_delay_ms=20,physics_dt=.00125,
    plant_sha256=sha(OUT/'plant/nominal.mjb'),action_names=s.names,records=records,
    contract='Nominal 12V standing sensor snapshots. DR resets are not individually calibrated.')
(OUT/'calibrated_reset_bank.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
