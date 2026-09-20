"""Frozen gait + body heading loop + optional head IMU actuator feedback."""
from dataclasses import asdict
import numpy as np
from imu_heading import matrix, from_rpy, wrap
from head_attitude import HeadKinematics, HeadController, HeadConfig, rotation_vector
from run_heading_stable_start import StableStartExperiment
from heading_sim import WARMUP_SECONDS


def cad_kinematics(sim):
    m = sim.model
    jids = [m.joint('robot/'+name).id for name in HeadKinematics.names]
    bids = m.jnt_bodyid[jids]
    assert list(m.body_parentid[bids]) == [sim.body, *bids[:-1]]
    assert np.all(m.body_jntnum[bids] == 1)
    return HeadKinematics([matrix(q) for q in m.body_quat[bids]], m.jnt_axis[jids],
                          m.qpos0[m.jnt_qposadr[jids]], matrix(m.site_quat[m.site('robot/head_imu').id]),
                          matrix(m.site_quat[m.site('robot/imu').id]), m.jnt_range[jids])


class HeadExperiment(StableStartExperiment):
    """Both arms use identical v7 stationary calibration and body heading loop."""
    def __init__(self, *args, head_enabled=True, head_config=None, head_motion='hold', **kwargs):
        super().__init__(*args, **kwargs)
        self.head_enabled = head_enabled
        self.head_config = head_config or HeadConfig()
        if head_motion not in ('hold', 'yaw_scan', 'pitch_scan', 'roll_scan'):
            raise ValueError(head_motion)
        self.head_motion = head_motion
        self.kinematics = cad_kinematics(self.sim)
        self.chain_indices = np.array([self.sim.names.index(n) for n in self.kinematics.names])
        self.head_indices = self.chain_indices[1:]
        self.head_controller = HeadController(self.kinematics, self.head_config)
        self.active = False
        self.sim.transform_target = self.transform_target
        previous = self.sim.substep_callback
        def capture():
            previous()
            if self.active:
                # Actual orientation is read only for scoring, never feedback.
                r = self.sim.data.site_xmat[self.sim.model.site('robot/head_imu').id].reshape(3, 3)
                gyro = self.sim.data.sensordata[self.sim.sensor_slices['head_imu_ang_vel']]
                omega = r@gyro
                self.head_physics.append(np.r_[self.sim.data.time-self.sim.model.opt.timestep,
                    np.arctan2(r[2, 1], r[2, 2]), omega,
                    self.sim.data.qpos[self.sim.jadr[self.head_indices]],
                    self.sim.data.actuator_force[np.asarray(self.sim.aids)[self.head_indices]]])
        self.sim.substep_callback = capture

    def reset(self, seed=0):
        self.active = False
        calibration = super().reset(seed)
        # Latest quantized encoder sample and IMU source sample are both at
        # 5.98 s in the nominal transport. Startup is verified stationary.
        measured = self.sim.obs_history[-1][6:25]+self.sim.home
        relative, _ = self.kinematics.forward(measured[self.chain_indices])
        expected = matrix(self.body_est.q)@relative
        independent = matrix(self.head_est.q)
        self.head_alignment = wrap(np.arctan2(expected[1, 0], expected[0, 0])-
                                   np.arctan2(independent[1, 0], independent[0, 0]))
        self.alignment_rotation = matrix(from_rpy(0, 0, self.head_alignment))
        self.head_controller.reset(self.sim.applied[self.head_indices])
        self.head_request = np.zeros(3)
        self.previous_desired = np.eye(3)
        self.head_rows = []
        self.head_physics = []
        self.active = True
        calibration['head_body_alignment_rad'] = self.head_alignment
        calibration['alignment_source'] = 'stationary encoder FK and accelerometer tilt, no world yaw'
        return calibration

    def transform_target(self, target, obs):
        if not self.active:
            return target
        measured = obs[6:25].astype(float)+self.sim.home
        orientation = self.alignment_rotation@matrix(self.head_est.q)
        gyro = self.stream.latest[1][2]-self.head_est.bias
        age = max(0., self.sim.data.time-self.head_est.last_time)
        # Camera desired forward follows user heading, not body correction.
        time = self.sim.data.time-WARMUP_SECONDS
        requested = np.zeros(3)
        if 2 <= time < 6 and self.head_motion != 'hold':
            axis, angle = {'yaw_scan': (2, .25), 'pitch_scan': (1, .12), 'roll_scan': (0, .10)}[self.head_motion]
            requested[axis] = angle
        # Explicit look commands are offsets from navigation heading; they are
        # never inferred from the controller's own corrective movement.
        self.head_request += np.clip(requested-self.head_request, -.5*self.sim.dt, .5*self.sim.dt)
        reference_rpy = self.head_request+np.array([0., 0., self.controller.reference])
        # 与训练侧同源:起身任务里头部偏航跟随躯干,而不是锁世界坐标(见 head_imu_action 的
        # yaw_follows_trunk 说明)。两侧必须一致,否则会重演今晚"训练有界、评估没有"的漂移。
        if getattr(self, 'yaw_follows_trunk', False):
            _r = self.sim.data.xmat[self.sim.body].reshape(3, 3)
            reference_rpy[2] += np.arctan2(_r[1, 0], _r[0, 0])
        desired = matrix(from_rpy(*reference_rpy))
        desired_omega = self.previous_desired@rotation_vector(self.previous_desired.T@desired)/self.sim.dt
        self.previous_desired = desired.copy()
        head_target = self.head_controller.update(target[self.head_indices], measured[self.chain_indices],
            orientation, gyro, desired, desired_omega, self.sim.dt, age)
        result = target.copy()
        if self.head_enabled:
            result[self.head_indices] = head_target
        self.head_rows.append(np.r_[self.sim.data.time-WARMUP_SECONDS, self.head_controller.error,
            self.head_controller.correction, self.head_controller.stale, self.head_controller.limited,
            age, result[self.head_indices], target[self.head_indices], reference_rpy])
        return result

    def run(self, *args, **kwargs):
        metrics, trace = super().run(*args, **kwargs)
        a = np.asarray(self.head_physics)
        controls = np.asarray(self.head_rows)
        full = np.asarray(self.sim.gaze_trace)[-len(a):]
        rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
        desired_pitch = np.interp(full[:, 0]-WARMUP_SECONDS, controls[:, 0], controls[:, 17])
        desired_roll = np.interp(full[:, 0]-WARMUP_SECONDS, controls[:, 0], controls[:, 16])
        metrics.update(head_enabled=self.head_enabled, head_config=asdict(self.head_config),
            head_motion=self.head_motion,
            camera_pitch_rms_deg=float(np.rad2deg(rms(full[:, 6]))),
            camera_roll_rms_deg=float(np.rad2deg(rms(a[:, 1]))),
            camera_pitch_tracking_rms_deg=float(np.rad2deg(rms(full[:, 6]+desired_pitch))),
            camera_roll_tracking_rms_deg=float(np.rad2deg(rms(a[:, 1]-desired_roll))),
            head_pitch_rate_rms_deg_s=float(np.rad2deg(rms(a[:, 3]))),
            head_roll_rate_rms_deg_s=float(np.rad2deg(rms(a[:, 2]))),
            head_feedback_stale_fraction=float(np.mean(controls[:, 7])) if self.head_enabled else 0.,
            head_controller_limited_fraction=float(np.mean(controls[:, 8])) if self.head_enabled else 0.,
            head_correction_max_rad=np.max(abs(controls[:, 4:7]), axis=0).tolist() if self.head_enabled else [0.]*3,
            head_target_slew_max_rad_s=float(np.max(abs(np.diff(controls[:, 10:13], axis=0)))/self.sim.dt),
            head_joint_position_min_rad=np.min(a[:, 5:8], axis=0).tolist(),
            head_joint_position_max_rad=np.max(a[:, 5:8], axis=0).tolist(),
            head_joint_torque_rms_nm=np.sqrt(np.mean(a[:, 8:11]**2, axis=0)).tolist(),
            head_joint_torque_peak_nm=float(np.max(abs(a[:, 8:11]))),
            head_joint_torque_saturation_fraction=float(np.mean(abs(a[:, 8:11]) >= .5999)),
            physics=self.sim.physics_metrics(metrics['seconds']))
        return metrics, trace


HEAD_TRACE_COLUMNS = ['time', 'error_roll', 'error_pitch', 'error_yaw', 'correction_pitch',
    'correction_yaw', 'correction_roll', 'stale', 'limited', 'age', 'target_pitch', 'target_yaw',
    'target_roll', 'policy_pitch', 'policy_yaw', 'policy_roll', 'reference_roll', 'reference_pitch', 'reference_yaw']
