"""Head IMU attitude feedback; pure NumPy, no simulator/world-state access.

Axes: camera-aligned X forward, Y left, Z up. The supplied CAD kinematics
maps delayed encoders to head IMU joint axes. Only pitch/yaw/roll are driven;
neck height, legs, jaw, arms and tail remain under the frozen gait policy.
"""
from dataclasses import dataclass
import numpy as np
from imu_heading import matrix, from_rpy, wrap


def rotation_vector(r):
    theta = float(np.arccos(np.clip((np.trace(r)-1)/2, -1, 1)))
    skew = np.array([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]])
    if theta < 1e-7:
        return .5*skew
    if np.pi-theta < 1e-5:
        values, vectors = np.linalg.eigh((r+r.T)/2)
        axis = vectors[:, np.argmax(values)]
        if axis@skew < 0:
            axis = -axis
        return theta*axis
    return theta/(2*np.sin(theta))*skew


def rotation_exp(vector):
    angle = float(np.linalg.norm(vector))
    q = np.r_[np.cos(angle/2), vector*(np.sin(angle/2)/angle if angle > 1e-12 else .5)]
    return matrix(q)


class HeadKinematics:
    """Static local rotations/axes only, no live MuJoCo data or FK engine."""
    names = ('neck_pitch', 'head_pitch', 'head_yaw', 'head_roll')

    def __init__(self, fixed_rotations, axes, joint_reference, head_imu_rotation,
                 body_imu_rotation, limits):
        self.fixed = np.asarray(fixed_rotations, float)
        self.axes = np.asarray(axes, float)
        self.reference = np.asarray(joint_reference, float)
        self.tip = np.asarray(head_imu_rotation, float)
        self.base = np.asarray(body_imu_rotation, float)
        self.limits = np.asarray(limits, float)

    def forward(self, angles):
        r = self.base.T.copy()
        axes = []
        for fixed, axis, q, reference in zip(self.fixed, self.axes, angles, self.reference):
            r = r@fixed
            axes.append(r@axis)
            r = r@rotation_exp(axis*(q-reference))
        r = r@self.tip
        # Angular Jacobian expressed in the head IMU frame.
        return r, r.T@np.asarray(axes[1:]).T


@dataclass(frozen=True)
class HeadConfig:
    # Rotation-vector axes (roll, pitch, yaw). Yaw gets the largest weight.
    kp: tuple = (.5, 1., 2.)
    kd: tuple = (.01, .03, .04)
    ki: tuple = (0., 0., 0.)
    integral_limit_rad: float = .1
    correction_limit_rad: tuple = (.35, .5, .18)  # Joint order: pitch,yaw,roll.
    correction_slew_rad_s: float = 2.
    target_slew_rad_s: float = 4.
    correction_filter_tau_s: float = .02
    joint_margin_rad: float = .035
    jacobian_damping: float = .05
    max_measurement_age_s: float = .12
    prediction_limit_s: float = .02
    nominal_target_filter_tau_s: float = .10


class HeadController:
    def __init__(self, kinematics, config=None):
        self.kinematics = kinematics
        self.config = config or HeadConfig()
        self.reset(np.zeros(3))

    def reset(self, initial_target):
        self.correction = np.zeros(3)
        self.integral = np.zeros(3)
        self.last_target = np.asarray(initial_target, float).copy()
        self.nominal_filtered = self.last_target.copy()
        self.error = np.zeros(3)
        self.stale = False
        self.limited = False

    def update(self, nominal_target, joint_angles, orientation, gyro, desired_orientation,
               desired_omega_world, dt, measurement_age):
        c = self.config
        values = np.r_[nominal_target, joint_angles, orientation.ravel(), gyro,
                       desired_orientation.ravel(), desired_omega_world, dt, measurement_age]
        if not np.isfinite(values).all() or dt <= 0 or measurement_age < 0:
            raise ValueError('Invalid head control input')
        _, jacobian = self.kinematics.forward(joint_angles)
        self.stale = measurement_age > c.max_measurement_age_s
        # Causal extrapolation from the last packet, not undelayed truth.
        predicted = orientation@rotation_exp(gyro*min(measurement_age, c.prediction_limit_s))
        self.error = rotation_vector(predicted.T@desired_orientation)
        gyro_error = gyro-predicted.T@desired_omega_world
        integral = np.clip(self.integral+np.asarray(c.ki)*self.error*dt,
                           -c.integral_limit_rad, c.integral_limit_rad)
        correction = np.asarray(c.kp)*self.error-np.asarray(c.kd)*gyro_error+integral
        # Damped inverse remains bounded near a singular neck configuration.
        desired = jacobian.T@np.linalg.solve(jacobian@jacobian.T+c.jacobian_damping**2*np.eye(3), correction)
        if self.stale:
            desired = np.zeros(3)
            integral = np.zeros(3)
        bounded = np.clip(desired, -np.asarray(c.correction_limit_rad), c.correction_limit_rad)
        filtered = self.correction+(1-np.exp(-dt/c.correction_filter_tau_s))*(bounded-self.correction)
        previous = self.correction.copy()
        self.correction += np.clip(filtered-self.correction,
                                   -c.correction_slew_rad_s*dt, c.correction_slew_rad_s*dt)
        alpha = 1-np.exp(-dt/c.nominal_target_filter_tau_s) if c.nominal_target_filter_tau_s > 0 else 1.
        self.nominal_filtered += alpha*(np.asarray(nominal_target)-self.nominal_filtered)
        raw = self.nominal_filtered+self.correction
        lo = self.kinematics.limits[1:, 0]+c.joint_margin_rad
        hi = self.kinematics.limits[1:, 1]-c.joint_margin_rad
        clipped = np.clip(raw, lo, hi)
        target = self.last_target+np.clip(clipped-self.last_target,
                                         -c.target_slew_rad_s*dt, c.target_slew_rad_s*dt)
        self.limited = bool(np.any(abs(desired-bounded)>1e-9) or np.any(abs(target-raw)>1e-9)
                            or np.any(abs(filtered-self.correction)>1e-9))
        # Conservative anti-windup: freeze all integral axes at any output limit.
        if not self.limited or self.stale:
            self.integral = integral
        self.last_target = target.copy()
        assert np.max(abs(self.correction-previous)) <= c.correction_slew_rad_s*dt+1e-9
        return target
