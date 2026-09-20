"""S288/3S/JY61P sensitivity model on the existing nominal v07 plant.

Known protocol resolutions are separate from unmeasured motor-envelope,
mechanical, voltage and timing hypotheses. Builtin implicit PD is retained.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import mujoco
import numpy as np
from evaluate_policy import Sim
from mjlab_microduck.s288_protocol import (
    OUTPUT_ENCODER_STEP, ROTOR_POSITION_OUTPUT_STEP, VELOCITY_STEP,
    KP_STEP, KD_STEP, JY61P_GYRO_STEP, JY61P_GYRO_LIMIT,
)


@dataclass
class HardwareCase:
    name: str = 's288_protocol'
    protocol: bool = True
    motor_curve: bool = False
    voltage: float = 12.6
    effort_fraction: float = 1.
    no_load_speed_at_12v: float = 16.5
    command_ms: int = 10
    position_ms: int = 20
    velocity_ms: int = 20
    physics_dt: float = .005
    joint_friction_scale: float = 1.
    kp_scale: float = 1.
    kd_scale: float = 1.
    armature_scale: float = 1.
    ground_friction: float = 1.
    # MuJoCo torque / normal-force coefficients are lengths in meters.
    # None preserves the compiled model; these are unmeasured hypotheses.
    foot_torsional_friction_m: float | None = None
    foot_rolling_friction_m: float | None = None
    foot_contact_timeconst_s: float | None = None
    body_contact_timeconst_s: float | None = None
    slope_deg: float = 0.
    head_mass_scale: float = 1.
    gyro_bias_deg_s: float = 0.
    imu_period_ms: int = 20
    feedback_drop_probability: float = 0.
    bus_stall_ms: int = 0
    push_force_n: float = 0.


def torque_bounds(velocity, voltage=12.6, effort_fraction=1., no_load_speed_at_12v=16.5):
    """Provisional 4-quadrant *bound*, not an identified S288 motor curve.

    Motoring capability tapers with speed; opposing braking remains available.
    Never hard-clamp qvel or create accelerating torque beyond no-load speed.
    Voltage scaling/linear curve need bench identification; .6 is a peak ceiling.
    """
    if not 6.4 <= voltage <= 12.6 or not 0 < effort_fraction <= 1 or no_load_speed_at_12v <= 0:
        raise ValueError('Invalid S288 envelope parameters')
    stall = .6 * voltage/12.6
    cap = .6 * effort_fraction
    speed = no_load_speed_at_12v * voltage/12.
    upper = np.clip(stall*(1-velocity/speed),0,cap)
    lower = -np.clip(stall*(1+velocity/speed),0,cap)
    return lower, upper


class HardwareSim(Sim):
    def __init__(self, plant, onnx, case=None):
        self.case = case or HardwareCase()
        self._initialized_hardware = False
        super().__init__(plant,onnx)
        c = self.case
        if c.command_ms <= 0 or c.position_ms <= 0 or c.velocity_ms <= 0:
            raise ValueError('Hardware cases require positive working-range delays')
        if not 0 <= c.feedback_drop_probability < 1:
            raise ValueError('Invalid packet loss probability')
        if not np.isfinite(c.ground_friction) or c.ground_friction<=0:
            raise ValueError('Sliding friction must be finite and positive')
        self.model.opt.timestep = c.physics_dt
        self.decimation = round(self.dt/c.physics_dt)
        self.command_lag = round(c.command_ms/1000/c.physics_dt)
        self.obs_lag = round(c.position_ms/1000/self.dt)
        self.velocity_lag = round(c.velocity_ms/1000/self.dt)
        assert np.isclose(self.command_lag*c.physics_dt,c.command_ms/1000)
        assert np.isclose(self.decimation*c.physics_dt,self.dt)
        m = self.model
        m.actuator_gainprm[self.aids,0] *= c.kp_scale
        m.actuator_biasprm[self.aids,1] *= c.kp_scale
        m.actuator_biasprm[self.aids,2] *= c.kd_scale
        if c.protocol:
            kp = np.round(m.actuator_gainprm[self.aids,0]/KP_STEP)*KP_STEP
            kd = np.round(-m.actuator_biasprm[self.aids,2]/KD_STEP)*KD_STEP
            m.actuator_gainprm[self.aids,0] = kp
            m.actuator_biasprm[self.aids,1] = -kp
            m.actuator_biasprm[self.aids,2] = -kd
        m.dof_frictionloss[self.vadr] *= c.joint_friction_scale
        m.dof_armature[self.vadr] *= c.armature_scale
        m.geom_friction[self.foot_geoms,0] = c.ground_friction
        for column, coefficient in ((1,c.foot_torsional_friction_m),(2,c.foot_rolling_friction_m)):
            if coefficient is not None:
                if not np.isfinite(coefficient) or coefficient <= 0:
                    raise ValueError('Foot friction coefficients must be finite and positive')
                m.geom_friction[self.foot_geoms,column] = coefficient
        terrain = m.geom('terrain').id
        for indices, timeconst in ((self.foot_geoms,c.foot_contact_timeconst_s),
                ([g for g in range(m.ngeom) if m.geom(g).name.endswith('_floor_proxy')],c.body_contact_timeconst_s)):
            if timeconst is not None:
                if timeconst < 2*c.physics_dt:raise ValueError('Contact time constant below two physics steps')
                m.geom_solref[indices,0]=timeconst
                # Changing priority to select solref must not inadvertently
                # lower the previously mixed body-floor friction coefficient.
                for gid in indices:
                    if m.geom_priority[gid]==m.geom_priority[terrain]:
                        m.geom_friction[gid]=np.maximum(m.geom_friction[gid],m.geom_friction[terrain])
                    elif m.geom_priority[gid]<m.geom_priority[terrain]:
                        m.geom_friction[gid]=m.geom_friction[terrain]
                m.geom_priority[indices]=1
        m.geom_friction[terrain,0] = c.ground_friction
        angle = np.deg2rad(c.slope_deg)
        m.geom_quat[terrain] = [np.cos(angle/2),0,np.sin(angle/2),0]
        # The compiled identity plane is marked SAMEFRAME_BODY. Clear that
        # optimization so runtime kinematics actually uses the new quaternion.
        m.geom_sameframe[terrain] = 0
        # Coherent mass/inertia scale of the rigid camera/head carrier body.
        m.body_mass[self.head] *= c.head_mass_scale
        m.body_inertia[self.head] *= c.head_mass_scale
        mujoco.mj_setConst(m,self.data)
        self.base_force_ranges = m.actuator_forcerange.copy()
        bus_map = json.loads((Path(__file__).resolve().parents[2]/'current/s288_joint_bus_map.json').read_text(encoding='utf-8'))
        by_joint = {r['joint'].removeprefix('CTRL_').removeprefix('DCTL_').lower():r['bus'] for r in bus_map}
        assert set(by_joint)==set(self.names)
        self.bus_ids = [[i for i,n in enumerate(self.names) if by_joint[n]==bus] for bus in ('A','B')]
        assert sorted(map(len,self.bus_ids))==[9,10]
        self.sensor_slices = {}
        for name in ('imu_accel','head_imu_accel','head_imu_ang_vel'):
            sid = m.sensor('robot/'+name).id
            self.sensor_slices[name] = slice(m.sensor_adr[sid],m.sensor_adr[sid]+m.sensor_dim[sid])
        self.contract['hardware_case'] = asdict(c)
        self._initialized_hardware = True
        self.substep_callback = self.capture_physics
        self.reset()

    def reset(self,seed=0,perturb=False):
        if self._initialized_hardware:
            self.model.actuator_forcerange[:] = self.base_force_ranges
        super().reset(seed,perturb)
        self.rng_hardware = np.random.default_rng(10000+seed)
        self.previous_measured = None
        self.previous_target = None
        self.held_gyro = None
        self.sample_age = np.zeros(2)
        self.feedback_timestamps = []
        self.last_feedback_age_ms = 0.
        self.trace = []
        self.actual_force_bounds = np.array([[-.6,.6]]*19)

    def transform_proprio(self, p):
        c = self.case
        if c.protocol:
            p[6:25] = np.round((p[6:25]+self.home)/OUTPUT_ENCODER_STEP)*OUTPUT_ENCODER_STEP-self.home
            p[25:44] = np.round(p[25:44]/VELOCITY_STEP)*VELOCITY_STEP
            p[:3] = np.round(np.clip(p[:3],-JY61P_GYRO_LIMIT,JY61P_GYRO_LIMIT)/JY61P_GYRO_STEP)*JY61P_GYRO_STEP
        p[2] += np.deg2rad(c.gyro_bias_deg_s)
        actor_tick = round(self.data.time/self.dt)
        hold_ticks = max(1,round(c.imu_period_ms/(self.dt*1000)))
        if self.held_gyro is None or actor_tick % hold_ticks == 0:
            self.held_gyro = p[:3].copy()
        p[:3] = self.held_gyro
        if self.previous_measured is not None:
            for b, ids in enumerate(self.bus_ids):
                stalled = b==0 and 3 <= self.data.time < 3+c.bus_stall_ms/1000
                drop = self.rng_hardware.random() < c.feedback_drop_probability
                if stalled or drop:
                    for offset in (6,25):
                        index = np.asarray(ids)+offset
                        p[index] = self.previous_measured[index]
                    self.sample_age[b] += self.dt*1000
                else:
                    self.sample_age[b] = 0
        self.feedback_timestamps.append(self.data.time-self.sample_age/1000)
        ages = [self.data.time-self.feedback_timestamps[max(0,len(self.feedback_timestamps)-1-lag)]
                for lag in (self.obs_lag,self.velocity_lag)]
        self.last_feedback_age_ms = float(np.max(ages)*1000)
        self.feedback_timestamps = self.feedback_timestamps[-(max(self.obs_lag,self.velocity_lag)+2):]
        self.previous_measured = p.copy()
        return p

    def before_physics_step(self, target):
        c = self.case
        if c.protocol:
            target = np.round(target/ROTOR_POSITION_OUTPUT_STEP)*ROTOR_POSITION_OUTPUT_STEP
        else:
            target = target.copy()
        if 3 <= self.data.time < 3+c.bus_stall_ms/1000 and self.previous_target is not None:
            target[self.bus_ids[0]] = self.previous_target[self.bus_ids[0]]
        self.previous_target = target.copy()
        if c.motor_curve:
            lo,hi = torque_bounds(self.data.qvel[self.vadr],c.voltage,c.effort_fraction,c.no_load_speed_at_12v)
            self.actual_force_bounds = np.stack([lo,hi],axis=1)
            self.model.actuator_forcerange[self.aids] = self.actual_force_bounds
        else:
            self.actual_force_bounds = self.base_force_ranges[self.aids]*c.effort_fraction
            self.model.actuator_forcerange[self.aids] = self.actual_force_bounds
        self.data.xfrc_applied[self.body,:] = 0
        if 3 <= self.data.time < 3.1:
            self.data.xfrc_applied[self.body,1] = c.push_force_n
        return target

    def capture_physics(self):
        d = self.data
        qvel = d.qvel[self.vadr].copy()
        torque = d.actuator_force[self.aids].copy()
        acc = np.r_[d.sensordata[self.sensor_slices['imu_accel']],d.sensordata[self.sensor_slices['head_imu_accel']]]
        hg = d.sensordata[self.sensor_slices['head_imu_ang_vel']].copy()
        self.trace.append((qvel,torque,acc,hg,self.actual_force_bounds.copy(),self.last_feedback_age_ms))

    def fall_test(self, samples):
        # A walking robot descends in world Z on a downhill plane. Judge body
        # clearance relative to that plane, not the world-origin height.
        terrain = self.model.geom('terrain').id
        normal = self.data.geom_xmat[terrain].reshape(3,3)[:,2]
        clearance = (samples[:,:3]-self.data.geom_xpos[terrain]) @ normal
        return bool(np.any(clearance < .06) or np.any(samples[:,6] > 60))

    def physics_metrics(self,seconds):
        traces = self.trace[-round(seconds/self.model.opt.timestep):]
        vel = np.array([r[0] for r in traces]); tau = np.array([r[1] for r in traces])
        acc = np.array([r[2] for r in traces]); gyro = np.array([r[3] for r in traces])
        bounds = np.array([r[4] for r in traces])
        power = vel*tau
        sat = (tau <= bounds[:,:,0]+1e-4) | (tau >= bounds[:,:,1]-1e-4)
        dt = self.model.opt.timestep
        return {'joint_speed_max_rad_s':float(np.abs(vel).max()),
            'joint_speed_p99_rad_s':float(np.percentile(np.abs(vel),99)),
            'joint_torque_rms_nm':np.sqrt(np.mean(tau**2,axis=0)).tolist(),
            'torque_squared_integral_nm2_s':(np.sum(tau**2,axis=0)*dt).tolist(),
            'positive_mechanical_energy_j':float(np.maximum(power,0).sum()*dt),
            'negative_mechanical_energy_j':float(-np.minimum(power,0).sum()*dt),
            'peak_positive_mechanical_power_w':float(np.maximum(power,0).sum(axis=1).max()),
            'dynamic_torque_saturation_fraction':float(sat.mean()),
            'body_specific_force_peak_g':float(np.linalg.norm(acc[:,:3],axis=1).max()/9.81),
            'head_specific_force_peak_g':float(np.linalg.norm(acc[:,3:],axis=1).max()/9.81),
            'head_gyro_rms_deg_s':float(np.rad2deg(np.sqrt(np.mean(np.sum(gyro**2,axis=1))))),
            'max_feedback_age_ms':max(r[5] for r in traces)}
