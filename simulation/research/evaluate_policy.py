"""Reproducible CPU rollouts of a nominal model compiled by the training stack.

No random dynamics are silently added. Delay probes use explicit fixed lags;
training's random delay distribution requires a separate batch evaluation.
"""
import argparse
import hashlib
import gzip
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Sim:
    def __init__(self, plant, onnx, command_lag=0, obs_lag=0, velocity_lag=None):
        plant = Path(plant)
        self.contract = json.loads((plant / 'contract.json').read_text())
        binary = plant / 'nominal.mjb'
        model_bytes = binary.read_bytes() if binary.exists() else gzip.decompress(
            (plant / 'nominal.mjb.gz').read_bytes())
        self.model = mujoco.MjModel.from_binary_path('nominal.mjb',
            assets={'nominal.mjb': model_bytes})
        self.data = mujoco.MjData(self.model)
        self.session = ort.InferenceSession(str(onnx), providers=['CPUExecutionProvider'])
        assert self.session.get_inputs()[0].shape == [1, 81]
        assert self.session.get_outputs()[0].shape == [1, 19]
        self.names = self.contract['action_names']
        self.home = np.array(self.contract['action_offset'], dtype=np.float32).reshape(-1, 19)[0]
        self.scale = np.array(self.contract['action_scale'], dtype=np.float32)
        self.jids = [self.model.joint('robot/' + n).id for n in self.names]
        self.jadr = self.model.jnt_qposadr[self.jids]
        self.vadr = self.model.jnt_dofadr[self.jids]
        self.aids = [int(np.where(self.model.actuator_trnid[:, 0] == j)[0][0]) for j in self.jids]
        self.body = self.model.body('robot/trunk_base').id
        self.head = self.model.body('robot/jaw_soft').id
        self.feet = [self.model.body('robot/' + n).id for n in ('ankle_left', 'ankle_right')]
        self.foot_geoms = [self.model.geom('robot/' + n + '_foot_collision').id for n in ('left', 'right')]
        sensor = self.model.sensor('robot/imu_ang_vel').id
        self.imu_adr = self.model.sensor_adr[sensor]
        self.command_lag, self.obs_lag = command_lag, obs_lag
        self.velocity_lag = obs_lag if velocity_lag is None else velocity_lag
        if min(self.command_lag, self.obs_lag, self.velocity_lag) < 0:
            raise ValueError('Delay lags must be nonnegative')
        self.dt = self.contract['policy_dt']
        self.decimation = round(self.dt / self.model.opt.timestep)
        self.reset()

    def reset(self, seed=0, perturb=False):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0, 0, 0.117182]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[self.jadr] = self.home
        if perturb:
            rng = np.random.default_rng(seed)
            yaw = np.deg2rad(rng.uniform(-1, 1))
            self.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            self.data.qpos[self.jadr] += np.deg2rad(rng.uniform(-1, 1, 19))
            self.data.qvel[self.vadr] = rng.uniform(-.2, .2, 19)
        self.raw = np.zeros(19, dtype=np.float32)
        self.applied = self.home.copy()
        self.data.ctrl[self.aids] = self.home
        self.command_history, self.obs_history = [], []
        mujoco.mj_forward(self.model, self.data)

    def obs(self, command):
        d = self.data
        rotation = d.xmat[self.body].reshape(3, 3)
        proprio = np.concatenate([d.sensordata[self.imu_adr:self.imu_adr + 3],
                                  rotation.T @ [0., 0., -1.],
                                  d.qpos[self.jadr] - self.home, d.qvel[self.vadr]])
        transform = getattr(self, 'transform_proprio', None)
        if transform is not None:
            proprio = transform(proprio)
        self.obs_history.append(proprio.copy())
        # Explicit joint-feedback probe; IMU still fresh. Raw last_action and
        # commands never inherit sensor delay or amplitude scaling.
        if self.obs_lag:
            delayed = self.obs_history[max(0, len(self.obs_history) - 1 - self.obs_lag)]
            proprio[6:25] = delayed[6:25]
        if self.velocity_lag:
            delayed = self.obs_history[max(0, len(self.obs_history) - 1 - self.velocity_lag)]
            proprio[25:44] = delayed[25:44]
        self.obs_history = self.obs_history[-(max(self.obs_lag, self.velocity_lag) + 2):]
        return np.concatenate([proprio, self.raw, command]).astype(np.float32)[None]

    def step(self, command, amplitude=1.0):
        obs = self.obs(command)
        self.raw = self.session.run(None, {self.session.get_inputs()[0].name: obs})[0][0]
        if not np.isfinite(self.raw).all():
            raise ValueError('Nonfinite policy output')
        target = self.home + self.raw * self.scale * amplitude
        # Optional sensor-driven actuator adapter. Raw policy last_action stays
        # unchanged; the resulting target uses the same filter, bus and servo.
        adapter = getattr(self, 'transform_target', None)
        if adapter is not None:
            target = adapter(target, obs[0])
        alpha = self.contract['action_filter']['alpha_new']
        smoothed = self.applied + alpha * (target - self.applied) if alpha > 0 else target
        limit = self.contract['action_filter']['max_delta_rad']
        self.applied += np.clip(smoothed - self.applied, -limit, limit)
        for _ in range(self.decimation):
            self.command_history.append(self.applied.copy())
            delayed = self.command_history[max(0, len(self.command_history) - 1 - self.command_lag)]
            transform = getattr(self, 'before_physics_step', None)
            if transform is not None:
                delayed = transform(delayed)
            self.data.ctrl[self.aids] = delayed
            mujoco.mj_step(self.model, self.data)
            callback = getattr(self, 'substep_callback', None)
            if callback is not None:
                callback()
            self.command_history = self.command_history[-(self.command_lag + 2):]
        # mjlab refreshes derived quantities after its decimation loop.
        mujoco.mj_forward(self.model, self.data)

    def sample(self):
        d, m = self.data, self.model
        r = d.xmat[self.body].reshape(3, 3)
        yaw = np.arctan2(r[1, 0], r[0, 0])
        local_velocity = r.T @ d.qvel[:3]
        tilt = np.rad2deg(np.arccos(np.clip(r[2, 2], -1, 1)))
        head_vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.head, head_vel, 0)
        contact = [False, False]
        for c in d.contact:
            for i, gid in enumerate(self.foot_geoms):
                if c.geom1 == gid or c.geom2 == gid:
                    contact[i] = True
        return np.r_[d.qpos[:3], yaw, local_velocity[:2], tilt,
                     d.xpos[self.feet, 2], contact,
                     d.qpos[self.jadr[self.names.index('tail_pitch')]],
                     np.linalg.norm(head_vel[:3]),
                     np.mean(np.abs(d.actuator_force[self.aids]) >= .599)]


def run_trial(sim, command, seed=0, perturb=False, amplitude=1, seconds=5):
    sim.reset(seed, perturb)
    for _ in range(round(2 / sim.dt)):
        sim.step(command, amplitude)
    points = [sim.sample()]
    for _ in range(round(seconds / sim.dt)):
        sim.step(command, amplitude)
        points.append(sim.sample())
    a = np.array(points)
    yaw = np.unwrap(a[:, 3])
    lifts = np.ptp(a[:, 7:9], axis=0) * 1000
    dy = a[-1, :2] - a[0, :2]
    heading = a[0, 3]
    return {'seed': seed, 'perturbed': perturb, 'amplitude': amplitude,
            'yaw_drift_deg': float(np.rad2deg(yaw[-1] - yaw[0])),
            'yaw_rate_rad_s': float((yaw[-1] - yaw[0]) / seconds),
            'vx_body_m_s': float(a[:, 4].mean()),
            'vy_body_m_s': float(a[:, 5].mean()),
            'lateral_displacement_mm': float((-np.sin(heading) * dy[0] + np.cos(heading) * dy[1]) * 1000),
            'horizontal_speed_m_s': float(np.linalg.norm(a[:, 4:6], axis=1).mean()),
            'tilt_max_deg': float(a[:, 6].max()),
            'fell': (sim.fall_test(a) if hasattr(sim, 'fall_test') else
                     bool(np.any(a[:, 2] < .06) or np.any(a[:, 6] > 60))),
            'foot_lift_mm': lifts.tolist(),
            'lift_asymmetry': float(max(lifts) / max(min(lifts), 1e-6)),
            'contact_fraction_LR': a[:, 9:11].mean(axis=0).tolist(),
            'tail_mean_deg': float(np.rad2deg(a[:, 11].mean())),
            'head_angular_speed_rms_deg_s': float(np.rad2deg(np.sqrt(np.mean(a[:, 12] ** 2)))),
            'torque_saturation_fraction': float(a[:, 13].mean())}


def evaluate(sim, trials):
    result = {'stand': [], 'forward': [], 'turn': [], 'tail_up': []}
    for seed in range(trials):
        result['stand'].append(run_trial(sim, np.zeros(18), seed, seed > 0))
    for amplitude in (1., 1.15):
        for seed in range(min(trials, 3)):
            cmd = np.zeros(18); cmd[0] = .55
            result['forward'].append(run_trial(sim, cmd, seed, seed > 0, amplitude, 6))
    for rate in (.9, -.9):
        cmd = np.zeros(18); cmd[2] = rate
        row = run_trial(sim, cmd, seconds=4)
        row['command_yaw_rate'] = rate
        result['turn'].append(row)
    for angle in (1.4, 1.57, 1.75):
        cmd = np.zeros(18); cmd[0] = .35; cmd[16] = angle
        row = run_trial(sim, cmd, seconds=6)
        row['command_tail_rad'] = angle
        result['tail_up'].append(row)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plant', required=True)
    p.add_argument('--onnx', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--trials', type=int, default=5)
    p.add_argument('--command-lag', type=int, default=0)
    p.add_argument('--obs-lag', type=int, default=0)
    args = p.parse_args()
    sim = Sim(args.plant, args.onnx, args.command_lag, args.obs_lag)
    result = {'onnx_sha256': sha(args.onnx), 'plant_sha256': sha(Path(args.plant) / 'nominal.mjb'),
              'onnx': args.onnx, 'contract': sim.contract,
              'command_delay_ms': args.command_lag * sim.model.opt.timestep * 1000,
              'joint_feedback_delay_ms': args.obs_lag * sim.dt * 1000,
              'trials': evaluate(sim, args.trials)}
    # Reset reproducibility is a behavior check, not a state-only assertion.
    first = run_trial(sim, np.zeros(18), seconds=1)
    cmd = np.zeros(18); cmd[2] = .9
    run_trial(sim, cmd, seconds=1)
    again = run_trial(sim, np.zeros(18), seconds=1)
    result['reset_reproducible'] = first == again
    assert result['reset_reproducible'], 'Reset retains history from prior trajectory'
    Path(args.out).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'out': args.out, 'stand_yaw_deg': [r['yaw_drift_deg'] for r in result['trials']['stand']],
                      'forward_vx': [r['vx_body_m_s'] for r in result['trials']['forward']],
                      'tail_vx': [r['vx_body_m_s'] for r in result['trials']['tail_up']],
                      'falls': sum(r['fell'] for rows in result['trials'].values() for r in rows)}, indent=2))


if __name__ == '__main__':
    main()
