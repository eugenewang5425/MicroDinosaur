"""Independent NumPy parity and causal transport/reset checks."""
import unittest
from pathlib import Path
import numpy as np
import torch
import mujoco
import re
from head_attitude import HeadController, HeadConfig, rotation_exp
from imu_heading import RelativeImuHeading, matrix
from mjlab_microduck.head_imu_action import static_kinematics
from mjlab_microduck.head_attitude_torch import TorchHeadController, TorchImu, exp_so3, log_so3
from deep_crouch_cfg import build_config
from types import SimpleNamespace as NS
from mjlab_microduck.tasks.mdp import CrouchCycleCommandCfg

ROOT = Path(__file__).parent


def kinematics():
    from head_attitude import HeadKinematics
    m = mujoco.MjModel.from_binary_path('nominal.mjb', assets={'nominal.mjb':
        (ROOT/'20260913_handoff/native_v07/nominal.mjb').read_bytes()})
    k = static_kinematics(m)
    return HeadKinematics(k.fixed, k.axes, k.reference, k.tip, k.base, k.limits)


class DeepCrouchTests(unittest.TestCase):
    def test_numpy_control_parity_stale_limits_and_partial_reset(self):
        rng = np.random.default_rng(914)
        k = kinematics(); n = 4
        cpu = [HeadController(k, HeadConfig(max_measurement_age_s=.04)) for _ in range(n)]
        batch = TorchHeadController(k, n, 'cpu', torch.float64)
        tensor = lambda a: torch.as_tensor(a, dtype=torch.float64)
        initial = np.array([[.3491, 0., 0.]]*n)
        batch.reset(slice(None), tensor(initial))
        for c in cpu: c.reset(initial[0])
        maximum = 0.
        for step in range(160):
            if step == 80:
                before = batch.correction.clone()
                batch.reset(torch.tensor([1]), tensor(initial[1:2])); cpu[1].reset(initial[1])
                np.testing.assert_array_equal(before[[0, 2, 3]], batch.correction[[0, 2, 3]])
            nominal = initial+rng.normal(0, .8, (n, 3))
            angles = rng.uniform(-.5, .5, (n, 4)); gyro = rng.normal(0, 2, (n, 3))
            orientation = np.array([rotation_exp(v) for v in rng.normal(0, .5, (n, 3))])
            desired = np.array([rotation_exp(v) for v in rng.normal(0, .3, (n, 3))])
            omega = rng.normal(0, .3, (n, 3)); age = np.array([.02, .04, .08, .2])
            actual = batch.update(*map(tensor, (nominal, angles, orientation, gyro, desired, omega)), .02, tensor(age))
            expected = np.array([cpu[i].update(nominal[i], angles[i], orientation[i], gyro[i], desired[i], omega[i], .02, age[i]) for i in range(n)])
            maximum = max(maximum, float(abs(actual.numpy()-expected).max()))
            np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
        print('controller_max_error_rad', maximum)

    def test_imu_packet_update_parity(self):
        cpu = RelativeImuHeading()
        accel = np.array([.2, -.4, 9.79]); gyro = np.array([.001, -.002, .003])
        cpu.calibrate(np.tile(gyro, (50, 1)), np.tile(accel, (50, 1)), 0.)
        batch = TorchImu(1, 'cpu', torch.float64)
        batch.initialize([0], torch.tensor(accel[None]), torch.tensor(gyro[None]))
        for i in range(1, 500):
            g = gyro+np.array([.03*np.sin(i*.1), .04*np.cos(i*.1), .1])
            a = accel+np.array([3*np.sin(i*.05), 0., 0.])
            cpu.update(g, a, i*.02)
            batch.update(torch.tensor(g[None]), torch.tensor(a[None]), .02)
            np.testing.assert_allclose(batch.r[0], matrix(cpu.q), atol=1e-10)

    def test_so3_zero_and_near_pi(self):
        v = torch.tensor([[0., 0., 0.], [np.pi-1e-7, 0., 0.], [0., -np.pi+1e-7, 0.]], dtype=torch.float64)
        torch.testing.assert_close(exp_so3(log_so3(exp_so3(v))), exp_so3(v), atol=1e-8, rtol=1e-8)

    def test_cycle_slew_return_and_reset_history(self):
        env = NS(num_envs=4, device='cpu', step_dt=.02, episode_length_buf=torch.ones(4, dtype=torch.long))
        term = CrouchCycleCommandCfg(ranges=((0., 0.),)*6, resampling_time_range=(100., 100.)).build(env)
        term._height_target[:] = torch.tensor([0., -.02, -.03, -.04])
        for step in range(600):
            previous = term.command.clone(); term._update_command()
            self.assertLessEqual(float((previous-term.command).abs().max()), .00040001)
            if step == 300: torch.testing.assert_close(term.command[:, 2], term._height_target)
        torch.testing.assert_close(term.command, torch.zeros(4, 6))
        elapsed = term.elapsed.clone(); term.reset(torch.tensor([1]))
        torch.testing.assert_close(term.elapsed[[0, 2, 3]], elapsed[[0, 2, 3]])
        self.assertEqual(float(term.elapsed[1]), 0.)

    def test_recipe_positive_delays_and_frozen_actor(self):
        _, cfg = build_config(8)
        self.assertEqual(cfg.env.actions['joint_pos'].__class__.__name__, 'HeadImuPositionActionCfg')
        self.assertEqual(cfg.env.commands['body_pose'].height_buckets, (0., -.02, -.03, -.04))
        for a in cfg.env.scene.entities['robot'].articulation.actuators:
            self.assertEqual((a.delay_min_lag, a.delay_max_lag), (4, 12))
            self.assertGreater(a.damping, 0)
        self.assertNotIn('lean_drift', cfg.env.rewards)
        self.assertNotIn('head_imu', cfg.env.observations['actor'].terms)

    def test_crouch_pose_reference_does_not_mutate_home(self):
        _, cfg = build_config(3)
        reward = cfg.env.rewards['pose']
        names = reward.params['reference_names']
        def find(patterns, preserve_order=False):
            indices = []
            for pattern in patterns:
                indices.extend(i for i, name in enumerate(names) if re.fullmatch(pattern, name))
            if not preserve_order: indices.sort()
            return indices, [names[i] for i in indices]
        home = torch.tensor(reward.params['reference_table'][0]).repeat(3, 1)
        original = home.clone()
        actual = torch.tensor([reward.params['reference_table'][i] for i in (0, 2, 4)])
        asset = NS(data=NS(default_joint_pos=home, joint_pos=actual), find_joints=find)
        commands = {'twist': torch.zeros(3, 3), 'body_pose': torch.zeros(3, 6)}
        commands['body_pose'][:, 2] = torch.tensor([0., -.02, -.04])
        env = NS(scene={'robot': asset}, device='cpu', command_manager=NS(get_command=lambda n: commands[n]))
        reward.params['asset_cfg'].joint_ids = find(reward.params['asset_cfg'].joint_names)[0]
        term = reward.func(reward, env)
        torch.testing.assert_close(term(env, **reward.params), torch.ones(3))
        torch.testing.assert_close(home, original, atol=0, rtol=0)
        asset.data.joint_pos = home.clone()
        scores = term(env, **reward.params)
        self.assertEqual(float(scores[0]), 1.)
        self.assertLess(float(scores[2]), .1)
        torch.testing.assert_close(home, original, atol=0, rtol=0)


if __name__ == '__main__': unittest.main()
