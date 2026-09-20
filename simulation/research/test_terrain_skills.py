import unittest
from types import SimpleNamespace as NS
import torch
import numpy as np
from terrain_skill_cfg import build_config
from mjlab_microduck.tasks.mdp import local_body_height_tracking, SmoothCrouchCommandCfg


class TerrainSkillTests(unittest.TestCase):
    def test_rough_grid_changes_height_along_forward_path(self):
        from terrain_skill_eval import ground_height
        h = ground_height('roughgrid_8mm', .3+(np.arange(16)+.5)*.25, .05)
        self.assertGreater(len(np.unique(h)), 4)
        self.assertGreaterEqual(h.min(), .002)
        self.assertLessEqual(h.max(), .008)

    def test_positive_millisecond_contract_and_current_v07(self):
        for skill in ('terrain', 'crouch'):
            _, c = build_config(skill)
            self.assertAlmostEqual(c.env.decimation*c.env.sim.mujoco.timestep, .02)
            for a in c.env.scene.entities['robot'].articulation.actuators:
                self.assertAlmostEqual(a.delay_min_lag*.00125, .005)
                self.assertAlmostEqual(a.delay_max_lag*.00125, .015)
                self.assertGreater(a.damping, 0)
            self.assertEqual(c.env.observations['actor'].terms['joint_pos'].delay_min_lag, 1)
            self.assertNotIn('body_clearance', c.env.observations['actor'].terms)
            self.assertNotIn('lean_drift', c.env.rewards)
            self.assertNotIn('body_pose_range', c.env.curriculum)
            self.assertEqual(c.env.scene.entities['robot'].spec_fn().modelname, 'microdinosaur_v07')

    def test_height_reward_follows_local_ground_and_command(self):
        command = torch.zeros(3, 6); command[:, 2] = torch.tensor([0., -.01, -.02])
        sensor = NS(data=NS(heights=torch.tensor([[.117182], [.107182], [.097182]])))
        env = NS(scene={'body_clearance': sensor}, command_manager=NS(get_command=lambda n: command))
        self.assertTrue(torch.allclose(local_body_height_tracking(env), torch.ones(3)))
        command[:, 2] = 0
        r = local_body_height_tracking(env)
        self.assertGreater(r[0], r[1]); self.assertGreater(r[1], r[2])

    def test_crouch_resampling_slew_and_partial_reset(self):
        env = NS(num_envs=8, device='cpu', step_dt=.02, episode_length_buf=torch.ones(8, dtype=torch.long))
        cfg = SmoothCrouchCommandCfg(resampling_time_range=(2., 4.), ranges=((0., 0.),)*6)
        term = cfg.build(env); ids = torch.arange(8)
        term._resample_command(ids)
        for _ in range(60):
            old = term.command.clone(); term._update_command()
            self.assertLessEqual(float(abs(term.command-old).max()), .00040001)
        old = term.command.clone(); env.episode_length_buf[:] = 10
        term._resample_command(ids); self.assertTrue(torch.equal(old, term.command))
        env.episode_length_buf[1] = 0; term._resample_command(torch.tensor([1]))
        self.assertTrue(torch.equal(term.command[1], torch.zeros(6)))
        self.assertTrue(torch.equal(old[2:], term.command[2:]))


if __name__ == '__main__': unittest.main()
