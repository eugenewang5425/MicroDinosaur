import unittest
from pathlib import Path
import mujoco
import numpy as np
import torch
from foot_flight_cfg import FootProgress,FootGeometry,FoldCommandCfg,build_config
from types import SimpleNamespace
from probe_tuck_jump import feet_metrics


class FeetTests(unittest.TestCase):
    def test_stand_one_foot_body_support_and_unarmed_air_earn_nothing(self):
        s=FootProgress(4,'cpu');s.grounded[:3]=True
        gaps=torch.tensor([[0.,0.],[.03,0.],[.03,.03],[.03,.03]])
        for i in range(100):s.update(i,.00125,gaps,torch.tensor([10.,0.,3.,0.]),torch.ones(4,dtype=torch.bool),torch.ones(4,dtype=torch.bool))
        self.assertEqual(float(s.height_gain.sum()+s.duration_gain.sum()+s.goal_bonus.sum()),0.)
    def test_contiguous_both_feet_goal_once_and_partial_reset(self):
        s=FootProgress(2,'cpu');s.grounded[:]=True;yes=torch.ones(2,dtype=torch.bool)
        for i in range(48):s.update(i,.00125,torch.full((2,2),.012),torch.zeros(2),yes,yes)
        self.assertTrue(s.qualified.all());torch.testing.assert_close(s.goal_bonus,torch.ones(2))
        before=s.run5.clone();s.update(47,.00125,torch.full((2,2),.012),torch.zeros(2),yes,yes)
        torch.testing.assert_close(s.run5,before)
        s.reset(torch.tensor([0]));self.assertFalse(s.grounded[0]);self.assertTrue(s.qualified[1])
        s.begin_control();self.assertEqual(float(s.goal_bonus.sum()),0.)
    def test_peak_and_duration_cannot_be_taken_from_different_hops(self):
        s=FootProgress(1,'cpu');s.grounded[:]=True;yes=torch.ones(1,dtype=torch.bool)
        for i in range(20):s.update(i,.00125,torch.full((1,2),.015),torch.zeros(1),yes,yes)
        s.update(20,.00125,torch.zeros((1,2)),torch.ones(1),yes,yes)
        for i in range(21,101):s.update(i,.00125,torch.full((1,2),.007),torch.zeros(1),yes,yes)
        self.assertFalse(s.qualified[0]);self.assertAlmostEqual(float(s.run5[0]),.1,places=5)
    def test_convex_hull_minimum_matches_every_original_mesh_vertex(self):
        path=Path(__file__).parent/'20260914_run_jump/plant/nominal.mjb'
        m=mujoco.MjModel.from_binary_path('x.mjb',assets={'x.mjb':path.read_bytes()});g=FootGeometry(m,'cpu')
        rng=np.random.default_rng(41);axes=rng.normal(size=(300,3)).astype(np.float32);axes/=np.linalg.norm(axes,axis=1)[:,None]
        for gid,hull in zip(g.gids,g.vertices):
            mesh=m.geom_dataid[gid];start=m.mesh_vertadr[mesh];v=m.mesh_vert[start:start+m.mesh_vertnum[mesh]]
            np.testing.assert_allclose((axes@v.T).min(-1),(axes@hull.numpy().T).min(-1),atol=1e-7)
    def test_cpu_goal_does_not_use_com_or_single_foot_lift(self):
        a=np.zeros((100,16));a[:,0]=np.arange(100)*.00125;a[:,3]=np.arange(100)*.002
        a[:,14]=.02;a[:,15]=0.;self.assertFalse(feet_metrics(a,.00125)['feet_goal'])
        a[:,15]=.012;self.assertTrue(feet_metrics(a,.00125)['feet_goal'])
        a[:,7]=2.;a[:,4]=2.;self.assertFalse(feet_metrics(a,.00125)['feet_goal'])
    def test_no_positive_com_launch_reward_and_original_physics(self):
        _,cfg=build_config(8)
        self.assertNotIn('hop_launch',cfg.env.rewards);self.assertNotIn('hop_flight',cfg.env.rewards)
        self.assertEqual(cfg.env.commands['body_pose'].deepest_m,.05)
        self.assertEqual(cfg.env.sim.mujoco.timestep,.00125)
        for a in cfg.env.scene.entities['robot'].articulation.actuators:
            self.assertEqual((a.delay_min_lag,a.delay_max_lag),(4,12));self.assertGreater(a.damping,0)
    def test_depth_curriculum_uses_resumed_step_offset(self):
        env=SimpleNamespace(num_envs=8,device='cpu',step_dt=.02,common_step_counter=399864)
        cmd=FoldCommandCfg(skill='jump',width=6,resampling_time_range=(100.,100.),deepest_m=.05,source_step=399864).build(env)
        cmd.reset(torch.arange(8));self.assertTrue(((cmd.depth>=.028)&(cmd.depth<=.03)).all())
        env.common_step_counter+=240*24;cmd.reset(torch.arange(8))
        self.assertTrue(((cmd.depth>=.048)&(cmd.depth<=.05)).all())


if __name__=='__main__':unittest.main()
