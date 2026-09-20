import unittest
from types import SimpleNamespace as NS
import torch
import numpy as np
from run_jump_cfg import HopState,MotionCommandCfg,build_config


class RunJumpTests(unittest.TestCase):
    def test_cpu_flight_rejects_floor_grazing_and_marks_falling_separately(self):
        from evaluate_run_jump import flight_segments
        a=np.zeros((100,16));a[:,0]=np.arange(100)*.00125
        a[:,14:16]=.003;a[:,2]=.2
        good=flight_segments(a,.00125)
        self.assertEqual(len(good),1)
        self.assertAlmostEqual(good[0]['duration_s'],.125)
        self.assertEqual(good[0]['takeoff_com_velocity_m_s'],0.)
        a[:,14]=.001
        self.assertEqual(flight_segments(a,.00125),[])

    def test_takeoff_uses_whole_robot_com_not_relative_trunk_motion(self):
        from evaluate_run_jump import flight_segments
        a=np.zeros((100,16));a[:,0]=np.arange(100)*.00125
        a[:,14:16]=.003;a[:,2]=.5;a[:,3]=.15-.2*a[:,0]
        segment=flight_segments(a,.00125)[0]
        self.assertGreater(segment['takeoff_up_velocity_m_s'],0.)
        self.assertAlmostEqual(segment['takeoff_com_velocity_m_s'],-.2,places=9)
        a[:,14]=.003;a[:,2]=-.4
        self.assertLess(flight_segments(a,.00125)[0]['takeoff_up_velocity_m_s'],0.)
        a[:,4]=1.
        self.assertEqual(flight_segments(a,.00125),[])

    def test_hop_state_does_not_pay_falling_or_repeat_a_frontier(self):
        s=HopState(3,'cpu');dt=.02
        vz=torch.tensor([.2,-.3,.2]);support=torch.tensor([False,False,True])
        yes=torch.ones(3,dtype=torch.bool)
        s.update(1,dt,vz,support,yes,yes)
        torch.testing.assert_close(s.flight,torch.tensor([.02,0.,0.]))
        old=s.flight.clone();s.update(1,dt,vz,support,yes,yes)
        torch.testing.assert_close(s.flight,old)
        s.update(2,dt,vz,support,yes,yes)
        torch.testing.assert_close(s.dv,torch.zeros(3))
        self.assertAlmostEqual(float(s.flight_frontier[0]),.04,places=6)
        s.reset(torch.tensor([0]));self.assertEqual(float(s.velocity_frontier[0]),0)
        self.assertAlmostEqual(float(s.velocity_frontier[2]),.2,places=6)
        self.assertFalse(bool(s.launched[0]))

    def test_flight_requires_upright_launch_window_and_real_no_support(self):
        s=HopState(3,'cpu')
        s.update(1,.02,torch.ones(3),torch.zeros(3,dtype=torch.bool),
                 torch.tensor([True,False,True]),torch.tensor([False,True,True]))
        torch.testing.assert_close(s.flight,torch.tensor([0.,0.,.02]))
        for step in range(2,40):
            s.update(step,.02,torch.zeros(3),torch.zeros(3,dtype=torch.bool),
                     torch.ones(3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool))
        self.assertLessEqual(float(s.flight_frontier.max()),.120001)

    def test_commands_have_observable_preparation_launch_and_recovery(self):
        env=NS(num_envs=2,device='cpu',step_dt=.02)
        t=MotionCommandCfg(skill='jump',width=6,resampling_time_range=(100.,100.)).build(env)
        expected={25:-.03,85:.02,180:0.}
        for k in range(200):
            t._update_command()
            if k in expected:self.assertAlmostEqual(float(t.command[0,2]),expected[k],places=6)
        keep=t.elapsed[1].clone();t.reset(torch.tensor([0]))
        self.assertEqual(float(t.elapsed[0]),0);self.assertEqual(t.elapsed[1],keep)
        old=t.elapsed.clone();t.compute(0.);torch.testing.assert_close(t.elapsed,old)

    def test_run_accelerates_and_requests_stop(self):
        env=NS(num_envs=1,device='cpu',step_dt=.02)
        t=MotionCommandCfg(skill='run',width=3,resampling_time_range=(100.,100.)).build(env)
        t.speed[:]=.8
        for k in range(500):
            old=t.command.clone();t._update_command()
            self.assertLessEqual(float((t.command-old).abs().max()),.030001)
            if k==200:self.assertAlmostEqual(float(t.command[0,0]),.8,places=6)
        self.assertEqual(float(t.command[0,0]),0.)

    def test_specialists_preserve_physics_and_do_not_train_stairs(self):
        for skill in ('run','jump'):
            _,c=build_config(skill,8)
            self.assertEqual(c.env.scene.terrain.terrain_type,'plane')
            self.assertEqual(c.agent.algorithm.class_name,'PPO')
            self.assertFalse(hasattr(c.agent.algorithm,'anchor_steps'))
            self.assertNotIn('push_robot',c.env.events)
            self.assertNotIn('action_rate_weight',c.env.curriculum)
            self.assertEqual(c.env.actions['joint_pos'].owned_head_indices,(1,))
            self.assertNotIn('head_imu',c.env.observations['actor'].terms)
            for a in c.env.scene.entities['robot'].articulation.actuators:
                self.assertEqual((a.delay_min_lag,a.delay_max_lag),(4,12))
                self.assertAlmostEqual(a.damping,.8002992796237024)
            self.assertIn('nonfoot_contact',c.env.terminations)
            for key in ('action_rate_l2','dof_pos_limits','foot_slip','self_collisions'):
                self.assertLess(c.env.rewards[key].weight,0)


if __name__=='__main__':unittest.main()
