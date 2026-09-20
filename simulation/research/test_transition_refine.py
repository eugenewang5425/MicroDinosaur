import unittest
from types import SimpleNamespace as NS
import torch
import numpy as np
from mjlab_microduck.head_attitude_torch import exp_so3
from mjlab_microduck.calibrated_head_action import SeededImu
from mjlab_microduck.tasks import mdp
from transition_refine_cfg import build_config


class TransitionRefineTests(unittest.TestCase):
    def test_navigation_reward_is_invariant_to_common_world_yaw(self):
        n=4; heading=torch.tensor([0.,.7,-1.,2.])
        term=NS(reward_heading_zero=heading,yaw_reference=torch.tensor([0.,.2,-.3,.4]))
        v=torch.zeros(n,3);v[:,2]=term.reward_heading_zero+term.yaw_reference
        data=NS(site_xmat=exp_so3(v)[:,None])
        env=NS(num_envs=n,device='cpu',action_manager=NS(get_term=lambda name:term),
            sim=NS(data=data,mj_model=NS(site=lambda name:NS(id=0))))
        torch.testing.assert_close(mdp.navigation_head_attitude(env),torch.ones(n))
        offset=torch.tensor([0.,.15,0.]).repeat(n,1)
        data.site_xmat=(exp_so3(v)@exp_so3(offset))[:,None]
        reward=mdp.navigation_head_attitude(env)
        torch.testing.assert_close(reward,torch.full((n,),torch.exp(torch.tensor(-.25))))

    def test_calibration_packet_not_integrated_twice(self):
        imu=SeededImu(3,'cpu');imu.accel[:,2]=9.81
        imu.skip_first[1]=True;before=imu.r.clone()
        gyro=torch.tensor([[0.,0.,.2]]*3);accel=imu.accel.clone()
        imu.update(gyro,accel,.02)
        torch.testing.assert_close(imu.r[1],before[1],atol=0,rtol=0)
        self.assertGreater(float((imu.r[0]-before[0]).abs().max()),.001)
        imu.update(gyro,accel,.02)
        self.assertGreater(float((imu.r[1]-before[1]).abs().max()),.001)

    def test_crouch_stand_resume_and_stop_are_separate_phases(self):
        env=NS(num_envs=4,device='cpu',step_dt=.02,episode_length_buf=torch.zeros(4,dtype=torch.long))
        twist=mdp.TransitionTwistCommandCfg(resampling_time_range=(100.,100.)).build(env)
        env.command_manager=NS(get_term=lambda name:twist)
        height=mdp.TransitionHeightCommandCfg(ranges=((0.,0.),)*6,resampling_time_range=(100.,100.)).build(env)
        twist.mode[:]=torch.arange(4);twist.depth[:]=-.04;twist.request[:,0]=.55
        for step in range(1000):
            previous=height.command.clone();twist._update_command();height._update_command()
            self.assertLessEqual(float((height.command-previous).abs().max()),.00040001)
            if step in (49,499):
                torch.testing.assert_close(twist.command[1:],torch.zeros(3,3))
            if step==250: torch.testing.assert_close(height.command[2:,2],torch.full((2,),-.04))
            if step==600:
                self.assertAlmostEqual(float(twist.command[3,0]),.35,places=6)
                torch.testing.assert_close(height.command,torch.zeros(4,6))
            if step==949: torch.testing.assert_close(twist.command[1:],torch.zeros(3,3))
        old=twist.elapsed.clone();twist.reset(torch.tensor([2]));height.reset(torch.tensor([2]))
        torch.testing.assert_close(twist.elapsed[[0,1,3]],old[[0,1,3]],atol=0,rtol=0)
        old=twist.elapsed.clone();twist_cmd=twist.command.clone();height_cmd=height.command.clone()
        twist.compute(0.);height.compute(0.)
        torch.testing.assert_close(twist.elapsed,old,atol=0,rtol=0)
        torch.testing.assert_close(twist.command,twist_cmd,atol=0,rtol=0)
        torch.testing.assert_close(height.command,height_cmd,atol=0,rtol=0)

    def test_frozen_recipe_keeps_positive_delays_and_separate_head_control(self):
        _,cfg=build_config(8)
        self.assertEqual(cfg.env.rewards['head_world_gaze'].func,mdp.navigation_head_attitude)
        self.assertEqual(cfg.env.rewards['head_pose_tracking'].func,mdp.neck_only_pose_tracking)
        self.assertNotIn('reset_base',cfg.env.events)
        self.assertEqual(cfg.agent.algorithm.anchor_steps,2)
        for actuator in cfg.env.scene.entities['robot'].articulation.actuators:
            self.assertEqual((actuator.delay_min_lag,actuator.delay_max_lag),(4,12))
            self.assertGreater(actuator.damping,0)

    def test_cpu_resume_command_reaches_body_heading_gate_before_stop(self):
        from evaluate_transition_refine import TransitionExperiment
        e=object.__new__(TransitionExperiment)
        e.program='resume';e.shaped_user=np.zeros(18);e.user_rows=[]
        moving=[]
        for tick in range(1000):
            command=e.user_command(tick*.02,'stand')
            moving.append(abs(command[0])>.05)
            self.assertTrue(np.all(command[1:]==0))
        self.assertFalse(any(moving[:550]))
        self.assertTrue(all(moving[565:850]))
        self.assertFalse(any(moving[865:]))


if __name__=='__main__': unittest.main()
