import unittest
from types import SimpleNamespace as NS
import numpy as np
import torch
from jump_refine_cfg import torque_bounds_torch,JumpProgress,whole_com,JumpCommandCfg,build_config
from hardware_sim import torque_bounds


class JumpRefineTests(unittest.TestCase):
    def test_motor_envelope_matches_cpu_and_keeps_braking(self):
        v=np.linspace(-30,30,121)
        for voltage in [9.9,11.1,12.6]:
            for effort in [.8,1.]:
                lo,hi=torque_bounds(v,voltage,effort)
                a,b=torque_bounds_torch(torch.tensor(v),torch.tensor(voltage),torch.tensor(.6*effort))
                np.testing.assert_allclose(a.numpy(),lo,atol=1e-7)
                np.testing.assert_allclose(b.numpy(),hi,atol=1e-7)
                self.assertEqual(float(b[-1]),0.);self.assertLess(float(a[-1]),0.)
                self.assertEqual(float(a[0]),0.);self.assertGreater(float(b[0]),0.)

    def test_whole_com_uses_all_body_masses_and_velocity(self):
        mass=torch.tensor([[1.,3.]])
        vel=torch.tensor([[[0.,0.,1.],[0.,0.,-1.]]])
        robot=NS(indexing=NS(body_ids=[0,1]),data=NS(body_com_lin_vel_w=vel))
        env=NS(scene={'robot':robot},sim=NS(model=NS(body_mass=mass),data=NS(xipos=torch.tensor([[[0.,0.,0.],[0.,0.,2.]]]))))
        pos,v=whole_com(env)
        self.assertEqual(float(pos[0,2]),1.5);self.assertEqual(float(v[0,2]),-.5)

    def test_history_invalid_launch_and_partial_reset(self):
        s=JumpProgress(3,'cpu');yes=torch.ones(3,dtype=torch.bool);no=~yes
        s.update(1,.02,torch.tensor([.3,.3,-.1]),no,yes,yes,torch.tensor([True,False,True]))
        torch.testing.assert_close(s.flight,torch.tensor([.02,0.,0.]))
        keep=s.dv.clone();s.update(1,.02,torch.ones(3),no,yes,yes,yes)
        torch.testing.assert_close(s.dv,keep)
        s.reset(torch.tensor([1]));self.assertAlmostEqual(float(s.flight[0]),.02,places=6)
        s.update(2,.02,torch.tensor([.2,.2,.2]),no,yes,yes,yes)
        self.assertEqual(float(s.dv[0]),0.)
        s.reset(torch.tensor([0]));self.assertFalse(s.launched[0]);self.assertGreater(float(s.flight[1]),0.)

    def test_command_schedule_and_physics_contract(self):
        env=NS(num_envs=8,device='cpu',step_dt=.02)
        cmd=JumpCommandCfg(skill='jump',width=6,resampling_time_range=(100.,100.)).build(env)
        cmd.reset(torch.arange(8));self.assertTrue(((cmd.depth>=.025)&(cmd.depth<=.03)).all())
        self.assertTrue(((cmd.preparation>=1.2)&(cmd.preparation<=1.8)).all())
        for _ in range(140):cmd._update_command()
        self.assertTrue((cmd.command[:,2]==0).all())
        _,cfg=build_config(8)
        self.assertEqual(cfg.env.sim.mujoco.timestep,.00125)
        self.assertEqual(cfg.env.episode_length_s,6.)
        self.assertEqual(cfg.agent.algorithm.class_name,'PPO')
        self.assertNotIn('reset_hop',cfg.env.events)
        self.assertIn('joint_limit_failure',cfg.env.terminations)
        for a in cfg.env.scene.entities['robot'].articulation.actuators:
            self.assertEqual((a.delay_min_lag,a.delay_max_lag),(4,12));self.assertGreater(a.damping,0.)


if __name__=='__main__':unittest.main()
