import unittest
from pathlib import Path
import numpy as np
from evaluate_policy import Sim
from hardware_sim import HardwareCase,HardwareSim,torque_bounds

BASE = Path(__file__).parent/'20260913_handoff'


class HardwarePhysicsTests(unittest.TestCase):
    def test_torque_bounds_keep_braking_and_reduce_motoring(self):
        v=np.array([-30.,-8.,0.,8.,30.])
        low,high=torque_bounds(v)
        np.testing.assert_allclose(low,-high[::-1])
        self.assertTrue(np.all(low<=0) and np.all(high>=0))
        self.assertLess(high[3],high[2]);self.assertEqual(high[-1],0)
        self.assertEqual(low[-1],-.6)
        _,low_voltage=torque_bounds(v,voltage=9.9)
        self.assertLess(low_voltage[3],high[3])

    def test_disabled_extensions_match_legacy_dynamics(self):
        a=Sim(BASE/'native_v07',BASE/'v7_reference.onnx',2,1)
        b=HardwareSim(BASE/'native_v07',BASE/'v7_reference.onnx',HardwareCase(protocol=False))
        cmd=np.zeros(18);cmd[0]=.55
        for _ in range(100):a.step(cmd);b.step(cmd)
        np.testing.assert_allclose(a.data.qpos,b.data.qpos,atol=1e-9,rtol=0)

    def test_episode_reset_clears_dropped_packet_and_motor_history(self):
        b=HardwareSim(BASE/'native_v07',BASE/'v7_reference.onnx',
            HardwareCase(motor_curve=True,feedback_drop_probability=.1,bus_stall_ms=40))
        cmd=np.zeros(18);cmd[0]=.55
        def trial():
            b.reset(2,True)
            for _ in range(170):b.step(cmd)
            return b.data.qpos.copy(),b.physics_metrics(1)
        q1,m1=trial();q2,m2=trial()
        np.testing.assert_array_equal(q1,q2)
        self.assertEqual(m1,m2)
        self.assertGreater(m1['max_feedback_age_ms'],20)
        self.assertGreater(m1['positive_mechanical_energy_j'],0)

    def test_slope_changes_actual_contact_frame_and_ground_friction(self):
        b=HardwareSim(BASE/'native_v07',BASE/'v7_reference.onnx',
            HardwareCase(slope_deg=5,ground_friction=.6,physics_dt=.0025))
        gid=b.model.geom('terrain').id
        expected=np.array([np.sin(np.deg2rad(5)),0,np.cos(np.deg2rad(5))])
        np.testing.assert_allclose(b.data.geom_xmat[gid].reshape(3,3)[:,2],expected,atol=1e-12)
        cmd=np.zeros(18)
        for _ in range(100):b.step(cmd)
        contacts=[c for c in b.data.contact if gid in (c.geom1,c.geom2)]
        self.assertTrue(contacts)
        for c in contacts:
            self.assertAlmostEqual(abs(float(np.dot(c.frame[:3],expected))),1.,places=10)
            self.assertAlmostEqual(float(c.friction[0]),.6)
        self.assertAlmostEqual(b.command_lag*b.model.opt.timestep,.01)
        b.reset()
        np.testing.assert_allclose(b.data.geom_xmat[gid].reshape(3,3)[:,2],expected,atol=1e-12)
        # Translate 2m downhill with unchanged .11m normal clearance: world Z
        # is negative, but the robot has not fallen relative to the terrain.
        points=np.zeros((2,7));points[:,:3]=expected*.11
        points[1,:3]+=np.array([2*np.cos(np.deg2rad(5)),0,-2*np.sin(np.deg2rad(5))])
        self.assertFalse(b.fall_test(points))
        points[1,:3]-=expected*.07
        self.assertTrue(b.fall_test(points))


if __name__=='__main__':unittest.main()
