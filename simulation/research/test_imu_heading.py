import unittest
from collections import deque
from pathlib import Path
import numpy as np
from imu_heading import RelativeImuHeading,HeadingController,from_rpy,matrix,wrap
from heading_sim import HeadingExperiment,WARMUP_SECONDS
from hardware_sim import HardwareSim,HardwareCase


class HeadingTests(unittest.TestCase):
    def calibrated(self,bias=(0,0,0),roll=0.,pitch=0.):
        e=RelativeImuHeading();r=matrix(from_rpy(roll,pitch))
        e.calibrate(np.tile(bias,(50,1)),np.tile(r.T@[0,0,9.81],(50,1)),0.)
        return e,r

    def test_full_3d_rotation_crosses_wrap_with_tilt_and_calibrated_bias(self):
        bias=np.array([.003,-.004,.008]);e,r=self.calibrated(bias,.3,-.2)
        rate=.4
        for i in range(1000):e.update(r.T@[0,0,rate]+bias,r.T@[0,0,9.81],(i+1)*.02)
        self.assertAlmostEqual(e.yaw,8.,places=8)
        self.assertAlmostEqual(np.linalg.norm(e.q),1.,places=12)

    def test_acceleration_rejected_and_bias_not_learned_from_motion(self):
        e,_=self.calibrated()
        for i in range(100):e.update([0,0,.2],[9.81,0,19.62],(i+1)*.02)
        self.assertEqual(e.accel_rejections,100);self.assertAlmostEqual(e.yaw,.4,places=9)
        np.testing.assert_array_equal(e.bias,np.zeros(3))

    def test_timestamps_and_reinitialization_do_not_accumulate_history(self):
        e,_=self.calibrated();e.update([0,0,1],[0,0,9.81],.02)
        first=e.yaw;e.update([0,0,1],[0,0,9.81],.02);self.assertEqual(first,e.yaw)
        with self.assertRaises(ValueError):e.update([0,0,1],[0,0,9.81],.01)
        e.update([0,0,1],[0,0,9.81],.3);self.assertEqual(first,e.yaw);self.assertEqual(e.gap_count,1)
        e.calibrate(np.zeros((50,3)),np.tile([0,0,9.81],(50,1)),1.)
        self.assertEqual(e.yaw,0);self.assertEqual(e.gap_count,0)

    def test_moving_calibration_is_rejected(self):
        e=RelativeImuHeading();gyro=np.zeros((50,3));gyro[:,0]=np.linspace(-.1,.1,50)
        with self.assertRaises(ValueError):e.calibrate(gyro,np.tile([0,0,9.81],(50,1)),0.)

    def test_reference_follows_user_intent_not_correction(self):
        c=HeadingController()
        for _ in range(100):c.update(-.2,0,.02)
        self.assertEqual(c.reference,0);self.assertGreater(c.output,0)
        c.reset()
        for _ in range(100):c.update(-.2,.45,.02)
        for _ in range(50):c.update(-.2,0.,.02)
        self.assertAlmostEqual(c.reference,.9,places=10)
        c.reset();self.assertEqual(c.integral,0);self.assertEqual(c.output,0)
        self.assertAlmostEqual(wrap(np.deg2rad(179)-np.deg2rad(-179)),np.deg2rad(-2))

    def test_delayed_plant_converges_without_unbounded_commands(self):
        c=HeadingController();yaw=0.;delay=deque([0.]*10);outputs=[]
        for _ in range(1500):
            output=c.update(yaw,0.,.02);outputs.append(output)
            delay.append(output);yaw+=(.7*delay.popleft()-.08)*.02
        self.assertLess(abs(yaw),.012)
        self.assertLessEqual(max(abs(np.array(outputs))),.7)
        self.assertLessEqual(max(abs(np.diff(np.r_[0,outputs])))/.02,1.500001)
        for _ in range(100):c.update(1.,0.,.02,measurement_age=.3)
        self.assertTrue(c.stale);self.assertEqual(c.integral,0);self.assertAlmostEqual(c.output,0.,places=10)

    def test_sensor_observer_is_read_only_and_trial_reset_repeats(self):
        root=Path(__file__).parent/'20260913_handoff';case=HardwareCase(physics_dt=.00125)
        e=HeadingExperiment(root/'native_v07',root/'v7_reference.onnx',case)
        first,trace=e.run('straight','open',0,1.,True)
        reference=HardwareSim(root/'native_v07',root/'v7_reference.onnx',case)
        for _ in range(round(WARMUP_SECONDS/reference.dt)):reference.step(np.zeros(18))
        command=np.zeros(18);command[0]=.55
        for _ in range(50):reference.step(command)
        np.testing.assert_array_equal(reference.data.qpos,e.sim.data.qpos)
        e.run('left_then_hold','imu',0,3.)
        again,again_trace=e.run('straight','open',0,1.,True)
        np.testing.assert_array_equal(trace,again_trace);self.assertEqual(first,again)
        self.assertGreaterEqual(first['sensor_age_max_ms'],19.999)

    def test_poisoned_scoring_yaw_cannot_change_imu_feedback(self):
        root=Path(__file__).parent/'20260913_handoff';case=HardwareCase(physics_dt=.00125)
        clean=HeadingExperiment(root/'native_v07',root/'v7_reference.onnx',case)
        poisoned=HeadingExperiment(root/'native_v07',root/'v7_reference.onnx',case)
        original=poisoned.sim.view
        def false_scoring_state(t):
            a=original(t);a[4]+=.5*np.sin(t);a[5]-=.4*np.cos(t)
            return a
        poisoned.sim.view=false_scoring_state
        _,a=clean.run('straight','imu',1,2.,True)
        _,b=poisoned.run('straight','imu',1,2.,True)
        np.testing.assert_array_equal(a[:,6],b[:,6])
        np.testing.assert_array_equal(clean.sim.data.qpos,poisoned.sim.data.qpos)


if __name__=='__main__':unittest.main()
