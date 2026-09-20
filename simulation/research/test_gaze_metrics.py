import unittest
import numpy as np
from pathlib import Path
from evaluate_gaze_ablation import optical_angles_and_yaw_rate,complete_stance_durations,GazeSim
from hardware_sim import HardwareSim,HardwareCase


class CameraMetricTests(unittest.TestCase):
    def test_optical_yaw_rate_accounts_for_tilt_and_roll(self):
        f=np.array([.8,.3,.5]);f/=np.linalg.norm(f);w=np.array([.2,-.4,.7])
        yaw,pitch,rate=optical_angles_and_yaw_rate(f,w)
        dt=1e-7;next_f=f+dt*np.cross(w,f)
        expected=(np.arctan2(next_f[1],next_f[0])-yaw)/dt
        self.assertAlmostEqual(rate,expected,places=6)
        # Reflect polar direction and axial angular velocity about y=0.
        reflected=optical_angles_and_yaw_rate(f*[1,-1,1],w*[-1,1,-1])
        np.testing.assert_allclose(reflected,[-yaw,pitch,-rate])

    def test_stance_statistics_exclude_partial_boundary_runs(self):
        result=complete_stance_durations([1,1,0,0,1,1,1,0,1,1],.01)
        self.assertEqual(result,[.03])
        self.assertEqual(complete_stance_durations([1]*10,.01),[])

    def test_camera_observer_preserves_physics_and_uses_forward_home_axis(self):
        root=Path(__file__).parent/'20260913_handoff'
        case=HardwareCase(physics_dt=.00125)
        a=HardwareSim(root/'native_v07',root/'v7_reference.onnx',case)
        b=GazeSim(root/'native_v07',root/'v7_reference.onnx',case)
        np.testing.assert_allclose(b.camera_home_angles,[0,0],atol=1e-6)
        command=np.zeros(18);command[0]=.55
        for _ in range(100):a.step(command);b.step(command)
        np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
        self.assertEqual(len(b.gaze_trace),1600)
        b.reset();self.assertEqual(b.gaze_trace,[])


if __name__=='__main__':unittest.main()
