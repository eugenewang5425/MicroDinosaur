import unittest
from pathlib import Path
import mujoco
import numpy as np
from imu_heading import matrix, from_rpy
from head_attitude import HeadController, HeadConfig, rotation_exp, rotation_vector
from head_attitude_sim import HeadExperiment, cad_kinematics
from run_heading_stable_start import StableStartExperiment
from hardware_sim import HardwareCase

ROOT = Path(__file__).parent/'20260913_handoff'


def experiment(enabled=True, config=None):
    return HeadExperiment(ROOT/'native_v07', ROOT/'v7_reference.onnx',
                          HardwareCase(physics_dt=.00125), head_enabled=enabled, head_config=config)


class HeadTests(unittest.TestCase):
    def test_cad_fk_and_jacobian_match_random_physical_poses(self):
        e = experiment(); s = e.sim; rng = np.random.default_rng(241)
        kin = e.kinematics
        for _ in range(30):
            q = rng.uniform(kin.limits[:, 0]+.1, kin.limits[:, 1]-.1)
            s.data.qpos[s.jadr[e.chain_indices]] = q
            s.data.qpos[3:7] = from_rpy(*rng.uniform(-.4, .4, 3))
            mujoco.mj_forward(s.model, s.data)
            body = s.data.site_xmat[s.model.site('robot/imu').id].reshape(3, 3)
            head = s.data.site_xmat[s.model.site('robot/head_imu').id].reshape(3, 3)
            r, jac = kin.forward(q)
            np.testing.assert_allclose(r, body.T@head, atol=1e-8)
            for j in range(3):
                shifted = q.copy(); shifted[j+1] += 1e-6
                r2, _ = kin.forward(shifted)
                numerical = rotation_vector(r.T@r2)/1e-6
                np.testing.assert_allclose(jac[:, j], numerical, atol=1e-6)

    def test_rotation_error_across_wrap_and_half_turn(self):
        for vector in ([0, 0, np.pi], [.2, -.1, .3], [1e-9, 0, 0]):
            r = rotation_exp(np.array(vector))
            np.testing.assert_allclose(rotation_exp(rotation_vector(r)), r, atol=1e-8)
        a = matrix(from_rpy(0, 0, np.deg2rad(179)))
        b = matrix(from_rpy(0, 0, np.deg2rad(-179)))
        self.assertAlmostEqual(rotation_vector(a.T@b)[2], np.deg2rad(2))

    def test_disabled_head_control_matches_previous_body_loop(self):
        e = experiment(False)
        old = StableStartExperiment(ROOT/'native_v07', ROOT/'v7_reference.onnx', HardwareCase(physics_dt=.00125))
        _, a = e.run('straight', 'imu', 1, 1., True)
        _, b = old.run('straight', 'imu', 1, 1., True)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(e.sim.data.qpos, old.sim.data.qpos)

    def test_negative_feedback_reduces_a_tilted_static_head_error(self):
        e = experiment(); kin = e.kinematics; c = HeadController(kin)
        q = e.sim.home[e.chain_indices].astype(float); c.reset(q[1:])
        base = matrix(from_rpy(.08, -.06, .1))
        nominal = q[1:].copy()
        first = np.linalg.norm(rotation_vector((base@kin.forward(q)[0]).T))
        for _ in range(150):
            r, _ = kin.forward(q)
            target = c.update(nominal, q, base@r, np.zeros(3), np.eye(3), np.zeros(3), .02, .02)
            q[1:] += .2*(target-q[1:])
        last = np.linalg.norm(rotation_vector((base@kin.forward(q)[0]).T))
        self.assertLess(last, first*.7)

    def test_bounds_stale_release_and_reset(self):
        e = experiment(); c = e.head_controller; q = e.sim.home[e.chain_indices].astype(float)
        c.reset(q[1:]); previous = q[1:].copy()
        for _ in range(100):
            target = c.update(q[1:], q, matrix(from_rpy(.6, .8, 1.)), np.ones(3),
                              np.eye(3), np.zeros(3), .02, .02)
            self.assertLessEqual(np.max(abs(target-previous)), .080000001)
            previous = target.copy()
        self.assertTrue(c.limited)
        for _ in range(100):
            c.update(q[1:], q, np.eye(3), np.zeros(3), np.eye(3), np.zeros(3), .02, .3)
        self.assertTrue(c.stale)
        np.testing.assert_allclose(c.correction, 0., atol=1e-10)
        c.reset(q[1:]); np.testing.assert_array_equal(c.integral, np.zeros(3))
        with self.assertRaises(ValueError):
            c.update(q[1:], q, np.eye(3), np.full(3, np.nan), np.eye(3), np.zeros(3), .02, .02)

    def test_scoring_truth_cannot_change_head_commands_and_reset_repeats(self):
        clean = experiment(); poisoned = experiment()
        original = poisoned.sim.view
        def false_view(t):
            a = original(t); a[4] += .2*np.sin(t); a[5] -= .4*np.cos(t); a[6] += .2
            return a
        poisoned.sim.view = false_view
        first, a = clean.run('straight', 'imu', 1, 1., True)
        _, b = poisoned.run('straight', 'imu', 1, 1., True)
        np.testing.assert_array_equal(a[:, 6], b[:, 6])
        np.testing.assert_array_equal(np.array(clean.head_rows), np.array(poisoned.head_rows))
        np.testing.assert_array_equal(clean.sim.data.qpos, poisoned.sim.data.qpos)
        clean.run('left_then_hold', 'imu', 0, 3.)
        again, trace = clean.run('straight', 'imu', 1, 1., True)
        self.assertEqual(first, again); np.testing.assert_array_equal(a, trace)


if __name__ == '__main__':
    unittest.main()
