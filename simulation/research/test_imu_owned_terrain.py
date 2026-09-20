import unittest
from types import SimpleNamespace as NS
import numpy as np
import torch
import mujoco
import tempfile
from pathlib import Path
from head_attitude import HeadConfig
from imu_owned_head import ImuOwnedHeadController
from mjlab_microduck.imu_owned_head import TorchImuOwnedHeadController
from test_deep_crouch import kinematics
from mjlab_microduck.head_attitude_torch import exp_so3
from mjlab_microduck.tasks import mdp
from terrain_lane_cfg import StepLaneCfg,build_config


class OwnedTerrainTests(unittest.TestCase):
    def test_nan_guard_writes_without_symlinks_and_aborts(self):
        from mjlab.utils.nan_guard import NanGuardCfg
        from research_nan_guard import ResearchNanGuard
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>')
        data=NS(qpos=torch.zeros(1,model.nq),qvel=torch.zeros(1,model.nv),qacc=torch.zeros(1,model.nv),
            qacc_warmstart=torch.zeros(1,model.nv),sensordata=torch.zeros(1,3),ctrl=torch.zeros(1,0))
        with tempfile.TemporaryDirectory() as folder:
            guard=ResearchNanGuard(NanGuardCfg(enabled=True),1,model,output_dir=folder)
            guard.capture(data);self.assertFalse(guard.check_and_dump(data))
            data.qacc[0,0]=float('nan')
            with self.assertRaisesRegex(FloatingPointError,'Non-finite physics'):guard.check_and_dump(data)
            files=list(Path(folder).iterdir());self.assertEqual(len(files),3)
            self.assertFalse(any(p.is_symlink() for p in files))
            with np.load(next(Path(folder).glob('*.npz'))) as z:
                self.assertTrue(np.isnan(z['post_qacc'][0,0]));self.assertTrue(np.isfinite(z['history_qpos']).all())

    def test_controller_parity_ignored_yaw_stale_and_partial_reset(self):
        k=kinematics();n=3;rng=np.random.default_rng(915)
        cpu=[ImuOwnedHeadController(k,HeadConfig(max_measurement_age_s=.04),(1,)) for _ in range(n)]
        batch=TorchImuOwnedHeadController(k,n,'cpu',torch.float64,owned_indices=(1,))
        initial=np.tile([.3491,0.,0.],(n,1));batch.reset(slice(None),torch.tensor(initial))
        for c in cpu:c.reset(initial[0])
        largest=0.
        for step in range(180):
            if step==90:
                old=batch.last_target.clone();batch.reset([1],torch.tensor(initial[1:2]));cpu[1].reset(initial[1])
                torch.testing.assert_close(batch.last_target[[0,2]],old[[0,2]],atol=0,rtol=0)
            nominal=rng.normal(0,.2,(n,3));angles=rng.normal(0,.15,(n,4));angles[:,1]+=.3491
            orientation=exp_so3(torch.tensor(rng.normal(0,.1,(n,3)))).numpy()
            desired=np.tile(np.eye(3),(n,1,1));gyro=rng.normal(0,.3,(n,3));omega=np.zeros((n,3));age=np.array([.02,.02,.08])
            before=batch.last_target.clone()
            actual=batch.update(*map(torch.tensor,(nominal,angles,orientation,gyro,desired,omega)),.02,torch.tensor(age))
            altered=nominal.copy();altered[:,1]+=100  # Ignored yaw must have exactly zero effect.
            expected=np.asarray([cpu[i].update(altered[i],angles[i],orientation[i],gyro[i],desired[i],omega[i],.02,age[i]) for i in range(n)])
            largest=max(largest,float(abs(actual.numpy()-expected).max()))
            np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-9)
            torch.testing.assert_close(actual[2],before[2],atol=0,rtol=0)
            self.assertLessEqual(float(abs(actual-before).max()),.0800000001)
        print('imu_owned_cpu_torch_max_error_rad',largest)

    def test_step_geometry_matches_analytic_height_and_flat_start(self):
        for direction in ('flat','up','down'):
            cfg=StepLaneCfg(size=(6.,2.),direction=direction)
            spec=mujoco.MjSpec();spec.worldbody.add_body(name='terrain')
            output=cfg.function(.7,spec,np.random.default_rng(1));model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
            origin,parts=cfg.profile(.7)
            for left,right,height in parts:
                distance=mujoco.mj_ray(model,data,np.array([(left+right)/2,1.,1.]),np.array([0.,0.,-1.]),None,1,-1,np.array([-1],np.int32))
                self.assertAlmostEqual(1-distance,height,places=9)
            floor=next(z for left,right,z in parts if left<=origin[0]<right)
            self.assertEqual(floor,origin[2]);np.testing.assert_array_equal(origin,output.origin)

    def test_terrain_commands_keep_posture_on_flat_and_stop_after17(self):
        env=NS(num_envs=12,device='cpu',step_dt=.02,scene=NS(terrain=NS(terrain_types=torch.tensor([0]*6+[1]*3+[2]*3))))
        term=mdp.TerrainTransitionTwistCfg(resampling_time_range=(100.,100.)).build(env)
        term.reset(torch.arange(12));self.assertTrue((term.mode[6:]==0).all())
        self.assertTrue((term.request[6:,1:]==0).all())
        for v in term.request[6:,0]:self.assertTrue(any(abs(float(v)-s)<1e-6 for s in (.2,.35,.55)))
        for _ in range(950):term.compute(.02)
        torch.testing.assert_close(term.command,torch.zeros(12,3))
        before=term.elapsed.clone();term.compute(0.);torch.testing.assert_close(term.elapsed,before,atol=0,rtol=0)

    def test_recipe_keeps_hardware_and_teacher(self):
        _,cfg=build_config(8)
        self.assertEqual(cfg.env.actions['joint_pos'].owned_head_indices,(1,))
        self.assertEqual(cfg.agent.algorithm.anchor_steps,2)
        self.assertEqual(cfg.env.scene.terrain.terrain_generator.sub_terrains['flat'].proportion,.5)
        for a in cfg.env.scene.entities['robot'].articulation.actuators:
            self.assertEqual((a.delay_min_lag,a.delay_max_lag),(4,12));self.assertGreater(a.damping,0)
        self.assertNotIn('head_imu',cfg.env.observations['actor'].terms)


if __name__=='__main__':unittest.main()
