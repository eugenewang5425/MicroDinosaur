import unittest
from types import SimpleNamespace as NS
import numpy as np
import torch
from demonstration_expert import CalibratedExpert
from mjlab_microduck.demonstration_ppo import DemonstrationPPO
from demonstration_cfg import build_config


class DemonstrationTests(unittest.TestCase):
    def test_fault_keeps_randomized_fields_and_still_aborts(self):
        import json,tempfile,mujoco
        from pathlib import Path
        from mjlab.utils.nan_guard import NanGuardCfg
        from demonstration_nan_guard import DemonstrationNanGuard
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>')
        data=NS(qpos=torch.zeros(2,model.nq),qvel=torch.zeros(2,model.nv),qacc=torch.zeros(2,model.nv),
            qacc_warmstart=torch.zeros(2,model.nv),sensordata=torch.zeros(2,3),ctrl=torch.zeros(2,0))
        data.qacc[1,0]=float('nan')
        with tempfile.TemporaryDirectory() as folder:
            guard=DemonstrationNanGuard(NanGuardCfg(enabled=True),2,model,output_dir=folder)
            guard.simulation=NS(expanded_fields={'body_mass'},model=NS(body_mass=torch.tensor([[1.,2.],[3.,4.]])))
            with self.assertRaises(FloatingPointError):guard.check_and_dump(data)
            r=json.loads(next(Path(folder).glob('*.json')).read_text());self.assertEqual(r['dumped_env_ids'],[1])
            with np.load(Path(folder)/r['randomized_context_file']) as z:np.testing.assert_array_equal(z['model_body_mass'],[[3.,4.]])

    def test_expert_keeps_student_command_and_history(self):
        def inference(outputs,inputs):return [inputs['obs'][:,44:63].copy()]
        wrapped=CalibratedExpert(NS(run=inference),.55)
        obs=np.zeros((1,81),np.float32);obs[0,63]=.2;obs[0,44:63]=np.arange(19)
        before=obs.copy();result=wrapped.run(None,{'obs':obs})
        np.testing.assert_array_equal(obs,before)
        self.assertAlmostEqual(float(wrapped.obs[0][63]),.2)
        self.assertAlmostEqual(float(wrapped.expert_obs[0][63]),.55)
        np.testing.assert_array_equal(wrapped.obs[0][44:63],result[0][0])
        changed=np.where(wrapped.obs[0]!=wrapped.expert_obs[0])[0]
        np.testing.assert_array_equal(changed,[63])

    def test_offline_batch_cannot_query_onpolicy_teacher(self):
        # Sentinel banks make leakage or changed supervision counts visible.
        learner=object.__new__(DemonstrationPPO);learner.device='cpu';learner.anchor_batch_size=1024
        learner.anchor_obs=torch.ones(17,81);learner.anchor_actions=torch.ones(17,19)*3
        learner.demo_obs=None;learner.demo_actions=None
        obs,target=learner.supervised_batch()
        self.assertEqual(obs.shape,(1024,81));self.assertTrue((target==3).all())
        learner.demo_obs=torch.ones(13,81)*2;learner.demo_actions=torch.ones(13,19)*4
        obs,target=learner.supervised_batch()
        self.assertEqual(int((obs[:,0]==1).sum()),512);self.assertEqual(int((obs[:,0]==2).sum()),512)
        self.assertTrue((target[:512]==3).all());self.assertTrue((target[512:]==4).all())

    def test_configs_change_only_algorithm_and_run_identity(self):
        import json
        from dataclasses import asdict
        def signature(env):
            def convert(v):
                if callable(v):return v.__module__+'.'+v.__qualname__
                if isinstance(v,np.ndarray):return v.tolist()
                return str(v)
            return json.dumps(asdict(env),default=convert,sort_keys=True)
        configs=[build_config(arm,8,43)[1] for arm in ('legacy','archive','demonstration')]
        self.assertEqual(signature(configs[0].env),signature(configs[1].env))
        self.assertEqual(signature(configs[0].env),signature(configs[2].env))
        for cfg in configs:
            self.assertEqual(cfg.agent.algorithm.anchor_steps,2)
            self.assertEqual(cfg.agent.algorithm.anchor_batch_size,1024)
            self.assertEqual(cfg.env.actions['joint_pos'].owned_head_indices,(1,))


if __name__=='__main__':unittest.main()
