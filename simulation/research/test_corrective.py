import unittest
import json
from dataclasses import asdict
import numpy as np
import torch
from mjlab_microduck.corrective_ppo import CorrectivePPO
from corrective_cfg import build_config


class CorrectiveTests(unittest.TestCase):
    def test_sampling_preserves_standing_stopping_and_total_budget(self):
        learner=object.__new__(CorrectivePPO);learner.device='cpu'
        keys=('old_stand','old_walk','stop','success','frontier','continuation')
        learner.banks={key:(torch.full((5,81),float(i)),torch.full((5,19),float(i+10))) for i,key in enumerate(keys)}
        for arm in ('control','corrective'):
            learner.corrective_arm=arm;obs,actions,groups=learner.supervised_batch()
            self.assertEqual(obs.shape,(1536,81));self.assertEqual(actions.shape,(1536,19))
            self.assertTrue(torch.equal(obs[:,0]+10,actions[:,0]))
            self.assertEqual(int((obs[:,0]==0).sum()),128)
            self.assertEqual(int((obs[:,0]==1).sum()),384)
            self.assertEqual(int((obs[:,0]==2).sum()),256)
            self.assertEqual(int((obs[:,0]==4).sum()),256 if arm=='corrective' else 0)
            self.assertEqual(int((obs[:,0]==5).sum()),256 if arm=='corrective' else 0)

    def test_environment_and_reward_configuration_are_paired(self):
        def default(x):
            if callable(x):return x.__module__+'.'+x.__qualname__
            if isinstance(x,np.ndarray):return x.tolist()
            return str(x)
        a,b=(build_config(arm,8,47)[1] for arm in ('control','corrective'))
        self.assertEqual(json.dumps(asdict(a.env),default=default,sort_keys=True),json.dumps(asdict(b.env),default=default,sort_keys=True))
        for cfg in (a,b):
            self.assertEqual(cfg.agent.algorithm.anchor_batch_size,1536)
            self.assertEqual(cfg.agent.algorithm.anchor_steps,4)
            self.assertEqual(cfg.env.actions['joint_pos'].owned_head_indices,(1,))


if __name__=='__main__':unittest.main()

