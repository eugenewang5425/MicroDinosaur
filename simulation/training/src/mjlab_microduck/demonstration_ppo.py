"""Optional offline-only imitation, isolated from the historical retention PPO."""
from dataclasses import dataclass
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.algorithms.ppo import PPO
from mjlab_microduck.retention_ppo import RetentionPPO,RetentionPpoCfg


@dataclass
class DemonstrationPpoCfg(RetentionPpoCfg):
    demonstration_dataset: str=''
    class_name: str='mjlab_microduck.demonstration_ppo:DemonstrationPPO'


class DemonstrationPPO(RetentionPPO):
    def __init__(self,*args,demonstration_dataset='',**kwargs):
        super().__init__(*args,**kwargs)
        self.demo_obs=None;self.demo_actions=None
        if demonstration_dataset:
            with np.load(demonstration_dataset) as z:
                obs=z['obs'].copy();actions=z['actions'].copy()
            if obs.ndim!=2 or obs.shape[1]!=81 or actions.shape!=(len(obs),19):
                raise ValueError('Demonstrations must retain the 81-to-19 contract')
            if not np.isfinite(obs).all() or not np.isfinite(actions).all():
                raise ValueError('Nonfinite demonstration')
            if not np.allclose(obs[:,63],.2) or not np.all(obs[:,72]==0):
                raise ValueError('Expected zero-crouch .2m/s step demonstrations')
            self.demo_obs=torch.as_tensor(obs,device=self.device)
            self.demo_actions=torch.as_tensor(actions,device=self.device)

    def supervised_batch(self):
        count=self.anchor_batch_size if self.demo_obs is None else self.anchor_batch_size//2
        idx=torch.randint(len(self.anchor_obs),(count,),device=self.device)
        obs=self.anchor_obs[idx];target=self.anchor_actions[idx]
        if self.demo_obs is not None:
            idx=torch.randint(len(self.demo_obs),(self.anchor_batch_size-count,),device=self.device)
            obs=torch.cat((obs,self.demo_obs[idx]));target=torch.cat((target,self.demo_actions[idx]))
        return obs,target

    def update(self):
        # PPO rollout data and rewards remain intact. There is no old-teacher
        # target for unfamiliar on-policy terrain or transition states.
        result=PPO.update(self);total=0.
        for _ in range(self.anchor_steps):
            obs,target=self.supervised_batch()
            prediction=self.actor(TensorDict({'actor':obs},batch_size=[len(obs)]))
            loss=(prediction-target).square().mean()*self.anchor_weight
            self.optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm)
            self.optimizer.step();total+=float(loss.detach())
        result['gait_anchor']=total/max(self.anchor_steps,1)
        result['anchor_eligible_fraction']=0.
        return result
