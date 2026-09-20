"""PPO with fixed standing/stopping quotas and executed recovery supervision."""
from dataclasses import dataclass
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.algorithms.ppo import PPO
from mjlab_microduck.retention_ppo import RetentionPPO,RetentionPpoCfg


@dataclass
class CorrectivePpoCfg(RetentionPpoCfg):
    corrective_arm: str='control'
    success_dataset: str=''
    stop_dataset: str=''
    recovery_dataset: str=''
    class_name: str='mjlab_microduck.corrective_ppo:CorrectivePPO'


class CorrectivePPO(RetentionPPO):
    def __init__(self,*args,corrective_arm,success_dataset,stop_dataset,recovery_dataset,**kwargs):
        super().__init__(*args,**kwargs)
        if corrective_arm not in ('control','corrective'):raise ValueError(corrective_arm)
        if self.anchor_batch_size!=1536:raise ValueError('Fixed supervision quota requires 1536')
        self.corrective_arm=corrective_arm
        standing=self.anchor_obs[:,63:66].abs().max(dim=1).values<1e-6
        self.banks={'old_stand':(self.anchor_obs[standing],self.anchor_actions[standing]),
            'old_walk':(self.anchor_obs[~standing],self.anchor_actions[~standing])}
        for key,path in (('success',success_dataset),('stop',stop_dataset),('recovery',recovery_dataset)):
            with np.load(path) as data:
                obs=data['obs'].copy();actions=data['actions'].copy()
                if obs.ndim!=2 or obs.shape[1]!=81 or actions.shape!=(len(obs),19):raise ValueError(f'{key}: layout')
                if not np.isfinite(obs).all() or not np.isfinite(actions).all():raise ValueError(f'{key}: nonfinite')
                if key=='stop':
                    # Preserve the existing .03-per-tick braking ramp and body
                    # heading PI. A stop request does not instantly zero actor inputs.
                    if not np.allclose(obs[:,64],0) or np.any(obs[:,63]<0) or np.any(obs[:,63]>.32001):
                        raise ValueError('Unexpected braking command')
                    for trial in np.unique(data['trial_id']):
                        rows=obs[data['trial_id']==trial]
                        expected=np.maximum(.32-.03*np.arange(len(rows)),0)
                        if not np.allclose(rows[:,63],expected,atol=1e-6):raise ValueError('Broken stop ramp')
                        if not np.allclose(rows[50:,63:66],0):raise ValueError('Nonzero settled stop command')
                        if np.max(np.abs(np.diff(rows[:,65])))>.03001:raise ValueError('Broken heading-command slew')
                elif not np.allclose(obs[:,63],.2):raise ValueError('Step teacher command must be .2')
                tensors=(torch.as_tensor(obs,device=self.device),torch.as_tensor(actions,device=self.device))
                self.banks[key]=tensors
                if key=='recovery':
                    frontier=torch.as_tensor(data['frontier'].astype(bool),device=self.device)
                    self.banks['frontier']=tuple(x[frontier] for x in tensors)
                    self.banks['continuation']=tuple(x[~frontier] for x in tensors)
        if any(len(obs)==0 for obs,_ in self.banks.values()):raise ValueError('Empty sampling stratum')

    def quota(self):
        shared=[('old_stand',128),('old_walk',384),('stop',256)]
        return shared+([('success',768)] if self.corrective_arm=='control' else [('success',256),('frontier',256),('continuation',256)])

    def supervised_batch(self):
        observations=[];actions=[];groups=[]
        for key,count in self.quota():
            obs,target=self.banks[key];ids=torch.randint(len(obs),(count,),device=self.device)
            observations.append(obs[ids]);actions.append(target[ids]);groups.append((key,count))
        return torch.cat(observations),torch.cat(actions),groups

    def update(self):
        result=PPO.update(self);losses={}
        for _ in range(self.anchor_steps):
            obs,target,groups=self.supervised_batch()
            prediction=self.actor(TensorDict({'actor':obs},batch_size=[len(obs)]))
            errors=(prediction-target).square().mean(dim=1)
            self.optimizer.zero_grad(set_to_none=True)
            (errors.mean()*self.anchor_weight).backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm)
            self.optimizer.step();offset=0
            for key,count in groups:
                losses[key]=losses.get(key,0.)+float(errors[offset:offset+count].mean().detach())
                offset+=count
        result.update({f'imitation_{key}':value/self.anchor_steps for key,value in losses.items()})
        return result


class CorrectivePpoV2Cfg(CorrectivePpoCfg):
    class_name: str = 'mjlab_microduck.corrective_ppo:CorrectivePPOv2'


class CorrectivePPOv2(CorrectivePPO):
    """v2 配额(2026-09-17):停稳回放 256→384。

    依据:最新纠正组模型 d5 停稳 9/9 残速 ~35mm/s 全败(门 10mm/s),而
    probe+collect_v2 从该模型实际失败初态新采 1500 帧停稳纠正(旧库 1050)。
    RESULTS 下一步要求"停稳回放配额给足"。总锚定批保持 1536:
    128 stand + 384 walk + 384 stop + 256 success + 192 frontier + 192 continuation。
    """

    def quota(self):
        return ([('old_stand', 128), ('old_walk', 384), ('stop', 384),
                 ('success', 256), ('frontier', 192), ('continuation', 192)])
