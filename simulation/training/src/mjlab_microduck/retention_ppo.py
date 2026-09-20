"""PPO followed by bounded actor distillation on zero-height gait commands.

PPO samples are untouched. Extra supervised updates use a frozen normalized
v7 teacher on current on-policy states and its recorded successful gait states.
The teacher never supervises a nonzero crouch command. No ERPO components.
"""
from dataclasses import dataclass
import copy
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.algorithms.ppo import PPO
from mjlab.rl.config import RslRlPpoAlgorithmCfg


@dataclass
class RetentionPpoCfg(RslRlPpoAlgorithmCfg):
    symmetry_cfg: dict | None = None
    teacher_checkpoint: str = ''
    anchor_dataset: str = ''
    anchor_steps: int = 2
    anchor_batch_size: int = 1024
    anchor_weight: float = 1.
    class_name: str = 'mjlab_microduck.retention_ppo:RetentionPPO'


class RetentionPPO(PPO):
    def __init__(self, *args, teacher_checkpoint, anchor_dataset, anchor_steps=2,
                 anchor_batch_size=1024, anchor_weight=1., **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.is_multi_gpu and not self.actor.is_recurrent
        self.teacher = copy.deepcopy(self.actor)
        state = torch.load(teacher_checkpoint,map_location=self.device,weights_only=False)
        self.teacher.load_state_dict(state['actor_state_dict'])
        self.teacher.eval().requires_grad_(False)
        data = np.load(anchor_dataset)
        self.anchor_obs = torch.tensor(data['obs'],device=self.device)
        self.anchor_actions = torch.tensor(data['actions'],device=self.device)
        assert self.anchor_obs.shape[1] == 81 and self.anchor_actions.shape[1] == 19
        assert torch.all(self.anchor_obs[:,72] == 0)
        self.anchor_steps,self.anchor_batch_size,self.anchor_weight = anchor_steps,anchor_batch_size,anchor_weight
        # Independent check against recorded official ONNX outputs.
        # Validate on CPU FP32 without changing mjlab's GPU precision settings.
        # Fast CUDA matmul differs by ~2.4e-4 rad on this bank (measured).
        check_teacher=copy.deepcopy(self.teacher).cpu().eval()
        with torch.no_grad():
            y=check_teacher(TensorDict({'actor':self.anchor_obs[:64].cpu()},batch_size=[64]))
            torch.testing.assert_close(y,self.anchor_actions[:64].cpu(),atol=2e-5,rtol=2e-5)
        del check_teacher

    def update(self):
        current = self.storage.observations['actor'].reshape(-1,81)
        eligible = current[current[:,72].abs() < 1e-6].detach().clone()
        result = super().update()
        loss_sum = 0.
        for _ in range(self.anchor_steps):
            half = self.anchor_batch_size//2
            idx=torch.randint(len(self.anchor_obs),(half,),device=self.device)
            obs=self.anchor_obs[idx]; target=self.anchor_actions[idx]
            if len(eligible):
                on_policy=eligible[torch.randint(len(eligible),(half,),device=self.device)]
                with torch.no_grad(): wanted=self.teacher(TensorDict({'actor':on_policy},batch_size=[half]))
                obs=torch.cat((obs,on_policy));target=torch.cat((target,wanted))
            td=TensorDict({'actor':obs},batch_size=[len(obs)])
            predicted=self.actor(td)
            loss=(predicted-target).square().mean()*self.anchor_weight
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm)
            self.optimizer.step()
            loss_sum += float(loss.detach())
        result['gait_anchor'] = loss_sum/max(self.anchor_steps,1)
        result['anchor_eligible_fraction'] = float(len(eligible)/len(current))
        return result
