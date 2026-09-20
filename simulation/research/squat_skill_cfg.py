"""Independent v07 squat expert with a reward-only feasible pose reference."""
from dataclasses import dataclass
import json
import numpy as np
import torch
from mjlab.managers import EventTermCfg,RewardTermCfg,TerminationTermCfg
from mjlab_microduck.tasks import mdp
from jump_refine_cfg import CurveImuAction,CurveImuActionCfg,limit_failure
from squat_plant import scene_config,OUT,XML


class SquatAction(CurveImuAction):
    def process_actions(self, actions):
        bounded=actions.clone()
        # Command operating envelope; physical joint stops and motor model stay
        # unchanged. The CPU evaluator applies the identical angle bounds.
        ankles=[self._target_names.index(n) for n in ('left_ankle','right_ankle')]
        target=self._offset+self._scale*bounded
        target[:,ankles]=target[:,ankles].clamp(-np.deg2rad(51),np.deg2rad(51))
        target[:,self._target_names.index('jaw_hinge')]=.04
        bounded=(target-self._offset)/self._scale
        super().process_actions(bounded)
        self._raw_actions[:]=actions


@dataclass(kw_only=True)
class SquatActionCfg(CurveImuActionCfg):
    def build(self,env):return SquatAction(self,env)


class SquatResidualAction(SquatAction):
    """Validated nominal squat plus a bounded learned balance correction."""
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.bank['raw_action'].zero_()
        contract=json.loads((OUT/'plant/contract.json').read_text())
        self.nominal_home=torch.tensor(contract['action_offset'][0],device=env.device)
        self.nominal_scale=torch.tensor(contract['action_scale'],device=env.device)
        pose=next(r for r in json.loads((OUT/'pose_references.json').read_text()) if r['depth_mm']==25)
        self.delta=torch.tensor(pose['target'],device=env.device)-self.nominal_home
        self.legs=[i for i,n in enumerate(self._target_names) if n.startswith(('left_','right_'))]

    def process_actions(self,actions):
        fraction=(-self.env.command_manager.get_command('body_pose')[:,2]/.025).clamp(0,1)
        goal=self.bank['applied'][self.bank_index].clone()
        goal[:,self.legs]+=fraction[:,None]*self.delta[self.legs]
        goal[:,self.legs]+=.015*torch.tanh(actions[:,self.legs])
        # Nominal conversion deliberately preserves inherited action-scale DR.
        nominal=(goal-self.nominal_home)/self.nominal_scale
        super().process_actions(nominal)
        self._raw_actions[:]=actions


@dataclass(kw_only=True)
class SquatResidualActionCfg(SquatActionCfg):
    def build(self,env):return SquatResidualAction(self,env)


def build_config(envs=64,seed=914,residual=False):
    task,cfg=scene_config(envs,seed)
    args=vars(cfg.env.actions['joint_pos']).copy()
    args['bank_path']=str(OUT/'calibrated_reset_bank.json')
    cfg.env.actions['joint_pos']=(SquatResidualActionCfg if residual else SquatActionCfg)(**args)
    cfg.env.commands['twist']=mdp.IndependentSquatCommandCfg(width=3,resampling_time_range=(100.,100.))
    cfg.env.commands['body_pose']=mdp.IndependentSquatCommandCfg(width=6,resampling_time_range=(100.,100.))
    cfg.env.commands['jaw_pose'].ranges=((.04,.04),)
    cfg.env.episode_length_s=13.
    cfg.env.events.pop('reset_hop')
    cfg.env.events['reset_squat_motor']=EventTermCfg(func=mdp.reset_squat_motor_envelope,mode='reset')
    cfg.env.curriculum={}
    keep=('body_ang_vel','dof_pos_limits','action_rate_l2','foot_slip','head_world_gaze',
          'arm_pose_tracking','tail_pose_tracking','head_pose_tracking','translation')
    cfg.env.rewards={k:v for k,v in cfg.env.rewards.items() if k in keep}
    cfg.env.rewards['head_world_gaze'].weight=.5
    cfg.env.rewards['action_rate_l2'].weight=-.05
    names=json.loads((OUT/'plant/contract.json').read_text())['action_names']
    rows=json.loads((OUT/'pose_references.json').read_text())[:6]
    cfg.env.rewards['squat_tracking']=RewardTermCfg(func=mdp.independent_squat_tracking,weight=5.,
        params={'reference_names':names,'reference_table':[r['target'] for r in rows]})
    cfg.env.terminations['joint_limit_failure']=TerminationTermCfg(func=limit_failure,time_out=False)
    cfg.env.terminations['squat_envelope']=TerminationTermCfg(func=mdp.independent_squat_envelope,time_out=False)
    cfg.env.terminations['fell_over'].params['limit_angle']=np.deg2rad(25)
    cfg.agent.experiment_name='microdinosaur_squat_specialist'
    return task,cfg
