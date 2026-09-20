"""Feet-first jump objective: measured whole-sole clearance, deep preload and airborne tuck."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import ConvexHull
from mjlab.managers import EventTermCfg,MetricsTermCfg,RewardTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.sensor import ContactSensorCfg,ContactMatch
from jump_refine_cfg import (build_config as previous,CurveImuAction,CurveImuActionCfg,
    JumpCommand,JumpCommandCfg,reset_refined_jump,joint_excess,whole_com,STAND_Z)
from run_jump_cfg import contacts

OUT=Path(__file__).parent/'20260914_foot_flight'


class FootProgress:
    """Only continuous, simultaneously unsupported whole-foot clearance earns jump credit."""
    def __init__(self,n,device):
        for name in ['run2','run5','event_peak','height_frontier','duration_frontier','height_gain','duration_gain','goal_bonus']:
            setattr(self,name,torch.zeros(n,device=device))
        for name in ['grounded','qualified','real_flight','air']:
            setattr(self,name,torch.zeros(n,dtype=torch.bool,device=device))
        self.last_substep=-1
    def reset(self,ids):
        for name,value in vars(self).items():
            if torch.is_tensor(value):value[ids]=0
    def begin_control(self):
        for name in ['height_gain','duration_gain','goal_bonus']:getattr(self,name).zero_()
    def update(self,substep,dt,gaps,support,safe,window):
        if substep==self.last_substep:return
        self.last_substep=substep
        self.grounded |= support>.1
        gap=gaps.amin(-1)
        valid=(support<.05)&safe&window&self.grounded
        self.air=valid&(gap>.002)
        air5=valid&(gap>.005)
        self.run2=torch.where(self.air,self.run2+dt,0.)
        self.run5=torch.where(air5,self.run5+dt,0.)
        # Keep peak and duration from the same continuous >5 mm interval.
        self.event_peak=torch.where(air5,torch.maximum(self.event_peak,gap),0.)
        height=torch.where(self.air,(gap-.002).clamp(0,.018),0.)
        high=torch.maximum(self.height_frontier,height)
        self.height_gain+=high-self.height_frontier;self.height_frontier=high
        high=torch.maximum(self.duration_frontier,self.run5.clamp(max=.12))
        self.duration_gain+=high-self.duration_frontier;self.duration_frontier=high
        self.real_flight |= self.run2>=.04-1e-7
        goal=(self.run5>=.06-1e-7)&(self.event_peak>=.010)
        self.goal_bonus+=(goal&~self.qualified).float();self.qualified |= goal


class FootGeometry:
    def __init__(self,model,device):
        self.gids=[model.geom('robot/'+s+'_foot_collision').id for s in ['left','right']]
        self.vertices=[]
        for gid in self.gids:
            mesh=model.geom_dataid[gid];start=model.mesh_vertadr[mesh]
            v=model.mesh_vert[start:start+model.mesh_vertnum[mesh]].copy()
            # The minimum of a linear functional is unchanged by taking the
            # exact hull vertices. No sole-center, toe or bounding-box shortcut.
            v=v[ConvexHull(v).vertices]
            self.vertices.append(torch.as_tensor(v,device=device))
    def clearance(self,data):
        h=[]
        for gid,vertices in zip(self.gids,self.vertices):
            R=data.geom_xmat[:,gid].reshape(-1,3,3)
            h.append((R[:,2,:]@vertices.T).amin(-1)+data.geom_xpos[:,gid,2])
        return torch.stack(h,-1)


@dataclass(kw_only=True)
class FootActionCfg(CurveImuActionCfg):
    def build(self,env):return FootAction(self,env)


class FootAction(CurveImuAction):
    def process_actions(self,actions):
        self.env._foot_progress.begin_control()
        return super().process_actions(actions)


class FoldCommand(JumpCommand):
    def _resample_command(self,ids):
        super()._resample_command(ids)
        fraction=min(1.,max(0.,(self._env.common_step_counter-self.cfg.source_step)/(240*24)))
        depth=.03+(self.cfg.deepest_m-.03)*fraction
        self.depth[ids]=torch.empty(len(ids),device=self.device).uniform_(depth-.002,depth)
        self.preparation[ids]=torch.empty(len(ids),device=self.device).uniform_(1.6,2.0)
        self.extension[ids]=torch.empty(len(ids),device=self.device).uniform_(.010,.020)


@dataclass(kw_only=True)
class FoldCommandCfg(JumpCommandCfg):
    deepest_m:float=.045
    source_step:int=0
    def build(self,env):return FoldCommand(self,env)


@requires_model_fields('actuator_forcerange')
def reset_feet(env,env_ids):
    reset_refined_jump(env,env_ids)
    if not hasattr(env,'_foot_progress'):
        env._foot_progress=FootProgress(env.num_envs,env.device)
        env._foot_geometry=FootGeometry(env.sim.mj_model,env.device)
    env._foot_progress.reset(slice(None) if env_ids is None else env_ids)


def measure_substep(env):
    command=env.command_manager.get_term('body_pose')
    gap=env._foot_geometry.clearance(env.sim.data)
    force=env.scene['whole_ground_force'].data.force.reshape(env.num_envs,-1,3)
    support=force[...,2].abs().sum(-1)
    safe=(env.scene['robot'].data.projected_gravity_b[:,2]<-.8660254)&(joint_excess(env)<=.005)
    window=(command.elapsed>=command.preparation)&(command.elapsed<command.preparation+1.2)
    env._foot_progress.update(env._sim_step_counter,env.physics_dt,gap,support,safe,window)
    return env._foot_progress.air.float()


def foot_height(env):return env._foot_progress.height_gain/.018/env.step_dt
def foot_duration(env):return env._foot_progress.duration_gain/.12/env.step_dt
def foot_goal(env):return env._foot_progress.goal_bonus/env.step_dt
def goal_metric(env):return env._foot_progress.qualified.float()


def pose_tracking(env,target,phase):
    robot=env.scene['robot'];names=robot.joint_names
    ids=[i for i,n in enumerate(names) if n.startswith(('left_','right_'))]
    reference=torch.tensor(target,device=env.device)
    c=env.command_manager.get_term('body_pose')
    if phase=='crouch':
        depths=torch.tensor([.03,.035,.04,.045,.05],device=env.device)
        index=torch.searchsorted(depths,c.depth).clamp(1,4)
        fraction=((c.depth-depths[index-1])/(depths[index]-depths[index-1])).clamp(0,1)
        reference=reference[index-1]+(reference[index]-reference[index-1])*fraction[:,None]
    else:reference=reference.expand(env.num_envs,-1)
    # Reference uses the actuator/contract order, identical for these 19 joints.
    error=(robot.data.joint_pos[:,ids]-reference[:,ids]).square().mean(-1)
    active=(c.elapsed<c.preparation) if phase=='crouch' else env._foot_progress.air
    return torch.exp(-error/.09)*active


def settle_after_flight(env):
    c=env.command_manager.get_term('body_pose');r=env.scene['robot'];_,v=whole_com(env)
    z=env.scene['body_clearance'].data.heights[:,0]
    good=env._foot_progress.real_flight&contacts(env,'feet_ground_contact').all(-1)&(c.elapsed>=c.preparation+c.extension_duration)
    tilt=r.data.projected_gravity_b[:,:2].square().sum(-1)
    return good*torch.exp(-((z-STAND_Z)/.02)**2-v.square().sum(-1)/.0009-tilt/.04)


def build_config(envs=64,seed=61):
    task,cfg=previous(envs,seed)
    selection=json.loads((OUT/'selection.json').read_text())
    target_data=json.loads((OUT/'deep_targets.json').read_text())
    targets={round(r['depth_m'],6):r['target'] for r in target_data['rows']}
    depth=selection['training_depth_m']
    cfg.env.actions['joint_pos']=FootActionCfg(**vars(cfg.env.actions['joint_pos']))
    cfg.env.commands['twist']=FoldCommandCfg(skill='jump',width=3,resampling_time_range=(100.,100.),deepest_m=depth,source_step=selection['source_common_step_counter'])
    cfg.env.commands['body_pose']=FoldCommandCfg(skill='jump',width=6,resampling_time_range=(100.,100.),deepest_m=depth,source_step=selection['source_common_step_counter'])
    cfg.env.scene.sensors=(*cfg.env.scene.sensors,ContactSensorCfg(name='whole_ground_force',
        primary=ContactMatch(mode='subtree',pattern='trunk_base',entity='robot'),
        secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found','force'),reduce='netforce',num_slots=1))
    cfg.env.events.pop('reset_refined_jump');cfg.env.events['reset_feet']=EventTermCfg(func=reset_feet,mode='reset')
    cfg.env.rewards.pop('hop_launch');cfg.env.rewards.pop('hop_flight')
    cfg.env.rewards['foot_clearance_progress']=RewardTermCfg(func=foot_height,weight=16.)
    cfg.env.rewards['foot_duration_progress']=RewardTermCfg(func=foot_duration,weight=16.)
    cfg.env.rewards['foot_goal_once']=RewardTermCfg(func=foot_goal,weight=8.)
    cfg.env.rewards['deep_pose_reference']=RewardTermCfg(func=pose_tracking,weight=2.,params={'target':[targets[d] for d in [.03,.035,.04,.045,.05]],'phase':'crouch'})
    cfg.env.rewards['air_tuck_reference']=RewardTermCfg(func=pose_tracking,weight=4.,params={'target':targets[.05],'phase':'air'})
    cfg.env.rewards['hop_land']=RewardTermCfg(func=settle_after_flight,weight=8.)
    cfg.env.metrics['true_foot_air_fraction']=MetricsTermCfg(func=measure_substep,per_substep=True)
    cfg.env.metrics['feet_goal_achieved']=MetricsTermCfg(func=goal_metric,reduce='last')
    cfg.agent.experiment_name='microdinosaur_foot_flight'
    return task,cfg
