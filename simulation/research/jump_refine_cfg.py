"""Jump continuation: whole-body launch, limit margin, landing, per-substep motor envelope."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from mjlab.managers import EventTermCfg,RewardTermCfg,TerminationTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab_microduck.imu_owned_head import ImuOwnedAction,ImuOwnedActionCfg
from run_jump_cfg import build_config as base_config,MotionCommand,MotionCommandCfg,contacts,STAND_Z

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_jump_refine'
SOURCE=Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_jump_specialist/20260914_train_512x401/model_16650.pt')


def torque_bounds_torch(velocity,voltage,cap):
    # Same provisional four-quadrant envelope as HardwareSim. No qvel clamp;
    # braking remains available when motoring torque reaches zero.
    stall=.6*voltage/12.6
    speed=16.5*voltage/12.
    hi=torch.minimum((stall*(1-velocity/speed)).clamp(min=0),cap)
    lo=-torch.minimum((stall*(1+velocity/speed)).clamp(min=0),cap)
    return lo,hi


@dataclass(kw_only=True)
class CurveImuActionCfg(ImuOwnedActionCfg):
    def build(self,env):return CurveImuAction(self,env)


class CurveImuAction(ImuOwnedAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        m=env.sim.mj_model
        self.ctrl_ids=torch.tensor([int(np.flatnonzero(m.actuator_trnid[:,0]==m.joint('robot/'+n).id)[0])
                                   for n in self._target_names],device=env.device)
        self.voltage=torch.full((env.num_envs,1),12.6,device=env.device)
        self.cap=torch.full((env.num_envs,len(self._target_ids)),.6,device=env.device)
        self.limit_peak=torch.zeros(env.num_envs,device=env.device)

    def process_actions(self,actions):
        self.limit_peak.zero_()
        return super().process_actions(actions)

    def apply_actions(self):
        velocity=self._entity.data.joint_vel[:,self._target_ids]
        lo,hi=torque_bounds_torch(velocity,self.voltage,self.cap)
        self.env.sim.model.actuator_forcerange[:,self.ctrl_ids,0]=lo
        self.env.sim.model.actuator_forcerange[:,self.ctrl_ids,1]=hi
        self.limit_peak=torch.maximum(self.limit_peak,joint_excess(self.env))
        return super().apply_actions()


class JumpCommand(MotionCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.preparation=torch.full_like(self.elapsed,1.5)
        self.extension_duration=torch.full_like(self.elapsed,.7)
        self.depth=torch.full_like(self.elapsed,.03)
        self.extension=torch.full_like(self.elapsed,.01)

    def _resample_command(self,ids):
        super()._resample_command(ids)
        for name,(lo,hi) in dict(preparation=(1.2,1.8),extension_duration=(.4,.8),
                                depth=(.025,.03),extension=(.008,.015)).items():
            getattr(self,name)[ids]=torch.empty(len(ids),device=self.device).uniform_(lo,hi)

    def _update_command(self):
        self.elapsed+=self._env.step_dt
        if self.cfg.width==6:
            self._command[:,2]=torch.where(self.elapsed<self.preparation,-self.depth,
                torch.where(self.elapsed<self.preparation+self.extension_duration,self.extension,0.))


@dataclass(kw_only=True)
class JumpCommandCfg(MotionCommandCfg):
    def build(self,env):return JumpCommand(self,env)


def whole_com(env):
    robot=env.scene['robot'];ids=robot.indexing.body_ids
    mass=env.sim.model.body_mass[:,ids]
    velocity=robot.data.body_com_lin_vel_w
    pos=env.sim.data.xipos[:,ids]
    total=mass.sum(-1,keepdim=True)
    return (pos*mass[...,None]).sum(1)/total,(velocity*mass[...,None]).sum(1)/total


def joint_excess(env):
    r=env.scene['robot'];q=r.data.joint_pos
    limit=r.data.joint_pos_limits
    return torch.maximum(limit[...,0]-q,q-limit[...,1]).clamp(min=0).amax(-1)


class JumpProgress:
    def __init__(self,n,device):
        self.velocity_max=torch.zeros(n,device=device)
        self.flight_max=torch.zeros(n,device=device)
        self.flight=torch.zeros(n,device=device)
        self.launched=torch.zeros(n,device=device,dtype=torch.bool)
        self.dv=torch.zeros(n,device=device);self.df=torch.zeros_like(self.dv)
        self.last_step=-1

    def reset(self,ids):
        for key in ('velocity_max','flight_max','flight','launched','dv','df'):getattr(self,key)[ids]=0

    def update(self,step,dt,vz,supported,upright,window,within_limits):
        if step==self.last_step:return
        self.last_step=step
        eligible=upright & window & within_limits
        high=torch.maximum(self.velocity_max,torch.where(eligible,vz.clamp(0,.7),0.))
        self.dv=high-self.velocity_max;self.velocity_max=high
        self.launched |= eligible & (vz>.1)
        self.flight=torch.where(~supported & upright & within_limits & self.launched,self.flight+dt,0.)
        high=torch.maximum(self.flight_max,self.flight.clamp(max=.16))
        self.df=high-self.flight_max;self.flight_max=high


def progress(env):
    state=env._jump_refine_state
    command=env.command_manager.get_term('body_pose');t=command.elapsed
    _,v=whole_com(env)
    state.update(env.common_step_counter,env.step_dt,v[:,2],contacts(env,'robot_ground').any(-1),
        env.scene['robot'].data.projected_gravity_b[:,2]<-.866,
        (t>=command.preparation)&(t<command.preparation+1.),joint_excess(env)<=.005)
    return state


@requires_model_fields('actuator_forcerange')
def reset_refined_jump(env,env_ids):
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    action=env.action_manager.get_term('joint_pos')
    # This runs after nominal-relative effort randomization, before action reset.
    # Save sampled caps once: never compound last substep's speed-reduced limit.
    action.cap[ids]=env.sim.model.actuator_forcerange[ids[:,None],action.ctrl_ids,1]
    action.voltage[ids]=torch.empty((len(ids),1),device=env.device).uniform_(11.1,12.6)
    action.limit_peak[ids]=0
    if not hasattr(env,'_jump_refine_state'):env._jump_refine_state=JumpProgress(env.num_envs,env.device)
    env._jump_refine_state.reset(ids)


def prepare(env):
    command=env.command_manager.get_term('body_pose')
    z=env.scene['body_clearance'].data.heights[:,0]
    return torch.exp(-((z-(STAND_Z-command.depth))/.015)**2)*(command.elapsed<command.preparation)


def launch(env):return progress(env).dv/.7/env.step_dt
def flight(env):return progress(env).df/.16/env.step_dt


def landing(env):
    state=progress(env);command=env.command_manager.get_term('body_pose');r=env.scene['robot']
    z=env.scene['body_clearance'].data.heights[:,0];_,v=whole_com(env)
    good=(state.flight_max>=.04)&contacts(env,'feet_ground_contact').all(-1)&(command.elapsed>=command.preparation+command.extension_duration)
    tilt=r.data.projected_gravity_b[:,:2].square().sum(-1)
    return good*torch.exp(-((z-STAND_Z)/.02)**2-v.square().sum(-1)/.0009-tilt/.04)


def limit_margin(env):
    r=env.scene['robot'];q=r.data.joint_pos;limit=r.data.joint_pos_limits
    distance=torch.minimum(q-limit[...,0],limit[...,1]-q)
    return ((.10-distance).clamp(min=0)/.10).square().sum(-1)


def limit_failure(env):
    action=env.action_manager.get_term('joint_pos')
    return torch.maximum(action.limit_peak,joint_excess(env))>.02


def touchdown_speed(env):
    _,v=whole_com(env)
    return contacts(env,'feet_ground_contact').any(-1)*(-v[:,2]-.4).clamp(min=0).square()


def build_config(envs=64,seed=61):
    task,cfg=base_config('jump',envs,seed)
    cfg.env.actions['joint_pos']=CurveImuActionCfg(**vars(cfg.env.actions['joint_pos']))
    cfg.env.commands['twist']=JumpCommandCfg(skill='jump',width=3,resampling_time_range=(100.,100.))
    cfg.env.commands['body_pose']=JumpCommandCfg(skill='jump',width=6,resampling_time_range=(100.,100.))
    cfg.env.events.pop('reset_hop')
    cfg.env.events['reset_refined_jump']=EventTermCfg(func=reset_refined_jump,mode='reset')
    for name,func,weight in [('hop_prepare',prepare,2.),('hop_launch',launch,6.),('hop_flight',flight,8.),
                             ('hop_land',landing,8.),('joint_margin',limit_margin,-.75),('touchdown_speed',touchdown_speed,-2.)]:
        cfg.env.rewards[name]=RewardTermCfg(func=func,weight=weight)
    cfg.env.terminations['joint_limit_failure']=TerminationTermCfg(func=limit_failure,time_out=False)
    cfg.env.rewards['action_rate_l2'].weight=-.05
    for name in ('arm_pose_tracking','tail_pose_tracking','jaw_pose_tracking'):cfg.env.rewards[name].weight=.4
    cfg.agent.experiment_name='microdinosaur_jump_refine'
    return task,cfg
