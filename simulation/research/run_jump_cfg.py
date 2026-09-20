"""Separate running/hop specialists on v07, shared S288/IMU execution contract."""
from dataclasses import dataclass, fields
import json
from pathlib import Path

import mujoco
import torch
from mjlab.managers import CommandTerm, CommandTermCfg, EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg
from mjlab_microduck.imu_owned_head import ImuOwnedActionCfg
from transition_refine_cfg import build_config as previous
from terrain_skill_cfg import XML
from evaluate_policy import sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_run_jump'
SOURCE=Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_demo_demonstration/20260914_train_512x101/model_16250.pt')
PROXIES=ROOT/'20260914_terrain_skills/planned_motion_contacts/audit.json'
BANK=ROOT/'20260914_transition_refine/calibrated_reset_bank.json'
STAND_Z=sum(r['root_qpos'][2] for r in json.loads(BANK.read_text())['records'])/len(json.loads(BANK.read_text())['records'])


def robot_spec():
    """Reuse audited current-CAD floor boxes; no extra mass or self contacts."""
    audit=json.loads(PROXIES.read_text())
    assert audit['current_xml_sha256']==sha(XML)
    spec=mujoco.MjSpec.from_file(str(XML))
    for proxy in audit['proxies']:
        name=proxy['body'].removeprefix('robot/')
        spec.body(name).add_geom(name=name+'_floor_proxy',type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=proxy['center'],size=proxy['half_size'],contype=4,conaffinity=0,
            group=3,density=0,rgba=[.3,.6,.4,.15])
    return spec


def ground_proxy_mask(spec):
    for geom in spec.geoms:
        if geom.group==0:
            geom.conaffinity |= 4


class MotionCommand(CommandTerm):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.elapsed=torch.zeros(self.num_envs,device=self.device)
        self.speed=torch.zeros_like(self.elapsed)
        self._command=torch.zeros((self.num_envs,cfg.width),device=self.device)

    @property
    def command(self): return self._command

    def compute(self,dt):
        if dt>0: super().compute(dt)

    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        self.elapsed[ids]=0;self._command[ids]=0
        return super().reset(env_ids)

    def _resample_command(self,env_ids):
        self.speed[env_ids]=torch.empty(len(env_ids),device=self.device).uniform_(.45,.8)

    def _update_command(self):
        self.elapsed+=self._env.step_dt
        if self.cfg.width==3:
            target=torch.where((self.elapsed>=1.)&(self.elapsed<7.),self.speed,0.) if self.cfg.skill=='run' else torch.zeros_like(self.speed)
            self._command[:,0]+=(target-self._command[:,0]).clamp(-.03,.03)
        elif self.cfg.skill=='jump':
            # Existing body-pose Z semantics: crouch request, upward request,
            # return to neutral. This is a command, not a joint demonstration.
            z=torch.where(self.elapsed<1.5,-.03,torch.where(self.elapsed<2.2,.02,0.))
            self._command[:,2]=z

    def _update_metrics(self): pass


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
    skill:str='run'
    width:int=3
    def build(self,env): return MotionCommand(self,env)


def contacts(env,name):
    return env.scene[name].data.found.reshape(env.num_envs,-1)>0


def nonfoot_contact(env):
    return contacts(env,'nonfoot_ground').any(-1)


class HopState:
    def __init__(self,n,device):
        self.velocity_frontier=torch.zeros(n,device=device)
        self.flight_frontier=torch.zeros(n,device=device)
        self.flight=torch.zeros(n,device=device)
        self.launched=torch.zeros(n,dtype=torch.bool,device=device)
        self.dv=torch.zeros(n,device=device)
        self.df=torch.zeros(n,device=device)
        self.last_step=-1

    def reset(self,ids):
        for name in ('velocity_frontier','flight_frontier','flight','launched','dv','df'):
            getattr(self,name)[ids]=0

    def update(self,step,dt,vz,supported,upright,launch_window):
        if step==self.last_step:return
        self.last_step=step
        eligible=launch_window & upright
        frontier=torch.maximum(self.velocity_frontier,torch.where(eligible,vz.clamp(0,.5),0.))
        self.dv=frontier-self.velocity_frontier;self.velocity_frontier=frontier
        self.launched |= eligible & (vz>.1)
        self.flight=torch.where(~supported & upright & self.launched,self.flight+dt,0.)
        maximum=torch.maximum(self.flight_frontier,self.flight.clamp(max=.12))
        self.df=maximum-self.flight_frontier;self.flight_frontier=maximum


def hop_state(env):
    if not hasattr(env,'_hop_state'):env._hop_state=HopState(env.num_envs,env.device)
    s=env._hop_state
    robot=env.scene['robot']
    t=env.command_manager.get_term('twist').elapsed
    supported=contacts(env,'robot_ground').any(-1)
    # Root projected gravity is privileged reward data, never a new actor input.
    upright=robot.data.projected_gravity_b[:,2]<-.866
    s.update(env.common_step_counter,env.step_dt,robot.data.root_link_lin_vel_w[:,2],supported,upright,(t>=1.5)&(t<2.5))
    return s


def reset_hop(env,env_ids):
    if not hasattr(env,'_hop_state'):env._hop_state=HopState(env.num_envs,env.device)
    env._hop_state.reset(slice(None) if env_ids is None else env_ids)


def hop_preparation(env):
    t=env.command_manager.get_term('twist').elapsed
    z=env.scene['body_clearance'].data.heights[:,0]
    return torch.exp(-((z-(STAND_Z-.03))/.015)**2)*(t<1.5)


def hop_launch_progress(env):return hop_state(env).dv/.5/env.step_dt
def hop_flight_progress(env):return hop_state(env).df/.12/env.step_dt


def hop_landing(env):
    s=hop_state(env);r=env.scene['robot'];t=env.command_manager.get_term('twist').elapsed
    z=env.scene['body_clearance'].data.heights[:,0]
    good=(s.flight_frontier>=.04)&contacts(env,'feet_ground_contact').all(-1)&(t>=2.2)
    speed=r.data.root_link_lin_vel_w.square().sum(-1)
    tilt=r.data.projected_gravity_b[:,:2].square().sum(-1)
    return good*torch.exp(-((z-STAND_Z)/.02)**2-speed/.01-tilt/.04)


def lateral_drift(env):
    v=env.scene['robot'].data.root_link_lin_vel_b
    return v[:,:2].square().sum(-1)


def run_airborne(env):
    r=env.scene['robot'];cmd=env.command_manager.get_command('twist')[:,0]
    airborne=~contacts(env,'robot_ground').any(-1)
    return airborne*(cmd>.45)*torch.exp(-((r.data.root_link_lin_vel_b[:,0]-cmd)/.25)**2)*(r.data.projected_gravity_b[:,2]<-.94)


def build_config(skill,envs=64,seed=51):
    assert skill in ('run','jump')
    task,cfg=previous(envs,seed)
    cfg.env.scene.entities['robot'].spec_fn=robot_spec
    cfg.env.scene.spec_fn=ground_proxy_mask
    cfg.env.sim.mujoco.iterations=30
    cfg.env.sim.mujoco.ls_iterations=50
    cfg.env.actions['joint_pos']=ImuOwnedActionCfg(**vars(cfg.env.actions['joint_pos']),owned_head_indices=(1,))
    cfg.env.scene.sensors=(*cfg.env.scene.sensors,
        ContactSensorCfg(name='robot_ground',primary=ContactMatch(mode='subtree',pattern='trunk_base',entity='robot'),
            secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found',),reduce='none',num_slots=1),
        ContactSensorCfg(name='nonfoot_ground',primary=ContactMatch(mode='geom',pattern='.*_floor_proxy',entity='robot'),
            secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found',),reduce='none',num_slots=1))
    cfg.env.commands['twist']=MotionCommandCfg(skill=skill,width=3,resampling_time_range=(100.,100.))
    cfg.env.commands['body_pose']=MotionCommandCfg(skill=skill,width=6,resampling_time_range=(100.,100.))
    cfg.env.episode_length_s=10. if skill=='run' else 6.
    cfg.env.terminations['nonfoot_contact']=TerminationTermCfg(func=nonfoot_contact,time_out=False)
    # Preserve inertial/friction/latency DR; no impulsive reset or push supplies
    # takeoff energy. Standalone specialists do not use old-skill teacher loss.
    cfg.env.events.pop('push_robot',None)
    cfg.env.events['calibrated_stand'].params={'yaw_range':(-.06,.06)}
    cfg.env.curriculum={k:v for k,v in cfg.env.curriculum.items() if k in ('com_range','head_com_range')}
    allowed={f.name for f in fields(PpoWithSymmetryCfg)}
    values={k:v for k,v in vars(cfg.agent.algorithm).items() if k in allowed and k!='class_name'}
    cfg.agent.algorithm=PpoWithSymmetryCfg(**values)
    cfg.env.rewards['pose'].weight=.15 if skill=='run' else .05
    cfg.env.rewards['action_rate_l2'].weight=-.05 if skill=='run' else -.03
    cfg.env.rewards['head_gaze'].weight=-.05
    cfg.env.rewards['head_world_gaze'].weight=.8
    for name in ('arm_pose_tracking','tail_pose_tracking','jaw_pose_tracking'):
        cfg.env.rewards[name].weight=.1
    if skill=='run':
        cfg.env.rewards['track_linear_velocity'].weight=4.
        cfg.env.rewards['body_pose_tracking'].params={'std':.025}
        cfg.env.rewards['body_pose_tracking'].weight=.5
        cfg.env.rewards['air_time'].params.update(threshold_min=.06,threshold_max=.20)
        cfg.env.rewards['airborne_running']=RewardTermCfg(func=run_airborne,weight=.8)
    else:
        cfg.env.events['reset_hop']=EventTermCfg(func=reset_hop,mode='reset')
        for name in ('air_time','foot_clearance','foot_swing_height','stand_slip','stand_jitter',
                     'stand_still','body_pose_tracking','track_linear_velocity'):
            cfg.env.rewards.pop(name,None)
        cfg.env.rewards['hop_prepare']=RewardTermCfg(func=hop_preparation,weight=2.)
        cfg.env.rewards['hop_launch']=RewardTermCfg(func=hop_launch_progress,weight=6.)
        cfg.env.rewards['hop_flight']=RewardTermCfg(func=hop_flight_progress,weight=8.)
        cfg.env.rewards['hop_land']=RewardTermCfg(func=hop_landing,weight=4.)
        cfg.env.rewards['translation']=RewardTermCfg(func=lateral_drift,weight=-.5)
    cfg.agent.experiment_name='microdinosaur_'+skill+'_specialist'
    return task,cfg
