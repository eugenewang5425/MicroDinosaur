"""Calibrated reset, common head reference, transition episodes and v7 anchors."""
from pathlib import Path
from mjlab.managers import EventTermCfg,RewardTermCfg
from mjlab_microduck.tasks import mdp
from mjlab_microduck.calibrated_head_action import CalibratedHeadActionCfg
from mjlab_microduck.retention_ppo import RetentionPpoCfg
from deep_crouch_cfg import build_config as build_previous

OUT=Path(__file__).parent/'20260914_transition_refine'
TEACHER=Path('D:/microduck_rl/logs/rsl_rl/velocity_microdinosaur/2026-09-13_13-46-57_velocity_microdinosaur/model_15500.pt')


def build_config(envs=64,seed=42):
    task,cfg=build_previous(envs,seed)
    cfg.env.actions['joint_pos']=CalibratedHeadActionCfg(**vars(cfg.env.actions['joint_pos']),
        bank_path=str(OUT/'calibrated_reset_bank.json'))
    cfg.env.events.pop('reset_base');cfg.env.events.pop('reset_robot_joints')
    cfg.env.events['calibrated_stand']=EventTermCfg(func=mdp.reset_calibrated_stand,mode='reset')
    cfg.env.commands['twist']=mdp.TransitionTwistCommandCfg(resampling_time_range=(100.,100.))
    cfg.env.commands['body_pose']=mdp.TransitionHeightCommandCfg(ranges=((0.,0.),)*6,
        resampling_time_range=(100.,100.))
    cfg.env.rewards['head_world_gaze']=RewardTermCfg(func=mdp.navigation_head_attitude,weight=1.5,
        params={'std':(.5,.3,.3)})
    # Preserve the neck's original quarter of the old averaged 4-joint reward.
    cfg.env.rewards['head_pose_tracking']=RewardTermCfg(func=mdp.neck_only_pose_tracking,
        weight=.2,params={'std':.3})
    values=vars(cfg.agent.algorithm).copy();values.pop('class_name')
    cfg.agent.algorithm=RetentionPpoCfg(**values,teacher_checkpoint=str(TEACHER),
        anchor_dataset=str(OUT/'v7_anchor.npz'),anchor_steps=2,anchor_batch_size=1024,anchor_weight=1.)
    cfg.agent.experiment_name='microdinosaur_transition_refine'
    return task,cfg
