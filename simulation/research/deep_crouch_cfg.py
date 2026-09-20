"""Frozen next-stage recipe: head sensor loop + feasible 0/2/3/4cm cycles."""
import json
from pathlib import Path
from mjlab.managers import RewardTermCfg
from mjlab_microduck.tasks import mdp
from mjlab_microduck.head_imu_action import HeadImuPositionActionCfg
from terrain_skill_cfg import build_config as build_base

ROOT = Path(__file__).parent


def build_config(envs=64, seed=42):
    task, cfg = build_base('crouch', envs, seed)
    cfg.env.actions['joint_pos'] = HeadImuPositionActionCfg(**vars(cfg.env.actions['joint_pos']))
    cfg.env.commands['body_pose'] = mdp.CrouchCycleCommandCfg(
        ranges=((0., 0.),)*6, resampling_time_range=(100., 100.))
    twist = cfg.env.commands['twist']
    twist.rel_standing_envs = .75
    twist.rel_turn_in_place_envs = .05
    twist.resampling_time_range = (12., 12.)
    twist.ranges.lin_vel_x = (-.1, .55)
    twist.ranges.lin_vel_y = (-.03, .03)
    twist.ranges.ang_vel_z = (-.35, .35)
    cfg.env.commands['head_pose'].ranges = ((0., 0.),)*4
    # Avoid random tail/arm/jaw gestures during the physical folding stage.
    for name in ('arm_pose', 'tail_pose', 'jaw_pose'):
        cfg.env.commands[name].ranges = tuple((0., 0.) for _ in cfg.env.commands[name].ranges)
    geometry = json.loads((ROOT/'20260914_fold_recovery/fold_geometry.json').read_text())
    contract = json.loads((ROOT/'20260913_handoff/native_v07/contract.json').read_text())
    pose = cfg.env.rewards['pose']
    pose.func = mdp.crouch_reference_posture
    pose.params.update(reference_names=contract['action_names'], reference_table=[r['target'] for r in geometry[:5]])
    # A bounded posture stage: price unwanted zero-command translation/yaw,
    # not foot lifting or joint velocity while executing the commanded crouch.
    cfg.env.rewards['stand_still'] = RewardTermCfg(func=mdp.stand_still_penalty,
        weight=-.5, params={'lin_gain': 1., 'yaw_gain': 4., 'command_threshold': .01})
    # Head joint HOME penalties conflict with active IMU corrections. Retain
    # world gaze rewards and neck command tracking; disable just head HOME bias.
    cfg.env.curriculum.pop('head_pose_bias_weight', None)
    cfg.env.rewards['head_pose_bias'].weight = 0.
    cfg.agent.experiment_name = 'microdinosaur_deep_crouch'
    return task, cfg
