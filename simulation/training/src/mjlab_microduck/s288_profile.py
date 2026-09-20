"""Opt-in protocol-aware training; retains builtin implicit PD and existing delays.

Output encoder feedback and rotor-derived velocity are separate observations.
Encoder zero phase is provisionally aligned with the model joint zero; actual
zero offsets, gearbox compliance and enhanced JY61P firmware need measurement.
"""
from dataclasses import dataclass, replace

import torch
from mjlab.envs.mdp.observations import joint_pos_rel, joint_vel_rel
from mjlab.envs.mdp.dr.joint import joint_friction
from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_microduck.action_filter import FilteredJointPositionAction, FilteredJointPositionActionCfg
from mjlab_microduck.s288_protocol import (
    OUTPUT_ENCODER_STEP, ROTOR_POSITION_OUTPUT_STEP, VELOCITY_STEP,
    KP_STEP, KD_STEP, JY61P_GYRO_STEP, JY61P_GYRO_LIMIT, round_scalar,
)
from mjlab_microduck.tasks import mdp
from mjlab.managers.event_manager import requires_model_fields


def quantize(x, step):
    return torch.round(x / step) * step


def s288_joint_position(env, biased=False, asset_cfg=SceneEntityCfg('robot')):
    asset = env.scene[asset_cfg.name]
    home = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return quantize(joint_pos_rel(env, biased=biased, asset_cfg=asset_cfg) + home, OUTPUT_ENCODER_STEP) - home


def s288_joint_velocity(env, asset_cfg=SceneEntityCfg('robot')):
    return quantize(joint_vel_rel(env, asset_cfg), VELOCITY_STEP)


def jy61p_body_gyro(env, max_angle_deg=3.0):
    gyro = mdp.base_ang_vel_imu_misaligned(env, max_angle_deg)
    return quantize(gyro.clamp(-JY61P_GYRO_LIMIT,JY61P_GYRO_LIMIT), JY61P_GYRO_STEP)


@dataclass(kw_only=True)
class S288PositionActionCfg(FilteredJointPositionActionCfg):
    def build(self, env):
        return S288PositionAction(self, env)


class S288PositionAction(FilteredJointPositionAction):
    def process_actions(self, actions):
        super().process_actions(actions)
        # Host filter state remains continuous; transmitted absolute target is quantized.
        self._processed_actions = quantize(self._processed_actions, ROTOR_POSITION_OUTPUT_STEP)


@requires_model_fields('actuator_gainprm', 'actuator_biasprm')
def quantize_firmware_gains(env, env_ids):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs,device=env.device)
    ids = env.scene['robot'].indexing.ctrl_ids
    index = (env_ids[:,None], ids)
    gain, bias = env.sim.model.actuator_gainprm, env.sim.model.actuator_biasprm
    kp = quantize(gain[index][:,:,0], KP_STEP)
    kd = quantize(-bias[index][:,:,2], KD_STEP)
    gain[env_ids[:,None],ids,0] = kp
    bias[env_ids[:,None],ids,1] = -kp
    bias[env_ids[:,None],ids,2] = -kd


def apply_s288_protocol(cfg):
    """Use known numeric limits, without inventing a measured motor/thermal curve."""
    robot = cfg.scene.entities['robot']
    robot.articulation.actuators = tuple(replace(a, stiffness=round_scalar(a.stiffness,KP_STEP),
        damping=round_scalar(a.damping,KD_STEP)) for a in robot.articulation.actuators)
    old = cfg.actions['joint_pos']
    cfg.actions['joint_pos'] = S288PositionActionCfg(**vars(old))
    actor = cfg.observations['actor'].terms
    actor['joint_pos'].func = s288_joint_position
    actor['joint_vel'].func = s288_joint_velocity
    actor['base_ang_vel'].func = jy61p_body_gyro
    # The inherited BAM-only friction event silently skips builtin S288 PD.
    # Use mjlab's nominal-relative model-field randomizer; reset never compounds.
    if 'randomize_joint_friction' in cfg.events:
        old_friction = cfg.events['randomize_joint_friction']
        cfg.events['randomize_joint_friction'] = replace(old_friction,
            func=joint_friction, params={
                'asset_cfg': actor['joint_vel'].params['asset_cfg'],
                'ranges': old_friction.params['scale_range'], 'operation': 'scale'})
    # Insertion order: quantize after the existing per-reset gain randomizer.
    cfg.events['s288_wire_gains'] = EventTermCfg(func=quantize_firmware_gains, mode='reset')
    return cfg
