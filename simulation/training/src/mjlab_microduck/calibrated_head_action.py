"""Opt-in calibrated standing snapshots; keeps the previous action kernel.

Snapshots come from measured, delayed IMU packets during a 6s CPU v7 stand.
They are a reset distribution, not online calibration of each randomized plant.
Physics pose is written by the reset event; sensor state never reads world pose.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import torch
from mjlab_microduck.head_imu_action import HeadImuPositionAction, HeadImuPositionActionCfg
from mjlab_microduck.head_attitude_torch import TorchImu
from mjlab_microduck.s288_profile import quantize
from mjlab_microduck.s288_protocol import ROTOR_POSITION_OUTPUT_STEP


class SeededImu(TorchImu):
    def __init__(self, n, device):
        super().__init__(n, device)
        self.skip_first = torch.zeros(n, device=device, dtype=torch.bool)

    def update(self, gyro, accel, dt):
        # The calibration-ending packet is already represented in the seed.
        # CPU also does not integrate this held packet twice.
        r, a = self.r.clone(), self.accel.clone()
        super().update(gyro, accel, dt)
        self.r[self.skip_first] = r[self.skip_first]
        self.accel[self.skip_first] = a[self.skip_first]
        self.skip_first[:] = False
        return self.r


@dataclass(kw_only=True)
class CalibratedHeadActionCfg(HeadImuPositionActionCfg):
    bank_path: str

    def build(self, env): return CalibratedHeadAction(self, env)


class CalibratedHeadAction(HeadImuPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        data = json.loads(Path(cfg.bank_path).read_text())
        assert data['action_names'] == self._target_names and data['sample_count'] >= 20
        self.bank = {k: torch.tensor([r[k] for r in data['records']], device=env.device, dtype=torch.float32)
            for k in ('root_qpos','root_qvel','joint_pos','joint_vel','raw_action','applied',
                      'head_rotation','head_accel','head_bias','alignment','packet','heading_world_zero')}
        self.bank_index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.reward_heading_zero = torch.zeros(env.num_envs, device=env.device)
        self.imu = SeededImu(env.num_envs, env.device)
        # Allocate the one fused v07 command buffer before partial resets. A
        # snapshot row later fills the ring without advancing other rows.
        assert len(self._entity.actuators) == 1
        self.motor_buffer = self._entity.actuators[0]._delay_buffer
        assert self.motor_buffer is not None
        if not self.motor_buffer.is_initialized:
            self.motor_buffer.append(torch.zeros_like(self._applied))

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        selection = self.bank_index[ids]
        self.imu.r[ids] = self.bank['head_rotation'][selection]
        self.imu.accel[ids] = self.bank['head_accel'][selection]
        self.imu.bias[ids] = self.bank['head_bias'][selection]
        self.imu.skip_first[ids] = True
        self.alignment[ids] = self.bank['alignment'][selection]
        self.packet[ids] = self.bank['packet'][selection]
        self.packet_valid[ids] = self.ready[ids] = True
        self._applied[ids] = self.bank['applied'][selection]
        self._processed_actions[ids] = quantize(self._applied[ids], ROTOR_POSITION_OUTPUT_STEP)
        self._raw_actions[ids] = self.bank['raw_action'][selection]
        self.controller.reset(ids, self._applied[ids][:,self.head])
        for field in ('_action','_prev_action','_prev_prev_action'):
            getattr(self.env.action_manager,field)[ids] = self._raw_actions[ids]
        circular = self.motor_buffer._buffer
        targets = self._processed_actions[ids]-self._entity.data.encoder_bias[ids][:,self._target_ids]
        circular._buffer[:,ids] = targets
        circular._num_pushes[ids] = circular._max_len

