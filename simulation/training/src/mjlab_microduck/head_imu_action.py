"""Opt-in v07 sensor head control before EMA, wire quantization and motor delay.

Training starts its tilt estimator from the first delivered valid acceleration,
with known zero simulated bias. This is an accelerated reset, NOT the verified
6-second stationary calibration used at CPU evaluation/deployment. No world
orientation is available to this action. The actor remains 81D and raw actions
remain 19D. Body heading feedback is still an external navigation controller.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import mujoco
import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab_microduck.s288_profile import S288PositionActionCfg, quantize
from mjlab_microduck.s288_protocol import JY61P_GYRO_STEP, JY61P_GYRO_LIMIT, ROTOR_POSITION_OUTPUT_STEP
from mjlab_microduck.head_attitude_torch import TorchHeadController, TorchImu, exp_so3


def static_kinematics(model):
    names = ('neck_pitch', 'head_pitch', 'head_yaw', 'head_roll')
    joints = [model.joint('robot/'+n).id for n in names]
    bodies = model.jnt_bodyid[joints]
    assert list(model.body_parentid[bodies][1:]) == list(bodies[:-1])
    def rotation(q):
        r = np.empty(9); mujoco.mju_quat2Mat(r, q); return r.reshape(3, 3)
    return SimpleNamespace(names=names, fixed=np.array([rotation(q) for q in model.body_quat[bodies]]),
        axes=model.jnt_axis[joints], reference=model.qpos0[model.jnt_qposadr[joints]],
        tip=rotation(model.site_quat[model.site('robot/head_imu').id]),
        base=rotation(model.site_quat[model.site('robot/imu').id]), limits=model.jnt_range[joints])


@dataclass(kw_only=True)
class HeadImuPositionActionCfg(S288PositionActionCfg):
    yaw_follows_trunk: bool = False
    """True 时头部的期望偏航跟随**躯干**,而不是锁世界坐标。

    实测(2026-09-16):控制器原本把头部世界偏航锁在 yaw_reference(=转向指令积分,起身任务恒为 0),
    于是躯干偏航多少它就反向拧 head_yaw 多少 —— 四例实测 躯干偏航+head_yaw偏差 ≈ 0
    (+54.9/−51.2、+147.0/−146.4、−62.1/+61.9、−17.6/+13.7)。后果:站姿判据最大偏差 146°、
    用户看图说"站着头朝后"。用户设计意图是"头部稳定正向应与躯干方向一致",故起身开启此项。
    走路/蹲/跑**不启用**:它们的航向反馈依赖"镜头看向指令航向"。
    """
    def build(self, env):
        return HeadImuPositionAction(self, env)


class HeadImuPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.env = env
        assert abs(env.step_dt-.02) < 1e-8
        kin = static_kinematics(env.sim.mj_model)
        self.chain = [self._target_names.index(n) for n in kin.names]
        self.head = self.chain[1:]
        self.controller = TorchHeadController(kin, env.num_envs, env.device)
        self.imu = TorchImu(env.num_envs, env.device)
        self._applied = self._offset.clone()
        self.packet = torch.zeros(env.num_envs, 9, device=env.device)
        self.packet_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.ready = self.packet_valid.clone()
        self.alignment = torch.eye(3, device=env.device).repeat(env.num_envs, 1, 1)
        self.yaw_reference = torch.zeros(env.num_envs, device=env.device)
        self.yaw_rate = torch.zeros_like(self.yaw_reference)
        self.sensor_ids = []
        for name in ('head_imu_ang_vel', 'head_imu_accel', 'imu_accel'):
            s = env.sim.mj_model.sensor('robot/'+name)
            assert s.dim[0] == 3
            self.sensor_ids.extend(range(int(s.adr[0]), int(s.adr[0])+3))

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._applied[ids] = self._offset[ids]
        self.controller.reset(ids, self._offset[ids][:, self.head])
        self.packet[ids] = 0
        self.packet_valid[ids] = False
        self.ready[ids] = False
        self.yaw_reference[ids] = 0
        self.yaw_rate[ids] = 0
        self.alignment[ids] = torch.eye(3, device=self.env.device)
        self.imu.r[ids] = torch.eye(3, device=self.env.device)
        self.imu.accel[ids] = 0
        self.imu.bias[ids] = 0

    def process_actions(self, actions):
        from mjlab_microduck.head_attitude_torch import tilt_from_accel
        super().process_actions(actions)
        # Reuse the previously computed actor observation. Computing it again
        # would advance noise/delay history and consume a different packet.
        obs = self.env.obs_buf['actor']
        assert obs.shape[1] == 81
        measured = (obs[:, 6:25]+self._offset)[:, self.chain]
        old = self.packet.clone()  # delivered source is exactly 20 ms old
        gyro, accel, body_accel = old[:, :3], old[:, 3:6], old[:, 6:9]
        start = self.packet_valid & ~self.ready & (accel.norm(dim=-1) > 7.8) & (accel.norm(dim=-1) < 11.8)
        ids = start.nonzero(as_tuple=False).flatten()
        self.imu.initialize(ids, accel[ids])
        relative, _ = self.controller.forward(measured)
        expected = tilt_from_accel(body_accel)@relative
        difference = torch.atan2(expected[:, 1, 0], expected[:, 0, 0])-torch.atan2(self.imu.r[:, 1, 0], self.imu.r[:, 0, 0])
        vector = torch.zeros_like(gyro); vector[:, 2] = difference
        self.alignment[ids] = exp_so3(vector[ids])
        # Do not integrate the packet used for initialization a second time.
        before = self.imu.r.clone()
        self.imu.update(gyro, accel, .02)
        self.imu.r[start | ~self.ready] = before[start | ~self.ready]
        self.ready |= start
        # Navigation turn reference integrates user command only, with the
        # same 1.5 rad/s^2 slew and 0.7 rad/s limit as the CPU reference.
        requested = self.env.command_manager.get_command('twist')[:, 2].clamp(-.7, .7)
        self.yaw_rate += (requested-self.yaw_rate).clamp(-.03, .03)
        self.yaw_reference += self.yaw_rate*.02
        vector.zero_(); vector[:, 2] = self.yaw_reference
        if getattr(self.cfg, 'yaw_follows_trunk', False):
            qq = self.env.scene['robot'].data.root_link_quat_w
            qw, qx, qy, qz = qq[:, 0], qq[:, 1], qq[:, 2], qq[:, 3]
            vector[:, 2] = vector[:, 2] + torch.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
        desired = exp_so3(vector)
        omega = torch.zeros_like(gyro); omega[:, 2] = self.yaw_rate
        age = torch.where(self.ready, .02, 1.)
        target = self._processed_actions.clone()
        corrected = self.controller.update(target[:, self.head], measured,
            self.alignment@self.imu.r, gyro-self.imu.bias, desired, omega, .02, age)
        target[:, self.head] = corrected
        filtered = self._applied+self.cfg.lp_alpha*(target-self._applied) if self.cfg.lp_alpha > 0 else target
        if self.cfg.max_delta is not None:
            filtered = self._applied+(filtered-self._applied).clamp(-self.cfg.max_delta, self.cfg.max_delta)
        self._applied = filtered
        self._processed_actions = quantize(filtered, ROTOR_POSITION_OUTPUT_STEP)
        # New raw IMU sample enters transport AFTER control, for next step.
        self.packet = self.env.sim.data.sensordata[:, self.sensor_ids].clone()
        self.packet[:, :3] = quantize((self.packet[:, :3]+torch.randn_like(gyro)*np.deg2rad(.05)).clamp(
            -JY61P_GYRO_LIMIT, JY61P_GYRO_LIMIT), JY61P_GYRO_STEP)
        self.packet[:, 3:] += torch.randn_like(self.packet[:, 3:])*.02
        self.packet_valid[:] = True
