"""Flamingo(单脚站立)任务的 mdp 函数——移植自官方 flamingo 分支。

来源: pollen-robotics/microduck_rl @ flamingo 分支 tasks/mdp.py。
对移植的改动:关节目标改为**按名字**给定(我们的 19 关节契约顺序);
qpos/qvel 列地址用 jnt_qposadr/jnt_dofadr 计算(不假设连续)。
"""
import math

import numpy as np
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.entity import Entity

_DEFAULT_ASSET_CFG = SceneEntityCfg(name='robot')

# 前伸姿态(用户命名; 策略自己找到的平衡解, 可作上台阶初始动作)
_FORWARD = {'left_hip_pitch': 5., 'left_knee': 23., 'left_ankle': -19.}
_FOLDED = {'left_hip_pitch': 69., 'left_knee': -46., 'left_ankle': 46.}


def _servo_joint_ids(env, asset: Entity) -> list:
    cache = env.__dict__.setdefault('_servo_joint_ids_cache', {})
    key = id(asset)
    ids = cache.get(key)
    if ids is None:
        ids, _ = asset.find_joints(r'^(?!passive_).*')
        cache[key] = ids
    return ids


def _servo_joint_pos(env, asset):
    return asset.data.joint_pos[:, _servo_joint_ids(env, asset)]


def _servo_joint_vel(env, asset):
    return asset.data.joint_vel[:, _servo_joint_ids(env, asset)]


def set_flamingo_state(env, env_ids: torch.Tensor, joint_pose_by_name: dict,
                       base_roll: float, base_pitch: float,
                       joint_pose_by_name_mirror: dict | None = None,
                       z_min: float = .115, z_max: float = .125,
                       asset_cfg=_DEFAULT_ASSET_CFG, tilt_noise: float = 0.0,
                       joint_noise_std: float = 0.0, standing_prob: float = 0.0,
                       standing_z_min: float = .11, standing_z_max: float = .12,
                       side_fixed: float = 0.0, pose_fixed: float = 0.0):
    """在单脚姿态里出生(躯干侧倾/俯仰使支撑脚掌平贴地面),带关节与倾斜噪声、
    随机偏航;standing_prob 比例的环境在 HOME 直立出生(stage-2 过渡训练)。
    joint_pose_by_name: {关节名: 目标角(rad)},未提到的关节保持 HOME。"""
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)
    dev = env.device

    # 出生决定 side + pose, 并**钉进命令项**(官方 cycle 同构)。
    # mjlab 的 _reset_idx 里 reset 事件**先于**命令 reset 运行, 所以这里写的值
    # 就是本回合的指令; 命令在 obs 里, 策略看得见"该做哪一侧/哪个姿态"。
    # side_fixed/pose_fixed != 0 时全批固定(验收用: 必须能**按组**测,
    # 跟着出生随机的话某一组可能一个样本都没有, 无法给绑定指标)。
    if side_fixed:
        side = torch.full((num,), float(side_fixed), device=dev)
    else:
        side = torch.where(torch.rand(num, device=dev) < .5,
                           torch.ones(num, device=dev), -torch.ones(num, device=dev))
    if pose_fixed:
        pose = torch.full((num,), float(pose_fixed), device=dev)
    else:
        pose = torch.where(torch.rand(num, device=dev) < .5,
                           torch.ones(num, device=dev), -torch.ones(num, device=dev))
    # 存到 env, 由 **命令项从中读取**(而不是事件直接写命令项):
    # 实测 mjlab 的 _reset_idx 里命令 reset 在事件之后运行, 会重新 resample 把事件
    # 写进去的值覆盖掉 —— 第一版就是这样, 指令全长 0(而且被 try/except 静默吞掉)。
    ids = env_ids.long()
    if not hasattr(env, 'spawn_side') or env.spawn_side.shape[0] != env.num_envs:
        env.spawn_side = torch.ones(env.num_envs, device=dev)
        env.spawn_pose = torch.ones(env.num_envs, device=dev)
    env.spawn_side[ids] = side
    env.spawn_pose[ids] = pose
    use_forward = pose > 0
    is_stand = torch.rand(num, device=dev) < standing_prob
    yaw = torch.rand(num, device=dev) * 2 * np.pi - np.pi
    # 躯干侧倾方向按 side: 右脚支撑 -> +22.6°, 左脚支撑 -> -22.6°
    roll = float(base_roll) * side
    pitch = torch.full((num,), float(base_pitch), device=dev)
    if tilt_noise > 0.0:
        roll = roll + (torch.rand(num, device=dev) * 2 - 1) * tilt_noise
        pitch = pitch + (torch.rand(num, device=dev) * 2 - 1) * tilt_noise
    roll = torch.where(is_stand, torch.zeros_like(roll), roll)
    pitch = torch.where(is_stand, torch.zeros_like(pitch), pitch)
    cy, sy = torch.cos(yaw * .5), torch.sin(yaw * .5)
    cp, sp = torch.cos(pitch * .5), torch.sin(pitch * .5)
    cr, sr = torch.cos(roll * .5), torch.sin(roll * .5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat = torch.stack([qw, qx, qy, qz], dim=1)

    z = torch.rand(num, device=dev) * (z_max - z_min) + z_min
    z_stand = torch.rand(num, device=dev) * (standing_z_max - standing_z_min) + standing_z_min
    z = torch.where(is_stand, z_stand, z)

    env.sim.data.qpos[env_ids, 0:2] = (
        env.scene.env_origins[env_ids.long(), :2])   # 底座 xy = 车道原点
    env.sim.data.qpos[env_ids, 2] = z + env.scene.env_origins[env_ids.long(), 2]
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qvel[env_ids, :6] = 0.0

    asset: Entity = env.scene[asset_cfg.name]
    servo_ids = _servo_joint_ids(env, asset)
    # 官方同款列地址:自由关节 7 维在前,舵机关节连续(qpos 7+j / qvel 6+j)
    qcols = torch.as_tensor([7 + j for j in servo_ids], device=dev, dtype=torch.long)
    vcols = torch.as_tensor([6 + j for j in servo_ids], device=dev, dtype=torch.long)
    home = asset.data.default_joint_pos[env_ids.long()][:, servo_ids]
    pose = home.clone()
    names = _names_of(asset)
    lift_left = side > 0                 # +1 = 左脚摆动 / 右脚支撑
    # 支撑腿保持姿态表里的值(它是站住的腿)。
    # **必须按 side 选表**: 姿态表描述的是"右脚支撑"(右腿 14.8/0.3/-14.5 站立,
    # 左腿 68.8/-45.8/45.8 后折); side=-1(左脚支撑)时要用镜像表, 否则两条腿都拿不到
    # 站立支撑姿, 出生瞬间两只脚全悬空(实测: 左脚 z 118mm/右脚 129mm, 支撑脚着地
    # 奖励与摆动脚高度奖励整组归零)。
    tbl_plus = joint_pose_by_name
    tbl_minus = (joint_pose_by_name_mirror if joint_pose_by_name_mirror is not None
                 else joint_pose_by_name)
    for n in set(tbl_plus) | set(tbl_minus):
        if n not in names:
            continue
        idx = names.index(n)
        vp = tbl_plus.get(n)
        vm = tbl_minus.get(n)
        cur = pose[:, idx]
        a = torch.full((num,), float(vp), device=dev) if vp is not None else cur
        b = torch.full((num,), float(vm), device=dev) if vm is not None else cur
        pose[:, idx] = torch.where(lift_left, a, b)
    # 摆动腿: 按 side 选腿, 按 pose 选前伸/后折(右腿 = 左腿值取反, 同 mirror_pose)
    for lift_l, sgn in ((True, 1.), (False, -1.)):
        m = (lift_left == lift_l)
        if not m.any():
            continue
        for base_j in ('left_hip_pitch', 'left_knee', 'left_ankle'):
            jn = base_j if lift_l else mirror_joint_name(base_j)
            if jn not in names:
                continue
            tf = torch.deg2rad(torch.tensor(float(_FORWARD[base_j]) * sgn, device=dev))
            tl = torch.deg2rad(torch.tensor(float(_FOLDED[base_j]) * sgn, device=dev))
            v = torch.where(m & use_forward, tf.expand(num).to(dev),
                            tl.expand(num).to(dev))
            pose[:, names.index(jn)] = torch.where(m, v, pose[:, names.index(jn)])
    pose = torch.where(is_stand.unsqueeze(1), home, pose)
    if joint_noise_std > 0.0:
        pose = pose + torch.randn_like(pose) * joint_noise_std
    env.sim.data.qpos[env_ids.unsqueeze(1).long(), qcols.unsqueeze(0)] = pose
    env.sim.data.qvel[env_ids.unsqueeze(1).long(), vcols.unsqueeze(0)] = 0.0


def _names_of(asset):
    cache = asset.__dict__.setdefault('_names_cache', {})
    if 'names' not in cache:
        names = getattr(asset, 'joint_names', None)
        if names is None:
            names = [asset.joint(i).name for i in range(asset.num_joints)]
        cache['names'] = [n.split('/')[-1] for n in names]
    return cache['names']


def _foot_found(env, sensor_name: str, slot: int) -> torch.Tensor:
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    found = env.scene.sensors[sensor_name].data.found
    if found.dim() > 1:
        found = found[:, slot]
    return torch.clamp(torch.nan_to_num(found.float(), nan=0.), 0., 1.)


def foot_contact_reward(env, sensor_name: str, slot: int) -> torch.Tensor:
    """+1 while the stance foot touches the ground (pin it, anti-hop)."""
    return _foot_found(env, sensor_name, slot)


def foot_contact_penalty(env, sensor_name: str, slot: int) -> torch.Tensor:
    """−1 while the swing foot touches ground (self-negating → POSITIVE weight).
    摆动脚落地是**软失败**:便宜、绝不终止(官方设计)。"""
    return -_foot_found(env, sensor_name, slot)


def _robot_com_xy(env, asset: Entity) -> torch.Tensor:
    """整机水平质心(根 body 的 subtree_com)。不能用 trunk 自身 CoM——
    头占 38% 质量,抬起一条腿后两者差数毫米。"""
    com = env.sim.data.subtree_com[:, asset.indexing.root_body_id, :2]
    return torch.nan_to_num(com, nan=0.)


def com_over_foot(env, asset_cfg: SceneEntityCfg, std: float = .02) -> torch.Tensor:
    """质心到支撑脚 site 的水平距离高斯(std 0.03 起步,com_std 课程收到 0.02)。
    **这是主平衡信号**——比"直立"重要得多。"""
    asset: Entity = env.scene[asset_cfg.name]
    com_xy = _robot_com_xy(env, asset)
    foot_xy = asset.data.site_pos_w[:, asset_cfg.site_ids[0], :2]
    dist2 = torch.nan_to_num(((com_xy - foot_xy) ** 2).sum(dim=-1), nan=1.)
    return torch.exp(-dist2 / (std ** 2))


def foot_height_gaussian(env, asset_cfg: SceneEntityCfg, target: float = .05,
                         std: float = .03, gate_sensor_name=None, gate_slot: int = 1):
    """摆动脚 site 高度围绕 target 的高斯;门=支撑脚接触(防跳跃刷分)。"""
    asset: Entity = env.scene[asset_cfg.name]
    z = asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2] \
        - env.scene.terrain.env_origins[:, 2]
    z = torch.nan_to_num(z, nan=0.)
    reward = torch.exp(-((z - target) / std) ** 2)
    if gate_sensor_name is not None:
        reward = reward * _foot_found(env, gate_sensor_name, gate_slot)
    return reward


def projected_gravity_match(env, target=(0., 0., -1.), std: float = .15,
                            asset_cfg=_DEFAULT_ASSET_CFG):
    """‖g_b − target‖ 高斯:奖励**保持特定躯干倾斜**( flamingo 躯干侧倾 ~24°,
    普通直立奖励会和姿态打架)。"""
    asset: Entity = env.scene[asset_cfg.name]
    g = torch.nan_to_num(asset.data.projected_gravity_b, nan=0.)
    t = torch.tensor(target, device=env.device, dtype=g.dtype).unsqueeze(0)
    err2 = ((g - t) ** 2).sum(dim=-1)
    return torch.exp(-err2 / (std ** 2))


def projected_gravity_match_cmd(env, target=(-.154, -.379, -.912), std: float = .15,
                                asset_cfg=_DEFAULT_ASSET_CFG):
    """指令驱动的重力目标: side 决定躯干往哪边侧倾, 所以重力在基座系的 y 分量要
    跟着翻。镜像时只翻 y —— 侧倾是绕 x 的 roll, pitch 贡献落在 x 上, 不翻。"""
    asset: Entity = env.scene[asset_cfg.name]
    g = torch.nan_to_num(asset.data.projected_gravity_b, nan=0.)
    t = torch.tensor(target, device=env.device, dtype=g.dtype).unsqueeze(0)
    t = t.expand(env.num_envs, 3).clone()
    st = _side(env) > 0                      # +1 = 右脚支撑
    t[:, 1] = torch.where(st, t[:, 1], -t[:, 1])
    return torch.exp(-((g - t) ** 2).sum(dim=-1) / (std ** 2))


def lateral_tilt_penalty(env, threshold: float = .45, direction: float = -1.,
                         asset_cfg=_DEFAULT_ASSET_CFG):
    """躯干向 direction 侧滚过 threshold 的二次罚(≤0,自抵消→正权重)。
    向支撑腿侧倒没有东西接住——硬失败方向,早罚、重罚;向摆动腿侧倒便宜。"""
    asset: Entity = env.scene[asset_cfg.name]
    gy = torch.nan_to_num(asset.data.projected_gravity_b[:, 1], nan=0.)
    over = torch.clamp(direction * gy - threshold, min=0.)
    return -(over ** 2)


def joint_vel_gaussian(env, std: float = 1., gate_sensor_name=None, gate_slot: int = 1,
                       asset_cfg=_DEFAULT_ASSET_CFG):
    """安静:exp(−mean(q̇²)/std²),门=支撑脚接触(躺平不动不得分)。"""
    asset: Entity = env.scene[asset_cfg.name]
    qd = torch.nan_to_num(_servo_joint_vel(env, asset), nan=0.)
    reward = torch.exp(-(qd ** 2).mean(dim=-1) / (std ** 2))
    if gate_sensor_name is not None:
        reward = reward * _foot_found(env, gate_sensor_name, gate_slot)
    return reward


def joint_vel_gaussian_cmd(env, std: float = 1., sensor_name='feet_ground_contact',
                           asset_cfg=_DEFAULT_ASSET_CFG):
    """安静(指令驱动门): 门 = **支撑脚**着地, slot 按 side 选 —— 写死 slot 会让
    镜像侧整组归零(实测 side=-1 组全 0), 而这一项正是"别抖"的信号。"""
    asset: Entity = env.scene[asset_cfg.name]
    qd = torch.nan_to_num(_servo_joint_vel(env, asset), nan=0.)
    reward = torch.exp(-(qd ** 2).mean(dim=-1) / (std ** 2))
    st = _side(env) > 0                     # +1 = 右脚支撑 -> slot 1
    found = _foot_found(env, sensor_name, 1)
    found_l = _foot_found(env, sensor_name, 0)
    return reward * torch.where(st, found, found_l)


def pose_target_match(env, target_overrides=None, asset_cfg=_DEFAULT_ASSET_CFG,
                      std: float = .3, joint_indices=None):
    """围绕固定目标姿态的高斯姿态匹配(t=0 起全程同一目标,无路径插值)。"""
    asset = env.scene[asset_cfg.name]
    target = _servo_default(env, asset)
    if target_overrides:
        names = _names_of(asset)
        for n, val in target_overrides.items():
            if n in names:
                target[:, names.index(n)] = val
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return torch.exp(-((joint_pos - target) / std) ** 2).mean(dim=-1)


def _servo_default(env, asset):
    d = asset.data.default_joint_pos[:, _servo_joint_ids(env, asset)]
    return d.clone()


def joint_pos_limit_proximity(env, asset_cfg=_DEFAULT_ASSET_CFG, margin: float = .15):
    """关节角进入硬限位 margin 邻域的 L1 罚(低 kp 舵机会"免费"把关节停在
    硬限位上——踝/髋滑移的来源;这个罚提前咬)。"""
    asset = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    hard = asset.data.joint_pos_limits[:, jnt_ids]   # (B, J, 2)
    soft_lo = hard[..., 0] + margin
    soft_hi = hard[..., 1] - margin
    q = asset.data.joint_pos[:, jnt_ids]
    return (torch.clamp(soft_lo - q, min=0.) + torch.clamp(q - soft_hi, min=0.)).sum(-1)
    return (torch.clamp(lo - q, min=0.) + torch.clamp(q - hi, min=0.)).sum(dim=-1)


def _spawn_tag(env):
    """出生姿态标签: +1 = 前伸, -1 = 后折(set_flamingo_state 写入)。"""
    t = getattr(env, 'spawn_tag', None)
    if t is None:
        return torch.ones(env.num_envs, device=env.device)
    return t


def pose_hold_spawn(env, asset_cfg=_DEFAULT_ASSET_CFG, std: float = .35,
                    swing_joints=('left_hip_pitch', 'left_knee', 'left_ankle'),
                    forward_deg=None, folded_deg=None):
    """奖励**保持出生时的摆动腿姿态**(前伸或后折, 由出生标签决定)。

    用户 2026-09-19: 两种姿态都要 —— 前伸(策略找到的平衡解, 可作上台阶初始动作)
    与后折(出生姿态)。按标签给不同的目标角, 而不是把所有环境都往一个姿态上拉;
    出生态与目标态一致, 所以策略要做的只是"别把它改掉"。
    """
    asset = env.scene[asset_cfg.name]
    names = _names_of(asset)
    tag = _spawn_tag(env).unsqueeze(1)          # +1 前伸 / -1 后折
    # 度 -> 弧度! 传进来的是姿态表里的"度"(5/23/-19), 而 joint_pos 是弧度;
    # 不转换的话目标是 5.0 rad vs 实际 0.087 rad, 高斯直接归零
    # —— 这就是 pose_flamingo 全程 0.00、策略压根不保持姿态的原因。
    fwd = torch.deg2rad(torch.tensor([forward_deg[n] for n in swing_joints],
                                     device=env.device, dtype=torch.float32)).unsqueeze(0)
    fold = torch.deg2rad(torch.tensor([folded_deg[n] for n in swing_joints],
                                      device=env.device, dtype=torch.float32)).unsqueeze(0)
    target = torch.where(tag > 0, fwd.expand(env.num_envs, -1),
                         fold.expand(env.num_envs, -1))
    q = asset.data.joint_pos[:, [names.index(n) for n in swing_joints]]
    return torch.exp(-((q - target) / std) ** 2).mean(dim=-1)


def swing_height_by_tag(env, asset_cfg=SceneEntityCfg('robot', site_names=['left_foot']),
                        forward_z: float = .05, folded_z: float = .15,
                        std: float = .03, gate_sensor_name=None, gate_slot: int = 1):
    """摆动脚高度目标**按出生姿态区分**: 前伸 50mm / 后折 150mm。"""
    asset = env.scene[asset_cfg.name]
    z = asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2] \
        - env.scene.terrain.env_origins[:, 2]
    z = torch.nan_to_num(z, nan=0.)
    tag = _spawn_tag(env)
    target = torch.where(tag > 0, torch.full_like(z, float(forward_z)),
                         torch.full_like(z, float(folded_z)))
    reward = torch.exp(-((z - target) / std) ** 2)
    if gate_sensor_name is not None:
        reward = reward * _foot_found(env, gate_sensor_name, gate_slot)
    return reward

def _cmd(env, idx, default=1.0):
    """读 twist 指令槽(0=flag, 1=side, 2=pose); 无命令管理器时退化到默认值。

    为什么必须走命令而不是 env 上的私有状态(我第一版就是那么错的): 指令**在 obs 里**,
    策略看得见"现在该做哪一侧/哪个姿态"; 私有状态不在 obs 里, 策略无从知道该保持什么,
    于是无论出生在哪个姿态都收敛到同一个解——实测四个 episode 末态髋角全部落在 7-21°。
    官方做法同源: 出生事件把 side "pin" 进命令项(reset 事件先于命令 reset 运行)。
    """
    mgr = getattr(env, 'command_manager', None)
    if mgr is None:
        return torch.full((env.num_envs,), float(default), device=env.device)
    cmd = mgr.get_command('twist')
    if cmd is None or cmd.shape[1] <= idx:
        return torch.full((env.num_envs,), float(default), device=env.device)
    return cmd[:, idx]


def _side(env):
    """支撑侧: +1 = 右脚支撑(左脚摆动), -1 = 左脚支撑(官方 cycle 的 side 语义)。"""
    return torch.where(_cmd(env, 1) >= 0, torch.ones(env.num_envs, device=env.device),
                       -torch.ones(env.num_envs, device=env.device))


def _pose(env):
    """姿态: +1 = 前伸, -1 = 后折。"""
    return torch.where(_cmd(env, 2) >= 0, torch.ones(env.num_envs, device=env.device),
                       -torch.ones(env.num_envs, device=env.device))


def mirror_joint_name(n):
    if n.startswith('left_'):
        return n.replace('left_', 'right_', 1)
    if n.startswith('right_'):
        return n.replace('right_', 'left_', 1)
    return n

# ── 指令驱动的奖励(官方 cycle 同构: side 决定哪只脚, pose 决定姿态) ──────────

def _swing_stance_names(env):
    """按 side 返回 (摆动腿关节名, 支撑脚 site 名, 摆动脚 site 名)。"""
    left = _side(env) > 0                      # +1 = 左脚摆动 / 右脚支撑
    return left


def com_over_foot_cmd(env, std=.03, stance_site_prefix='right_foot',
                      swing_site_prefix='left_foot'):
    """质心压**支撑脚**; 支撑脚按 side 指令切换。"""
    asset = env.scene['robot']
    left_lift = _swing_stance_names(env)       # True = 左脚摆动 -> 支撑脚是右脚
    sites = [n.split('/')[-1] for n in asset.site_names]
    com = _robot_com_xy(env, asset)
    out = torch.zeros(env.num_envs, device=env.device)
    for lift_left, sitename in ((True, 'right_foot'), (False, 'left_foot')):
        m = (left_lift == lift_left)
        if not m.any():
            continue
        fxy = asset.data.site_pos_w[:, sites.index(sitename), :2]
        d2 = torch.nan_to_num(((com - fxy) ** 2).sum(dim=-1), nan=1.)
        out = out.clone()
        out[m] = torch.exp(-d2[m] / (std ** 2))
    return out


def pose_hold_cmd(env, asset_cfg=_DEFAULT_ASSET_CFG, std=.35,
                  fwd_left=None, fold_left=None):
    """保持**指令指定的姿态**(side 决定哪条腿, pose 决定前伸/后折)。

    摆动腿关节目标: 左腿姿态直接给; 右腿姿态 = 左腿值**取反**(实测左右腿 HOME
    完全相反, 和为 0; 与官方 mirror_pose 规则一致)。
    """
    asset = env.scene[asset_cfg.name]
    names = _names_of(asset)
    left_lift = _swing_stance_names(env)
    pose_fwd = (_pose(env) > 0)
    q = asset.data.joint_pos
    out = torch.zeros(env.num_envs, device=env.device)
    for lift_left in (True, False):
        m = (left_lift == lift_left)
        if not m.any():
            continue
        base = ['left_hip_pitch', 'left_knee', 'left_ankle']
        jnames = base if lift_left else [mirror_joint_name(n) for n in base]
        idx = [names.index(n) for n in jnames]
        sgn = 1. if lift_left else -1.
        tf = torch.deg2rad(torch.tensor([fwd_left[n] * sgn for n in base],
                                        device=env.device, dtype=torch.float32))
        tl = torch.deg2rad(torch.tensor([fold_left[n] * sgn for n in base],
                                        device=env.device, dtype=torch.float32))
        target = torch.where(pose_fwd[m].unsqueeze(1), tf.unsqueeze(0), tl.unsqueeze(0))
        e = ((q[m][:, idx] - target) / std) ** 2
        out = out.clone()
        out[m] = torch.exp(-e).mean(dim=-1)      # mask 赋值(mask 选中数 != 全体,
                                                 # 用 torch.where 会尺寸不匹配)
    return out


def swing_height_cmd(env, fwd_z=.05, fold_z=.15, std=.025,
                     gate_sensor_name=None):
    """摆动脚离地高度目标: 按 pose 选高度(前伸低/后折高), 摆动脚 site 按 side 选。"""
    asset = env.scene['robot']
    sites = [n.split('/')[-1] for n in asset.site_names]
    left_lift = _swing_stance_names(env)
    pose_fwd = (_pose(env) > 0)
    z_target = torch.where(pose_fwd, torch.full((env.num_envs,), float(fwd_z),
                                                device=env.device),
                           torch.full((env.num_envs,), float(fold_z), device=env.device))
    out = torch.zeros(env.num_envs, device=env.device)
    for lift_left, sitename in ((True, 'left_foot'), (False, 'right_foot')):
        m = (left_lift == lift_left)
        if not m.any():
            continue
        z = asset.data.site_pos_w[:, sites.index(sitename), 2]             - env.scene.terrain.env_origins[:, 2]
        z = torch.nan_to_num(z, nan=0.)
        r = torch.exp(-((z - z_target) / std) ** 2)
        if gate_sensor_name is not None:
            sup_slot = 1 if lift_left else 0
            found = env.sensors[gate_sensor_name].data.found                 if hasattr(env, 'sensors') else env.scene.sensors[gate_sensor_name].data.found
            r = r * torch.clamp(torch.nan_to_num(found[:, sup_slot].float(), nan=0.), 0., 1.)
        out = out.clone()
        out[m] = r[m]                             # r 是全量算的, 取子集再散进去
    return out


def stance_side_tilt_cmd(env, threshold=.45, asset_cfg=_DEFAULT_ASSET_CFG):
    """向支撑侧倾倒的二次罚; 方向按 side 指令取反。"""
    asset = env.scene[asset_cfg.name]
    gy = torch.nan_to_num(asset.data.projected_gravity_b[:, 1], nan=0.)
    # 支撑侧: 右脚支撑 -> 向 +y(右)倒危险(direction=-1); 左脚支撑 -> 反之
    direction = torch.where(_side(env) > 0, torch.full_like(gy, -1.),
                            torch.full_like(gy, 1.))
    over = torch.clamp(direction * gy - threshold, min=0.)
    return -(over ** 2)


def stance_foot_grounded_cmd(env, sensor_name='feet_ground_contact'):
    """支撑脚着地(按 side 指令选脚)。"""
    found = env.scene.sensors[sensor_name].data.found
    flat = found.reshape(env.num_envs, 2, -1)
    lift_left = _side(env) > 0
    sup = torch.where(lift_left, flat[:, 1, :].gt(0).any(-1), flat[:, 0, :].gt(0).any(-1))
    return sup.float()


def swing_foot_touch_cmd(env, sensor_name='feet_ground_contact'):
    """摆动脚触地 = 软失败(自抵消 -> 正权重), 按 side 选脚。"""
    found = env.scene.sensors[sensor_name].data.found
    flat = found.reshape(env.num_envs, 2, -1)
    lift_left = _side(env) > 0
    touch = torch.where(lift_left, flat[:, 0, :].gt(0).any(-1), flat[:, 1, :].gt(0).any(-1))
    return -touch.float()
