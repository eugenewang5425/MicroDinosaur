"""Flamingo(单脚站立)任务配置——官方 flamingo 分支配方移植到我们的 19 关节契约。

官方做法与我们的失败对照(为什么这次不一样):
  我们四轮失败,全是"从站立姿势出发试图抬腿"——但官方是 **直接在单脚姿态里
  出生**(set_flamingo_state,带 ±3° 倾斜与 0.05rad 关节噪声),任务只是"保持"。
  官方还明确:躯干侧倾 ~24° **就是姿态本身**,普通直立奖励会和姿态打架 ——
  要奖励姿态自己的重力向量(我们四轮用的 trunk_still/upright 全在跟姿态打架)。
  失败不对称:向摆动腿侧倒=软失败(脚落地,便宜);向支撑腿侧倒=硬失败(重罚)。

姿态数值(官方 FLAMINGO_POSE,右脚支撑/左脚摆动;rad):
  左腿(摆动,向前抬起): hip_yaw 0, hip_roll +0.30, hip_pitch +1.20(+69°),
                          knee -0.80(-46°), ankle +0.80
  颈/头(嘴转向支撑侧): neck_pitch +0.35, head_pitch +0.35, head_yaw -1.50, head_roll 0
  右腿(支撑):          hip_yaw +0.386, hip_roll -0.334, hip_pitch +0.258,
                          knee +0.005, ankle -0.253
  躯干: roll +22.6°(向支撑侧), pitch -8.9°, z≈0.120m
  重力目标(projected gravity b 系): (-0.154, -0.379, -0.912)
"""
import math
import dataclasses
from dataclasses import dataclass

import torch

from mjlab.managers import RewardTermCfg, TerminationTermCfg, EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
import flamingo_mdp as fmdp
from mjlab_microduck.tasks import mdp
from corrective_cfg import build_config as corrective_config
from terrain_lane_cfg import StepLaneCfg

# 右脚支撑 / 左脚摆动(与官方同侧)。镜像版(左支撑)后续按需加。
SWING = 'left'
STANCE = 'right'
SWING_SLOT, STANCE_SLOT = 0, 1

# ── 两个姿态(用户 2026-09-19 命名) ───────────────────────────────────────
# **前伸**: 策略自己找到的平衡解(髋+5°/膝+23°/踝-19°,腿伸直向前), 实测摆动脚~50mm。
#          用户指出它可作**上台阶的初始动作** —— 所以必须保留, 不能被后折覆盖。
# **后折**: 出生姿态(髋+69°/膝-46°/踝+46°,屈膝折叠), 摆动脚~170mm。
# 指令: twist 的 vy 维 —— +1 = 前伸, -1 = 后折(与单脚站立的"抬哪只脚"复用同一维,
#       该任务里 vx/wz 恒 0)。
POSE_FORWARD_DEG = {'left_hip_pitch': 5., 'left_knee': 23., 'left_ankle': -19.}
POSE_FOLDED_DEG = {'left_hip_pitch': 69., 'left_knee': -46., 'left_ankle': 46.}
SWING_Z_FORWARD = .050     # 前伸的摆动脚高度(实测稳定值)
SWING_Z_FOLDED = .150      # 后折的目标(向姿态的 170mm 靠, 取 150 留余量)

FLAMINGO_POSE_BY_NAME = {
    'left_hip_yaw': 0.000, 'left_hip_roll': 0.300, 'left_hip_pitch': 1.200,
    'left_knee': -0.800, 'left_ankle': 0.800,
    'neck_pitch': 0.349, 'head_pitch': 0.350, 'head_yaw': -1.500, 'head_roll': 0.000,
    'right_hip_yaw': 0.386, 'right_hip_roll': -0.334, 'right_hip_pitch': 0.258,
    'right_knee': 0.005, 'right_ankle': -0.253,
}
# ⚠ 单位约定(踩过坑): 下面这两个表用**度**, 而 FLAMINGO_POSE_BY_NAME 用的是**弧度**
# (1.2 rad = 69°)。混用会导致姿态奖励恒为 0——实测 pose_flamingo 全程 0.00,
# 策略因此完全不保持姿态。新增项一律用度并在此注明。
BASE_ROLL = math.radians(22.6)
BASE_PITCH = math.radians(-8.9)
FLAMINGO_GRAVITY_B = (-0.154, -0.379, -0.912)
SWING_TARGET_Z = 0.05
STANCE_TILT_THRESHOLD = 0.45
EPISODE_S = 6.0


@dataclass(kw_only=True)
class FlamingoCommandCfg(mdp.TerrainTransitionTwistCfg):
    """twist 三槽位承载 [flag, side, pose] —— 官方 cycle 的同构做法。

    vx = flag  : 1 = 单脚站立(本任务恒 1)
    vy = side  : +1 = 右脚支撑(左脚摆动), -1 = 左脚支撑
    wz = pose  : +1 = 前伸, -1 = 后折
    """

    def build(self, env):
        return FlamingoCommand(self, env)


class FlamingoCommand(mdp.TerrainTransitionTwist):
    def _resample_command(self, env_ids):
        # 不自己随机: side/pose 由**出生事件**决定(见 flamingo_mdp.set_flamingo_state),
        # 这里只负责把 env 上的值搬进命令。命令 reset 在事件之后运行, 所以必须
        # 在这里搬运, 而不是在事件里直接写命令项。
        super()._resample_command(env_ids)
        self.mode[env_ids] = 0
        self._sync_from_env(env_ids)

    def _sync_from_env(self, env_ids=None):
        side = getattr(self._env, 'spawn_side', None)
        pose = getattr(self._env, 'spawn_pose', None)
        if side is None:
            return
        ids = slice(None) if env_ids is None else env_ids.long()
        self.request[ids, 0] = 1.
        self.request[ids, 1] = side[ids]
        self.request[ids, 2] = pose[ids]
        self._command[ids, 0] = 1.
        self._command[ids, 1] = side[ids]
        self._command[ids, 2] = pose[ids]

    def _update_command(self):
        self.elapsed += self._env.step_dt
        self._sync_from_env()                  # 每 tick 保持与 env 一致


def _mirror_name(n):
    """左<->右互换; 中轴关节(颈/头)名字不变。"""
    if n.startswith('left_'):
        return n.replace('left_', 'right_', 1)
    if n.startswith('right_'):
        return n.replace('right_', 'left_', 1)
    return n


def _mirror_pose(pose_by_name):
    """镜像一个姿态表。

    规则(实测 preferred_plant): 左右腿的 HOME 完全相反(和为 0), 所以镜像 =
    **名字左右互换 + 数值取反**。中轴关节: 俯仰类(neck_pitch/head_pitch)不变,
    偏航/滚转类(head_yaw/head_roll)取反。
    """
    out = {}
    for n, v in pose_by_name.items():
        if n.startswith(('left_', 'right_')):
            out[_mirror_name(n)] = -v
        elif n in ('head_yaw', 'head_roll', 'neck_yaw', 'neck_roll'):
            out[_mirror_name(n)] = -v
        else:
            out[_mirror_name(n)] = v
    return out


def build_config(envs=512, seed=47, standing_prob=0.0, swing='left',
                 fix_side=0.0, fix_pose=0.0, push=0.0,
                 tilt_threshold=.40, tilt_weight=4.):
    # ── 左右镜像(用户任务链第②步: 另一只脚复刻) ───────────────────────────
    # 机器人左右腿 HOME 完全相反, 所以镜像 = 名字左右互换 + 数值取反;
    # 躯干侧倾/重力目标 y/倾斜罚方向 同样取反。swing='right' 即右脚摆动、左脚支撑。
    mir = (swing == 'right')
    pose_by_name = _mirror_pose(FLAMINGO_POSE_BY_NAME) if mir else dict(FLAMINGO_POSE_BY_NAME)
    pose_fwd = _mirror_pose(POSE_FORWARD_DEG) if mir else dict(POSE_FORWARD_DEG)
    pose_fold = _mirror_pose(POSE_FOLDED_DEG) if mir else dict(POSE_FOLDED_DEG)
    base_roll = -BASE_ROLL if mir else BASE_ROLL
    swing_name = swing
    stance_name = 'right' if swing == 'left' else 'left'
    swing_slot, stance_slot = (0, 1) if swing == 'left' else (1, 0)

    task, cfg = corrective_config('corrective', envs, seed)
    # 平地(单块 flat,不走课程——generator 课程已证不稳)
    tg = cfg.env.scene.terrain.terrain_generator
    tg.curriculum = False
    tg.difficulty_range = (0., 0.)
    tg.num_rows, tg.num_cols = 2, 3
    tg.sub_terrains = {'flat': StepLaneCfg(proportion=1.0, direction='flat')}
    cfg.env.scene.terrain.max_init_terrain_level = 0
    cfg.env.episode_length_s = EPISODE_S

    stance_site = SceneEntityCfg('robot', site_names=[f'{stance_name}_foot'])
    swing_site = SceneEntityCfg('robot', site_names=[f'{swing_name}_foot'])

    r = cfg.env.rewards
    # 移除与姿态打架/无关的项(官方同款移除清单)
    for k in ('track_linear_velocity', 'track_angular_velocity', 'air_time',
              'foot_clearance', 'foot_swing_height', 'foot_slip', 'pose',
              'upright', 'dof_pos_limits', 'head_pose_tracking',
              'body_pose_tracking', 'head_pose_bias', 'stand_still',
              'stand_slip', 'stand_jitter', 'lift_side_unload',
              'single_leg_lift', 'support_foot_grounded', 'lifted_foot_airborne',
              'single_leg_hold', 'trunk_still', 'stay_near_origin'):
        r.pop(k, None)

    # 官方奖励集(权重照抄,符号照抄)
    # 四项全部**指令驱动**: side 决定哪只脚当支撑/摆动, pose 决定前伸/后折
    r['com_over_stance_foot'] = RewardTermCfg(func=fmdp.com_over_foot_cmd, weight=3.,
        params=dict(std=.03))
    r['stance_foot_grounded'] = RewardTermCfg(func=fmdp.stance_foot_grounded_cmd,
        weight=1., params=dict(sensor_name='feet_ground_contact'))
    # w 1.5->3.0:实测 v1(400迭代)摆动腿下沉到均 14-39mm、最低 5.7mm 贴地飞——
    # 重力持续把摆动腿往下拉,1.5 的权重扛不住,策略选"悬停贴地"偷分
    # (foot_contact_penalty 只罚触地,5.7mm 悬停刚好躲过)。
    # 摆动脚高度按姿态区分: 前伸 50mm / 后折 150mm
    r['swing_foot_clear'] = RewardTermCfg(func=fmdp.swing_height_cmd,
        weight=3.0, params=dict(fwd_z=SWING_Z_FORWARD, fold_z=SWING_Z_FOLDED,
                                std=.025, gate_sensor_name='feet_ground_contact'))
    r['swing_foot_touch'] = RewardTermCfg(func=fmdp.swing_foot_touch_cmd,
        weight=.5, params=dict(sensor_name='feet_ground_contact'))
    # std 0.5 -> 0.35:v1 的宽松容差让腿姿漂移(摆动膝角偏差 ~30° 仍有分)
    # 双姿态(用户命名): 前伸=策略平衡解(上台阶初始动作) / 后折=出生姿态。
    # 目标**跟随出生标签**, 不再把所有环境往单一后折姿态拉。
    r['pose_flamingo'] = RewardTermCfg(func=fmdp.pose_hold_cmd, weight=3.,
        params=dict(std=.35, fwd_left=POSE_FORWARD_DEG, fold_left=POSE_FOLDED_DEG))
    # 重力目标也**指令驱动**(side 决定侧倾方向), 不再按 swing= 静态镜像
    r['gravity_flamingo'] = RewardTermCfg(func=fmdp.projected_gravity_match_cmd,
        weight=2., params=dict(std=.15))
    r['stillness'] = RewardTermCfg(func=fmdp.joint_vel_gaussian_cmd, weight=1.,
        params=dict(std=2., sensor_name='feet_ground_contact'))
    # threshold 随推力级收紧(参数化): 0.25 级实测左脚支撑组倾角峰 69.7° 贴线,
    # 0.4 级训练用 0.36/5.0 再拉余量
    r['stance_side_tilt'] = RewardTermCfg(func=fmdp.stance_side_tilt_cmd,
        weight=float(tilt_weight), params=dict(threshold=float(tilt_threshold)))
    r['joint_limit_proximity'] = RewardTermCfg(
        func=fmdp.joint_pos_limit_proximity, weight=-1.,
        params=dict(asset_cfg=SceneEntityCfg('robot', joint_names=tuple(
            n for n in ('left_hip_pitch', 'left_knee', 'left_ankle',
                        'right_hip_pitch', 'right_knee', 'right_ankle',
                        'neck_pitch', 'head_pitch', 'head_yaw', 'tail_pitch'))),
            margin=.10))
    # 平滑正则(官方:运动阻断项保持轻——平衡需要动作)
    cfg.env.rewards['action_rate_l2'].weight = -.1

    # ── 用户 2026-09-19 指示:放开尾巴/手臂的探索, 先零干扰站稳 ──────────────
    # 实测(v2, 八角度截图): 头偏航 -68~-94°(姿态要求 -86°,配平到位),
    # 但**尾巴只用了很小幅度**(尾偏航 6-14°、尾俯仰 6-19°,能力是 ±40°/-52~+100°),
    # 手臂同样。原因: tail_pose_tracking w=+1.0 / arm_pose_tracking w=+1.5
    # 把它们锁在 HOME 姿态上——**配重候选被奖励钉死了**。
    r['tail_pose_tracking'] = RewardTermCfg(func=r['tail_pose_tracking'].func,
        weight=.1, params=dict(r['tail_pose_tracking'].params))
    r['arm_pose_tracking'] = RewardTermCfg(func=r['arm_pose_tracking'].func,
        weight=.3, params=dict(r['arm_pose_tracking'].params))
    # 零干扰优先(用户:"先在零干扰下能站2秒钟再谈抗干扰"): push=0 时推力幅度归零,
    # 事件结构保留; 抗扰阶段用 push>0 爬坡。
    # y(侧向)是主导失败方向(单脚站立的支撑线窄), 所以 y 幅度给满、x 给 0.6 倍;
    # interval 从 3-6s 收紧到 1-3s —— 6s episode 里 3-6s 只会撞上 0-1 次推力,
    # 训练量不够。
    if push > 0:
        cfg.env.events['push_robot'].params['velocity_range'] = {
            'x': (-push * .6, push * .6), 'y': (-push, push)}
        cfg.env.events['push_robot'].interval_range_s = (1.0, 3.0)
    else:
        cfg.env.events['push_robot'].params['velocity_range'] = {'x': (0., 0.),
                                                                 'y': (0., 0.)}

    # 出生事件:直接在单脚姿态里出生(替换 calibrated_stand 的站立出生)
    cfg.env.events.pop('calibrated_stand', None)
    cfg.env.events['set_flamingo_state'] = EventTermCfg(
        func=fmdp.set_flamingo_state, mode='reset',
        params=dict(joint_pose_by_name=dict(FLAMINGO_POSE_BY_NAME),
                    joint_pose_by_name_mirror=_mirror_pose(FLAMINGO_POSE_BY_NAME),
                    base_roll=abs(BASE_ROLL), base_pitch=BASE_PITCH,
                    z_min=.115, z_max=.125, tilt_noise=math.radians(3.),
                    joint_noise_std=.05, standing_prob=standing_prob,
                    standing_z_min=.11, standing_z_max=.12,
                    side_fixed=float(fix_side), pose_fixed=float(fix_pose)))

    # 指令项: twist 三槽位 [flag=1, side=±1, pose=±1] —— 官方 cycle 的同构做法。
    # 单个策略 + 命令: side/pose 由出生事件钉进去(见 mdp.set_flamingo_state),
    # 策略从 obs 读到"该做哪一侧/哪个姿态"; 不装这个的话奖励按指令切脚,
    # 命令却恒 0, 两侧都退化成同一侧。
    _tw = cfg.env.commands['twist']
    cfg.env.commands['twist'] = FlamingoCommandCfg(
        **{f.name: getattr(_tw, f.name) for f in dataclasses.fields(_tw)})

    # 终止:保留基线 fell_over(70°;姿态自身 24°)
    cfg.agent.experiment_name = 'microdinosaur_flamingo'
    # side/pose 现在由**出生事件随机**、经命令项进入 obs(官方 cycle 同构),
    # 不再靠 swing= 这个静态参数; 这行打印保留仅为对齐旧的日志习惯。
    print(f'[flamingo] 指令驱动: 出生随机 [side,pose] 各半 (2 支撑侧 × 2 姿态) | '
          f'躯干侧倾 ±{math.degrees(BASE_ROLL):.1f}° | episode {EPISODE_S}s | '
          f'envs={envs} standing_prob={standing_prob} | 旧 swing={swing_name}', flush=True)
    return task, cfg
