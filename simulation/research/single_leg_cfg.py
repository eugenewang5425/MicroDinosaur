"""单脚站立任务配置(用户 2026-09-18 指定)。

目标:一条腿抬到指定高度并保持,另一条腿支撑,身体平衡。
平衡手段(用户指定):头部与尾巴同时朝抬起的脚的方向转动当配重。

与走路/台阶任务的区别:
  - 平地(terrain_type='plane',不走地形生成器 -> 顺带避开 generator 的偶发崩溃);
  - twist 命令恒 0(原地),去掉走路的抬脚项(foot_clearance/foot_swing_height/
    air_time——它们的目标是"走路时脚抬高 20mm",与"单脚抬起保持"冲突);
  - 新增:抬脚高度跟踪 + 支撑脚着地 + 抬起脚离地 + 躯干钉住。
不硬编码"头转多少度":先只给平衡与抬脚目标,让策略自己发现头尾是最快的配重;
验收时专门检查头尾是否参与配平。

分阶段(用户:先平衡,再练抬脚高度):
  target_height 从 25mm 起 -> 35 -> 45mm,每阶段过验收再升。
"""
from dataclasses import dataclass

import torch

from mjlab.managers import RewardTermCfg
from mjlab_microduck.tasks import mdp
from corrective_cfg import build_config as corrective_config
from terrain_lane_cfg import StepLaneCfg
import single_leg_mdp as sl

DEFAULT_TARGET_M = .025
LIFT_FOOT = 'left'


@dataclass(kw_only=True)
class StillTwistCfg(mdp.TerrainTransitionTwistCfg):
    lift_foot: str = 'random'      # random | left | right

    def build(self, env):
        return StillTwist(self, env)


class StillTwist(mdp.TerrainTransitionTwist):
    """原地 + **用 vy 维承载"抬哪只脚"的指令**(+1 抬左脚、-1 抬右脚)。

    vx/wz 恒 0(单脚站立不需要位移);vy 复用为抬脚侧指令,每回合开始时按
    lift_foot 重采(随机/固定左/固定右)。左右脚都能训,且策略看得见"该抬哪只"
    —— **不动 81 维契约**(twist 本就占 obs[63:66],原本三维都当速度用)。
    """

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)   # 平地下 terrain_types 全 0,不改东西
        self.mode[env_ids] = 0
        self.request[env_ids] = 0.
        n = len(env_ids)
        mode = getattr(self.cfg, 'lift_foot', 'random')
        if mode == 'random':
            sign = torch.where(torch.rand(n, device=self.device) < .5,
                               torch.ones(n, device=self.device),
                               -torch.ones(n, device=self.device))
        else:
            sign = torch.full((n,), 1. if mode == 'left' else -1.,
                              device=self.device)
        self.request[env_ids, 1] = sign

    def _update_command(self):
        self.elapsed += self._env.step_dt
        self._command[:, 0] = 0.
        self._command[:, 2] = 0.
        self._command[:, 1] = self.request[:, 1]   # 抬脚指令直接生效(离散信号)

def build_config(envs=512, seed=47, target_height=DEFAULT_TARGET_M,
                 lift_foot=LIFT_FOOT):
    task, cfg = corrective_config('corrective', envs, seed)
    # 平地:仍走 generator 但只放一块 flat(terrain_type='plane' 会让
    # lane_body_height_tracking / TerrainTransitionTwist 读不到 terrain_types 而报错)
    tg = cfg.env.scene.terrain.terrain_generator
    tg.curriculum = False
    tg.difficulty_range = (0., 0.)
    tg.num_rows, tg.num_cols = 2, 3
    tg.sub_terrains = {'flat': StepLaneCfg(proportion=1.0, direction='flat')}
    cfg.env.scene.terrain.max_init_terrain_level = 0
    cfg.env.commands['twist'] = StillTwistCfg(resampling_time_range=(100., 100.),
                                              lift_foot=lift_foot)

    r = cfg.env.rewards
    for k in ('foot_clearance', 'foot_swing_height', 'air_time',
              'track_linear_velocity', 'track_angular_velocity'):
        r.pop(k, None)
    # 线性抬脚奖励(权重 8):不动=0 分,抬到目标=满分
    # 三项都从 twist 指令读"抬哪只脚",不再硬编码左右(与指令同源,不会搞反)
    r['single_leg_lift'] = RewardTermCfg(func=sl.single_leg_lift, weight=8.,
        params=dict(target_height=target_height))
    r['support_foot_grounded'] = RewardTermCfg(func=sl.support_foot_grounded,
        weight=2.)
    r['lifted_foot_airborne'] = RewardTermCfg(func=sl.lifted_foot_airborne,
        weight=.5)
    # 连续保持时长:治"抬起-落下"交替(实测保持率只有 50%)。
    # floor_ratio=.4 而非 .7:G 轮用 .7(=7mm)时策略只抬到 6.5mm,阈值从未触发,
    # 该项全程 0 分 -> 抬脚反而从 9.5mm 退到 6.5mm。阈值必须低于实际能力。
    r['single_leg_hold'] = RewardTermCfg(func=sl.single_leg_hold_duration,
        weight=5., params=dict(target_height=target_height, floor_ratio=.4, cap_s=3.))
    # trunk_still 3.0 -> 1.5:它奖励"不动",权重过高会盖过"抬脚"(实测退化)
    # 重心转移(抬脚前置步骤):让策略先学会"把体重压到支撑腿上",
    # 而不是一步跨到"脚抬 10mm"。实测四轮都是"从未让脚离地",缺的就是这一步。
    r['lift_side_unload'] = RewardTermCfg(func=sl.lift_side_unload, weight=4.,
        params=dict(margin=.3))
    # 方案 A(用户 2026-09-18 拍板):脚底接触带仅 6.1mm 宽,静态单脚站立物理上
    # 不可行(容差 ±3mm)。改为允许"挪步式动态平衡",只禁止打转 + 限制别跑远。
    r['trunk_still'] = RewardTermCfg(func=sl.trunk_still, weight=2.,
        params=dict(omega_scale=.6))
    r['stay_near_origin'] = RewardTermCfg(func=sl.stay_near_origin, weight=-3.,
        params=dict(radius=.8))
    cfg.agent.experiment_name = 'microdinosaur_single_leg'
    print(f'[single-leg] 抬脚指令={lift_foot}(twist vy 维 ±1) 目标 '
          f'{target_height * 1000:.0f}mm | {envs} env | 奖励 {len(r)} 项', flush=True)
    return task, cfg
