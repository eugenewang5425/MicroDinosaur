"""上高台阶模组配置(hinge 抬脚奖励 + 人工分阶段加难)。

根因链(2026-09-17/18 实测):
  ① 奖励把**抬脚目标**钉在 0.02m 且是**双向**偏离惩罚 delta=|h-target|。
     台阶 20mm 时目标正好等于台阶高,余量为零;实测策略只抬 12-15mm
     (过 10mm 台阶够 11.4mm,15mm 就卡在台阶口反复震荡)。
  ② 原课程 difficulty_range=(0,1) 使最高难度只有 10mm/级,结构上到不了高台阶。
  ③ 第一次修法(把 target 直接提到 0.04 + 跳到 15mm 难度)失败:偏离反而从
     5-8mm 涨到 25-28mm,梯度混乱,200 迭代后连 10mm 都走不动(3.0m->0.51m,
     不摔只是不走)。教训:双向惩罚下加大 target 是反向激励。
  ④ terrain generator 的课程在 512 env 下三次早期崩 Non-finite,256 env 稳定。

本配置:
  - 抬脚奖励换成 mdp.feet_clearance_shortfall(**单向 hinge**,抬够零成本);
  - target = 2.2 × 每级高(留一倍余量),随难度自动缩放;
  - 难度固定不升级(避开生成器不稳定面),人工分阶段:10mm -> 12.5 -> 15 -> 20mm;
  - 建议 256 env。
"""
from mjlab.managers import RewardTermCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab_microduck.tasks import mdp
from corrective_rough_cfg import build_config as rough_config
from terrain_lane_cfg import StepLaneCfg


TARGET_RATIO = 1.8              # 抬脚目标 = 1.8 × 每级台阶高(够跨过且不过度)
DEFAULT_DIFFICULTY = 1.0        # h = 10mm/级,总高 30mm(v2 基线 2/2 通过)


def build_config(envs=256, seed=47, difficulty=DEFAULT_DIFFICULTY):
    task, cfg = rough_config(envs, seed)
    h_mm = 5. + 5. * float(difficulty)
    target = TARGET_RATIO * h_mm / 1000.
    # 保持 mjlab 内置的**双向**偏离惩罚,只把目标换成随台阶高缩放的值。
    # 实测依据:(a) 目标固定 0.02 而台阶长到 20mm 时余量为零 -> 卡在台阶口;
    # (b) 改成单向 hinge 后策略学会深折叠抬腿(15mm 地形抬到 54-99mm),但
    #     无上限约束 -> 抬过头失衡摔倒(2/2 跌倒)。所以正确形状是双向 + 缩放目标。
    cfg.env.rewards['foot_clearance'].params = dict(
        cfg.env.rewards['foot_clearance'].params, target_height=target)
    sw = cfg.env.rewards.get('foot_swing_height')
    if sw is not None:
        sw.params = dict(sw.params, target_height=target)
    tg = cfg.env.scene.terrain.terrain_generator
    tg.curriculum = False
    tg.difficulty_range = (float(difficulty), float(difficulty))
    tg.num_rows, tg.num_cols = 2, 3
    tg.sub_terrains = {
        'flat': StepLaneCfg(proportion=.30, direction='flat'),
        'up': StepLaneCfg(proportion=.70, direction='up'),
    }
    cfg.env.scene.terrain.max_init_terrain_level = 0
    cfg.agent.experiment_name = 'microdinosaur_stair_climb'
    print(f'[stair] 难度 {difficulty} -> 每级 {h_mm:.1f}mm 总高 {h_mm * 3:.0f}mm | '
          f'抬脚目标 {target * 1000:.1f}mm(双向,随台阶高缩放) | envs={envs} '
          f'| curriculum off', flush=True)
    return task, cfg
