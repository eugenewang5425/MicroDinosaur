"""v2 纠正轮装配:数据换 v2 库,配额换 CorrectivePPOv2,其余与 corrective_cfg 相同。"""
from mjlab_microduck.corrective_ppo import CorrectivePpoV2Cfg
from terrain_lane_cfg import build_config as previous_config
from corrective_common import OUT, PREVIOUS


def build_config(arm, envs=64, seed=47):
    if arm != 'corrective_v2':
        raise ValueError(arm)
    task, cfg = previous_config(envs, seed)
    values = vars(cfg.agent.algorithm).copy()
    values.pop('class_name')
    values.update(anchor_steps=4, anchor_batch_size=1536, anchor_weight=1.)
    cfg.agent.algorithm = CorrectivePpoV2Cfg(**values, corrective_arm='corrective',
        success_dataset=str(PREVIOUS / 'step_demonstrations.npz'),
        stop_dataset=str(OUT / 'stop_demonstrations_v2.npz'),
        recovery_dataset=str(OUT / 'recovery_demonstrations_v2.npz'))
    cfg.agent.experiment_name = 'microdinosaur_corrective_v2'
    return task, cfg
