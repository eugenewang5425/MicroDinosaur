"""Paired correction-data intervention with unchanged environment and rewards."""
from mjlab_microduck.corrective_ppo import CorrectivePpoCfg
from terrain_lane_cfg import build_config as previous_config
from corrective_common import OUT,PREVIOUS


def build_config(arm,envs=64,seed=47):
    if arm not in ('control','corrective'):raise ValueError(arm)
    task,cfg=previous_config(envs,seed)
    values=vars(cfg.agent.algorithm).copy();values.pop('class_name')
    values.update(anchor_steps=4,anchor_batch_size=1536,anchor_weight=1.)
    cfg.agent.algorithm=CorrectivePpoCfg(**values,corrective_arm=arm,
        success_dataset=str(PREVIOUS/'step_demonstrations.npz'),
        stop_dataset=str(OUT/'stop_demonstrations.npz'),recovery_dataset=str(OUT/'recovery_demonstrations.npz'))
    cfg.agent.experiment_name=f'microdinosaur_corrective_{arm}'
    return task,cfg

