"""Three matched arms: original retention, archive only, archive plus new demos."""
from mjlab_microduck.demonstration_ppo import DemonstrationPpoCfg
from terrain_lane_cfg import build_config as previous_config
from demonstration_expert import OUT


def build_config(arm,envs=64,seed=43):
    if arm not in ('legacy','archive','demonstration'):raise ValueError(arm)
    task,cfg=previous_config(envs,seed)
    if arm!='legacy':
        values=vars(cfg.agent.algorithm).copy();values.pop('class_name')
        cfg.agent.algorithm=DemonstrationPpoCfg(**values,
            demonstration_dataset=str(OUT/'step_demonstrations.npz') if arm=='demonstration' else '')
    cfg.agent.experiment_name=f'microdinosaur_demo_{arm}'
    return task,cfg
