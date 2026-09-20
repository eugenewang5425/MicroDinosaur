"""Current v07 terrain and crouch continuations; shared hardware/actor contract."""
from dataclasses import replace
from pathlib import Path
import mujoco
from mjlab.scripts.train import TrainConfig
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.sensor import TerrainHeightSensorCfg, ObjRef, GridPatternCfg
from mjlab.terrains import (BoxFlatTerrainCfg, BoxPyramidStairsTerrainCfg,
    BoxRandomGridTerrainCfg, HfPyramidSlopedTerrainCfg)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab_microduck.tasks import mdp
from mjlab_microduck.s288_profile import apply_s288_protocol

XML = Path('D:/microduck_rl/src/mjlab_microduck/robot/microdinosaur_v07/robot_microdinosaur_v07.xml')


def terrain_generator():
    return TerrainGeneratorCfg(seed=914, curriculum=True, size=(3., 3.), num_rows=3, num_cols=4,
        border_width=2., border_height=.1, add_lights=False, sub_terrains={
        'flat': BoxFlatTerrainCfg(proportion=.30),
        'stairs': BoxPyramidStairsTerrainCfg(proportion=.20, step_height_range=(.002, .010),
            step_width=.16, platform_width=.6, border_width=.2),
        'uneven': BoxRandomGridTerrainCfg(proportion=.25, grid_width=.32,
            grid_height_range=(.002, .008), platform_width=.6),
        'slope': HfPyramidSlopedTerrainCfg(proportion=.25, slope_range=(.02, .10),
            platform_width=.6, horizontal_scale=.05, vertical_scale=.001)})


def build_config(skill, envs=64, seed=42):
    if skill not in ('terrain', 'crouch'):
        raise ValueError(skill)
    task = 'Mjlab-Velocity-Rough-MicroDinosaur' if skill == 'terrain' else 'Mjlab-Velocity-Flat-MicroDinosaur'
    cfg = TrainConfig.from_task(task)
    cfg.agent.seed = cfg.env.seed = seed
    cfg.env.scene.num_envs = envs
    cfg.env.scene.entities['robot'].spec_fn = lambda: mujoco.MjSpec.from_file(str(XML))
    for name in ('lean_drift', 'contact_timing'):
        cfg.env.rewards.pop(name, None)
    apply_s288_protocol(cfg.env)
    # Positive real-time delays stay 5..15ms when refining the physics clock.
    cfg.env.sim.mujoco.timestep = .00125
    cfg.env.decimation = 16
    robot = cfg.env.scene.entities['robot']
    robot.articulation.actuators = tuple(replace(a, delay_min_lag=4, delay_max_lag=12) for a in robot.articulation.actuators)
    sensor = TerrainHeightSensorCfg(name='body_clearance', frame=ObjRef(type='body', name='trunk_base', entity='robot'),
        pattern=GridPatternCfg(size=(0., 0.), resolution=.01), ray_alignment='world',
        max_distance=1., include_geom_groups=(0,), exclude_parent_body=True, debug_vis=False)
    cfg.env.scene.sensors = (*cfg.env.scene.sensors, sensor)
    cfg.env.rewards['body_pose_tracking'] = RewardTermCfg(func=mdp.local_body_height_tracking,
        weight=1. if skill == 'terrain' else 3., params={'std': .02 if skill == 'terrain' else .01})
    cfg.env.terminations['local_clearance'] = TerminationTermCfg(func=mdp.body_clearance_below,
        params={'minimum_height': .055}, time_out=False)
    # This skill's height command must not be silently shrunk by the inherited
    # body_pose_range curriculum (the old config limited it to +/-5 mm).
    cfg.env.curriculum.pop('body_pose_range', None)
    if skill == 'terrain':
        cfg.env.scene.terrain.terrain_generator = terrain_generator()
        cfg.env.scene.terrain.max_init_terrain_level = 1
        cfg.env.commands['body_pose'].ranges = ((0., 0.),)*6
        # Keep random head gestures small during ground-skill acquisition;
        # deployed head stabilization is evaluated separately with every policy.
        cfg.env.curriculum.pop('head_pose_range', None)
        cfg.env.commands['head_pose'].ranges = ((-.015, .015),)*3+((-.01, .01),)
        cfg.env.commands['twist'].ranges.lin_vel_x = (-.25, .55)
        cfg.env.commands['twist'].ranges.lin_vel_y = (-.1, .1)
        cfg.env.commands['twist'].ranges.ang_vel_z = (-.6, .6)
    else:
        cfg.env.commands['body_pose'] = mdp.SmoothCrouchCommandCfg(
            ranges=((0., 0.),)*6, resampling_time_range=(2., 4.), height_buckets=(0., -.01, -.02))
        cfg.env.curriculum.pop('standing_envs', None)
        cfg.env.commands['twist'].rel_standing_envs = .65
        cfg.env.commands['twist'].rel_turn_in_place_envs = .05
        cfg.env.commands['twist'].ranges.lin_vel_x = (-.1, .3)
        cfg.env.commands['twist'].ranges.lin_vel_y = (-.03, .03)
        cfg.env.commands['twist'].ranges.ang_vel_z = (-.35, .35)
        cfg.env.curriculum.pop('head_pose_range', None)
        cfg.env.commands['head_pose'].ranges = ((-.015, .015),)*3+((-.01, .01),)
    cfg.agent.logger = 'tensorboard'; cfg.agent.upload_model = False
    cfg.agent.experiment_name = 'microdinosaur_terrain_skills'
    return task, cfg
