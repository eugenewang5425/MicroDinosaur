"""Small ascending/descending physical steps, flat calibrated starts and pauses."""
from dataclasses import dataclass
from pathlib import Path
import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import SubTerrainCfg,TerrainOutput,TerrainGeometry,TerrainGeneratorCfg
from mjlab.managers import RewardTermCfg,TerminationTermCfg
from mjlab_microduck.tasks import mdp
from mjlab_microduck.imu_owned_head import ImuOwnedActionCfg
from transition_refine_cfg import build_config as previous_config

OUT=Path(__file__).parent/'20260914_imu_owned_terrain'


@dataclass(kw_only=True)
class StepLaneCfg(SubTerrainCfg):
    direction: str='flat'

    def profile(self,difficulty):
        h=.005+.005*difficulty
        width=.16+.06*difficulty
        origin=np.array([.8,self.size[1]/2,3*h if self.direction=='down' else 0.])
        start=origin[0]+.28+.08*difficulty
        if self.direction=='flat':return origin,[(0.,self.size[0],0.)]
        if self.direction=='up':
            parts=[(0,start,0.)]+[(start+i*width,start+(i+1)*width,(i+1)*h) for i in range(2)]
            parts += [(start+2*width,self.size[0],3*h)]
        else:
            parts=[(0,start,3*h)]+[(start+i*width,start+(i+1)*width,(2-i)*h) for i in range(2)]
            parts += [(start+2*width,self.size[0],0.)]
        return origin,parts

    def function(self,difficulty,spec,rng):
        body=spec.body('terrain');origin,parts=self.profile(difficulty);geometries=[]
        for left,right,height in parts:
            geom=body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[(right-left)/2,self.size[1]/2,(height+.08)/2],
                pos=[(left+right)/2,self.size[1]/2,(height-.08)/2])
            geometries.append(TerrainGeometry(geom=geom,color=(.32,.39,.36,1.)))
        return TerrainOutput(origin=origin,geometries=geometries)


def build_config(envs=64,seed=42):
    task,cfg=previous_config(envs,seed)
    cfg.env.actions['joint_pos']=ImuOwnedActionCfg(**vars(cfg.env.actions['joint_pos']),owned_head_indices=(1,))
    cfg.env.scene.terrain.terrain_type='generator'
    cfg.env.scene.terrain.terrain_generator=TerrainGeneratorCfg(seed=915,curriculum=True,size=(6.,2.),
        num_rows=2,num_cols=3,border_width=2.,border_height=.1,add_lights=False,
        sub_terrains={'flat':StepLaneCfg(proportion=.5,direction='flat'),
            'up':StepLaneCfg(proportion=.25,direction='up'),'down':StepLaneCfg(proportion=.25,direction='down')})
    cfg.env.scene.terrain.max_init_terrain_level=1
    cfg.env.events['calibrated_stand'].params={'yaw_range':(-.06,.06)}
    cfg.env.commands['twist']=mdp.TerrainTransitionTwistCfg(resampling_time_range=(100.,100.))
    cfg.env.rewards['body_pose_tracking']=RewardTermCfg(func=mdp.lane_body_height_tracking,weight=3.)
    cfg.env.terminations['lane_boundary']=TerminationTermCfg(func=mdp.lane_boundary,time_out=True)
    cfg.agent.experiment_name='microdinosaur_imu_owned_terrain'
    return task,cfg
