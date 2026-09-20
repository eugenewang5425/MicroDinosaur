"""Prepared gated next stage: small uneven cells with flat/step retention."""
from dataclasses import dataclass
import numpy as np
import mujoco
from mjlab.terrains.terrain_generator import SubTerrainCfg,TerrainOutput,TerrainGeometry,TerrainGeneratorCfg
from terrain_lane_cfg import StepLaneCfg
from corrective_cfg import build_config as corrective_config


@dataclass(kw_only=True)
class SmallRoughLaneCfg(SubTerrainCfg):
    def profile(self,difficulty,rng):
        # A calibrated flat entrance is retained. Cell heights vary along x
        # and across the feet, with no hidden plane closing the low cells.
        origin=np.array([.8,self.size[1]/2,0.]);start=1.10
        amplitude=.002+.004*float(difficulty)
        cells=[(0.,start,0.,self.size[1],0.)]
        nx=int(np.ceil((self.size[0]-start)/.20));ny=int(np.ceil(self.size[1]/.20))
        heights=rng.uniform(0,amplitude,(nx,ny))
        for ix in range(nx):
            for iy in range(ny):
                cells.append((start+ix*.20,min(start+(ix+1)*.20,self.size[0]),
                    iy*.20,min((iy+1)*.20,self.size[1]),float(heights[ix,iy])))
        return origin,cells

    def function(self,difficulty,spec,rng):
        origin,cells=self.profile(difficulty,rng);body=spec.body('terrain');parts=[]
        for left,right,near,far,height in cells:
            geom=body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[(right-left)/2,(far-near)/2,(height+.08)/2],
                pos=[(left+right)/2,(near+far)/2,(height-.08)/2])
            parts.append(TerrainGeometry(geom=geom,color=(.32,.39,.36,1.)))
        return TerrainOutput(origin=origin,geometries=parts)


def build_config(envs=64,seed=47):
    task,cfg=corrective_config('corrective',envs,seed)
    cfg.env.scene.terrain.terrain_generator=TerrainGeneratorCfg(seed=916,curriculum=True,size=(6.,2.),
        num_rows=3,num_cols=4,border_width=2.,border_height=.1,add_lights=False,
        sub_terrains={'flat':StepLaneCfg(proportion=.50,direction='flat'),
            'up':StepLaneCfg(proportion=.20,direction='up'),'down':StepLaneCfg(proportion=.10,direction='down'),
            'uneven':SmallRoughLaneCfg(proportion=.20)})
    cfg.env.scene.terrain.max_init_terrain_level=0
    cfg.agent.experiment_name='microdinosaur_corrective_rough'
    return task,cfg

