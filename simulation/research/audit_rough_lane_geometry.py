"""Ray-check prepared small uneven cells; no policy or GPU training here."""
import json
import numpy as np
import mujoco
from corrective_rough_cfg import SmallRoughLaneCfg,build_config
from corrective_common import OUT


def main():
    results=[]
    for difficulty in (0.,.5,1.):
        cfg=SmallRoughLaneCfg(size=(6.,2.));rng=np.random.default_rng(916)
        origin,cells=cfg.profile(difficulty,rng)
        spec=mujoco.MjSpec();spec.worldbody.add_body(name='terrain')
        cfg.function(difficulty,spec,np.random.default_rng(916));model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        errors=[]
        for left,right,near,far,height in cells:
            point=np.array([(left+right)/2,(near+far)/2,1.]);gid=np.array([-1],np.int32)
            distance=mujoco.mj_ray(model,data,point,np.array([0.,0.,-1.]),None,1,-1,gid)
            assert gid[0]>=0;errors.append(abs((1-distance)-height))
        assert max(errors)<1e-8
        heights=np.asarray([r[-1] for r in cells[1:]]).reshape(-1,10)
        assert np.ptp(heights,axis=0).min()>0 and np.ptp(heights,axis=1).min()>0
        results.append(dict(difficulty=difficulty,cells=len(cells),max_ray_error_m=max(errors),max_height_mm=float(heights.max()*1000),flat_start_m=1.10-origin[0]))
    _,cfg=build_config(8);sub=cfg.env.scene.terrain.terrain_generator.sub_terrains
    assert list(sub)==['flat','up','down','uneven']
    proportions={k:v.proportion for k,v in sub.items()};assert abs(sum(proportions.values())-1)<1e-8
    result=dict(status='PASS',training_launched=False,role='prepared conditional curriculum, geometry only',checks=results,proportions=proportions)
    (OUT/'rough_curriculum_geometry.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
