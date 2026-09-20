"""Small fixed trajectory grid for a possible low-pose recovery curriculum."""
import json
import numpy as np
from fold_recovery_probe import Probe, OUT
from imu_heading import from_rpy


def main():
    rows=json.loads((OUT/'fold_geometry.json').read_text());fold=np.array(rows[4]['target'])
    results=[]
    for pose,quat in [('prone',from_rpy(0,np.pi/2)),('supine',from_rpy(0,-np.pi/2))]:
        for hip in (-.35,0,.35):
            for tail in (-.5,.8,1.5):
                e=Probe();target=fold.copy();target[e.names.index('left_hip_pitch')]+=hip
                target[e.names.index('right_hip_pitch')]-=hip
                target[e.names.index('tail_pitch')]=tail
                middle=e.home.copy();middle[e.names.index('tail_pitch')]=tail
                label=f'bridge_{pose}_hip{hip:+.2f}_tail{tail:+.1f}'
                result=e.dynamics(label,e.home,quat,[(0,e.home),(1,e.home),(3,target),(4,target),(6,middle),(8,e.home),(12,e.home)],seconds=12)
                result['search_parameters']=dict(hip_offset_rad=hip,tail_pitch_rad=tail)
                results.append(result)
    (OUT/'bridge_search.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('raw successes',sum(r['raw_recovery_criterion'] for r in results),'/',len(results),flush=True)


if __name__=='__main__':main()
