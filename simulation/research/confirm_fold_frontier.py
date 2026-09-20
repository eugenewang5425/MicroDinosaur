"""Confirm recommended depth at positive delays and map near-upright starts."""
import json
import numpy as np
from fold_recovery_probe import Probe,OUT
from imu_heading import from_rpy


def main():
    geometry=json.loads((OUT/'fold_geometry.json').read_text());folds=[];frontier=[]
    for depth in (30,40):
        target=np.array(next(r for r in geometry if r['depth_mm']==depth)['target'])
        for lag in (5,10,15):
            e=Probe(command_ms=lag)
            folds.append(e.dynamics(f'confirm_fold_{depth}mm',e.home,from_rpy(0,0),
                [(0,e.home),(1,e.home),(3,target),(5,target),(7,e.home),(10,e.home)],seed=1))
    for axis in ('pitch','roll'):
        for angle in (-20,-10,10,20):
            e=Probe();q=from_rpy(np.deg2rad(angle) if axis=='roll' else 0,
                np.deg2rad(angle) if axis=='pitch' else 0)
            frontier.append(e.dynamics(f'frontier_{axis}{angle:+d}',e.home,q,[(0,e.home),(8,e.home)],seconds=8))
    (OUT/'fold_confirmation.json').write_text(json.dumps(folds,indent=2),encoding='utf-8')
    (OUT/'upright_frontier.json').write_text(json.dumps(frontier,indent=2),encoding='utf-8')


if __name__=='__main__':main()
