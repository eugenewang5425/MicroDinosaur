from pathlib import Path
import json
import numpy as np
from squat_plant import OUT
from hardware_sim import HardwareCase
from try_squat_reference import run

folder=OUT/'demonstrations';folder.mkdir(exist_ok=False)
records=[];observation=[];action=[];episode=[]
for seed in (611,612,613):
    for depth in (15,20,25):
        case=HardwareCase(physics_dt=.00125,motor_curve=True,voltage=12.,
            ground_friction={611:.6,612:1.,613:1.3}[seed])
        row=run(OUT/'plant',depth,case,seed,folder,demonstration=True)
        records.append(row)
        if row['passed']:
            with np.load(folder/(row['key']+'_demonstration.npz')) as data:
                observation.append(data['obs']);action.append(data['actions'])
                episode.extend([len(records)-1]*len(data['obs']))
        (folder/'summary.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
assert len(observation)>=6,'Not enough accepted physical demonstrations'
np.savez_compressed(OUT/'squat_demonstrations.npz',obs=np.concatenate(observation),
    actions=np.concatenate(action),episode=np.array(episode))
print('ACCEPTED',len(observation),'FRAMES',sum(map(len,observation)),flush=True)
