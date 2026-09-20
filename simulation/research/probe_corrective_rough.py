"""Readiness probe only: unchanged policy on existing 2..8mm rough grid/slopes."""
import argparse
import json
from pathlib import Path
import numpy as np
from corrective_common import OUT
from evaluate_corrective import run
from evaluate_policy import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--label',required=True);a=p.parse_args()
    dest=OUT/'rough_probe'/a.label;dest.mkdir(parents=True,exist_ok=False)
    jobs=[(terrain,seed) for terrain in ('roughgrid_8mm','slope_3','slope_-3') for seed in (211,212,213)]
    (dest/'plan.json').write_text(json.dumps(dict(jobs=jobs,speed=.2,delay=10,seconds=12,role='readiness diagnostic, not training or selection'),indent=2))
    records=[]
    for terrain,seed in jobs:
        case=dict(terrain=terrain,scenario='straight',program='legacy',depth=0,delay=10,speed=.2,seconds=12)
        m,arrays=run(a.policy,case,seed);name=f'{terrain}_s{seed}'
        r=dict(name=name,case=case,seed=seed,metrics=m,policy_sha256=sha(a.policy));records.append(r)
        (dest/f'{name}.json').write_text(json.dumps(r,indent=2));np.savez_compressed(dest/f'{name}.npz',**arrays)
        (dest/'matrix.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(dict(label=a.label,terrain=terrain,seed=seed,fell=m['fell'],traversed=m['terrain_traversed'],vx=m['body_vx_mean_m_s'])),flush=True)
    (dest/'complete.json').write_text(json.dumps(dict(status='COMPLETE',trials=len(records)),indent=2))


if __name__=='__main__':main()
