"""Matched development probes for the post-hoc offline fitting diagnostic."""
import argparse
import json
from pathlib import Path
import numpy as np
from corrective_common import OUT
from evaluate_owned_terrain import cases,key
from evaluate_corrective import run
from evaluate_policy import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--seed-base',type=int,default=61);p.add_argument('--resume',action='store_true');a=p.parse_args()
    dest=OUT/'offline_probe'/'evaluation'/a.label;dest.mkdir(parents=True,exist_ok=a.resume)
    stairs=next(c for c in cases() if c['terrain']=='steps_10' and c['speed']==.2)
    jobs=[(dict(stairs,delay=delay),seed) for delay in (5,10,15) for seed in range(a.seed_base,a.seed_base+3)]
    jobs += [(c,a.seed_base) for c in cases() if c['program']=='resume' or (c['terrain']=='flat' and c['scenario']=='straight')]
    records=json.loads((dest/'matrix.json').read_text()) if a.resume and (dest/'matrix.json').exists() else []
    assert all(r['policy_sha256']==sha(a.policy) for r in records)
    for case,seed in jobs:
        name=key(case,seed)
        if any(r['name']==name for r in records):continue
        try:m,arrays=run(a.policy,case,seed)
        except ValueError as error:
            if str(error)!='Calibration angular motion is excessive':raise
            r=dict(name=name,case=case,seed=seed,metrics=None,policy_sha256=sha(a.policy),status='CALIBRATION_FAILED',error=str(error),role='startup failed before candidate execution')
            records.append(r);(dest/f'{name}.json').write_text(json.dumps(r,indent=2));(dest/'matrix.json').write_text(json.dumps(records,indent=2))
            print(json.dumps(dict(label=a.label,name=name,status=r['status'])),flush=True);continue
        r=dict(name=name,case=case,seed=seed,metrics=m,policy_sha256=sha(a.policy),role='post-hoc development diagnostic');records.append(r)
        (dest/f'{name}.json').write_text(json.dumps(r,indent=2));np.savez_compressed(dest/f'{name}.npz',**arrays)
        (dest/'matrix.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(dict(label=a.label,name=name,fell=m['fell'],traversed=m['terrain_traversed'],vx=m['body_vx_mean_m_s'])),flush=True)
    (dest/'complete.json').write_text(json.dumps(dict(status='COMPLETE',trials=len(records),calibration_failures=sum(r.get('status')=='CALIBRATION_FAILED' for r in records)),indent=2))


if __name__=='__main__':main()
