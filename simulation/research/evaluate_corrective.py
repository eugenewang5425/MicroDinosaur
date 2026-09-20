"""Original 45 retention cases plus nine independently seeded delay cases."""
import argparse
import json
from pathlib import Path
import numpy as np
from corrective_common import OUT
from evaluate_owned_terrain import cases,key,experiment,evaluate
from evaluate_policy import sha


def jobs(holdout_seed_base=201):
    yield from ((case,seed,'retention') for case in cases() for seed in (1,2,3))
    step=next(c for c in cases() if c['terrain']=='steps_10' and c['speed']==.2)
    yield from ((dict(step,delay=delay),seed,'holdout') for delay in (5,10,15) for seed in range(holdout_seed_base,holdout_seed_base+3))


def run(policy,case,seed):
    e=experiment(policy,case);before=e.sim.substep_callback;q=[]
    def capture():
        before()
        if e.active:q.append(e.sim.data.qpos[e.sim.jadr].copy())
    e.sim.substep_callback=capture;m,t=evaluate(e,case,seed);q=np.asarray(q)
    limits=e.sim.model.jnt_range[e.sim.jids];violation=np.maximum(np.maximum(limits[:,0]-q,q-limits[:,1]),0)
    m.update(joint_limit_violation_max_rad=float(violation.max()),worst_limit_joint=e.sim.names[int(violation.max(axis=0).argmax())])
    arrays=dict(trace=t,gaze=np.asarray(e.sim.gaze_trace),commands=np.asarray(e.command_rows),joints=q,
        head_controls=np.asarray(e.head_rows),head_physics=np.asarray(e.head_physics))
    assert all(np.isfinite(v).all() for v in arrays.values())
    return m,arrays


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--holdout-only',action='store_true');p.add_argument('--holdout-seed-base',type=int,default=201);a=p.parse_args()
    dest=OUT/'evaluation'/a.label;dest.mkdir(parents=True,exist_ok=False);records=[]
    for case,seed,suite in jobs(a.holdout_seed_base):
        if a.holdout_only and suite!='holdout':continue
        m,arrays=run(a.policy,case,seed);name=key(case,seed)
        record=dict(status='COMPLETE',case=case,seed=seed,suite=suite,name=name,metrics=m,policy_sha256=sha(a.policy))
        (dest/f'{name}.json').write_text(json.dumps(record,indent=2));np.savez_compressed(dest/f'{name}.npz',**arrays)
        records.append(record);(dest/'matrix.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(dict(label=a.label,suite=suite,name=name,fell=m['fell'],traversed=m['terrain_traversed'],vx=m['body_vx_mean_m_s'],limit=m['joint_limit_violation_max_rad'])),flush=True)
    (dest/'complete.json').write_text(json.dumps(dict(status='COMPLETE',trials=len(records),policy_sha256=sha(a.policy)),indent=2))


if __name__=='__main__':main()
