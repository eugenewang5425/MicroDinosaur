"""Same 45 cases per arm, with additional actual joint-range diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate_owned_terrain import cases,key,experiment,evaluate
from demonstration_expert import OUT,ROOT
from evaluate_policy import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--label',required=True);a=p.parse_args()
    folder=OUT/'evaluation'/a.label;folder.mkdir(parents=True,exist_ok=False);records=[]
    old=ROOT/'20260914_imu_owned_terrain/evaluation/frozen_yaw/matrix.json'
    runtime=json.loads(old.read_text())[0]['source_hashes']
    assert all(sha(ROOT/name)==digest for name,digest in runtime.items()),'Keep paired runtime immutable'
    for case in cases():
        for seed in (1,2,3):
            e=experiment(a.policy,case);before=e.sim.substep_callback;q=[]
            def capture():
                before()
                if e.active:q.append(e.sim.data.qpos[e.sim.jadr].copy())
            e.sim.substep_callback=capture
            m,t=evaluate(e,case,seed);q=np.asarray(q);limits=e.sim.model.jnt_range[e.sim.jids]
            violations=np.maximum(np.maximum(limits[:,0]-q,q-limits[:,1]),0)
            j=int(violations.max(axis=0).argmax())
            m.update(joint_limit_violation_max_rad=float(violations.max()),worst_limit_joint=e.sim.names[j])
            name=key(case,seed);record=dict(status='COMPLETE',name=name,case=case,seed=seed,metrics=m,
                policy_sha256=sha(a.policy),source_hashes=runtime,evaluator_sha256=sha(Path(__file__)))
            arrays=dict(trace=t,gaze=np.asarray(e.sim.gaze_trace),commands=np.asarray(e.command_rows),joints=q,
                head_controls=np.asarray(e.head_rows),head_physics=np.asarray(e.head_physics))
            if hasattr(e,'user_rows'):arrays['user_commands']=np.asarray(e.user_rows)
            assert all(np.isfinite(v).all() for v in arrays.values())
            (folder/f'{name}.json').write_text(json.dumps(record,indent=2));np.savez_compressed(folder/f'{name}.npz',**arrays)
            records.append(record)
            print(json.dumps(dict(label=a.label,name=name,fell=m['fell'],traversed=m['terrain_traversed'],vx=m['body_vx_mean_m_s'],limit=m['joint_limit_violation_max_rad'])),flush=True)
    (folder/'matrix.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
