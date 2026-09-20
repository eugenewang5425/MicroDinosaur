"""Second expert attempt: respect hip-yaw target range without changing physics."""
import argparse
import json
import numpy as np
from demonstration_expert import OUT,POLICY,run
from evaluate_policy import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('probe','collect'),required=True);a=p.parse_args()
    folder=OUT/f'bounded_{a.mode}';folder.mkdir(parents=True,exist_ok=False)
    if a.mode=='probe':jobs=[(drive,margin,10,seed) for drive in (.50,.55) for margin in (.03,.06) for seed in (11,101)]
    else:
        selection=json.loads((OUT/'bounded_probe/selection.json').read_text())
        jobs=[(selection['drive'],selection['margin'],delay,seed) for delay in (5,10,15) for seed in (11,12,13)]
    (folder/'plan.json').write_text(json.dumps(dict(jobs=jobs,reason='First expert failed collection: nominal falls and hip-yaw soft-limit violations. Preserve original trials. Clamp only two expert hip-yaw targets inside unchanged limits.',
        selection='Both probe seeds must pass original admission; then minimize mean physical speed error to .2. No admission threshold changes.',policy_sha256=sha(POLICY)),indent=2))
    records=[];bank={k:[] for k in ('obs','expert_obs','actions','trial_id')}
    for i,(drive,margin,delay,seed) in enumerate(jobs):
        m,arrays=run(drive,delay,seed,margin);name=f'drive{drive}_margin{margin}_lag{delay}_s{seed}'
        r=dict(name=name,drive=drive,margin=margin,delay=delay,seed=seed,metrics=m,policy_sha256=sha(POLICY));records.append(r)
        (folder/f'{name}.json').write_text(json.dumps(r,indent=2));np.savez_compressed(folder/f'{name}.npz',**arrays)
        if a.mode=='collect' and m['demonstration_accepted']:
            for k in ('obs','expert_obs','actions'):bank[k].extend(arrays[k])
            bank['trial_id'].extend([i]*600)
        print(json.dumps(dict(name=name,accepted=m['demonstration_accepted'],fell=m['fell'],vx=m['body_vx_mean_m_s'],distance=m['forward_displacement_m'],limit=m['joint_limit_violation_max_rad'])),flush=True)
    (folder/'matrix.json').write_text(json.dumps(records,indent=2))
    if a.mode=='probe':
        valid=[]
        for drive in (.50,.55):
            for margin in (.03,.06):
                rows=[r for r in records if r['drive']==drive and r['margin']==margin]
                if all(r['metrics']['demonstration_accepted'] for r in rows):valid.append((np.mean([abs(r['metrics']['body_vx_mean_m_s']-.2) for r in rows]),drive,margin))
        assert valid,'No safe bounded expert in this probe'
        _,drive,margin=min(valid)
        (folder/'selection.json').write_text(json.dumps(dict(drive=drive,margin=margin),indent=2))
    else:
        assert len(bank['obs'])>=1800,'Insufficient accepted demonstrations'
        path=OUT/'step_demonstrations.npz';assert not path.exists()
        np.savez_compressed(path,**{k:np.asarray(v,np.int32 if k=='trial_id' else np.float32) for k,v in bank.items()})
        (OUT/'dataset.json').write_text(json.dumps(dict(observations=len(bank['obs']),accepted_trials=len(bank['obs'])//600,attempted_trials=len(jobs),
            sha256=sha(path),source_policy_sha256=sha(POLICY),teacher_command=selection['drive'],hip_target_margin=selection['margin'],student_command=.2,
            source='bounded_collect',semantics='Desired speed .2; expert has explicit old-gait command calibration and bounded hip-yaw targets. Physical raw action history preserved.'),indent=2))


if __name__=='__main__':main()
