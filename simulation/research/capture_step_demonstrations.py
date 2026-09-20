"""Prespecified expert probes, then successful physical demonstration recording."""
import argparse
import json
import numpy as np
from demonstration_expert import OUT,POLICY,run
from evaluate_policy import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('probe','collect'),required=True);a=p.parse_args()
    folder=OUT/a.mode;folder.mkdir(parents=True,exist_ok=False)
    if a.mode=='probe':jobs=[(drive,10,101) for drive in (None,.45,.50,.55)]
    else:
        selection=json.loads((OUT/'probe/selection.json').read_text());drive=selection['drive']
        jobs=[(drive,delay,seed) for delay in (5,10,15) for seed in (11,12,13)]
    (folder/'plan.json').write_text(json.dumps(dict(jobs=jobs,desired_speed=.2,policy_sha256=sha(POLICY),
        admission='No fall, x displacement >=.8m, mean vx .15-.25m/s, joint limit violation <=.02rad, joint speed <=16.5rad/s, torque <=.6001Nm',
        selection='Among admitted non-baseline nominal probes, minimize absolute physical speed error to .2m/s; no physics or gain changes.'),indent=2))
    records=[];observations=[];actions=[];expert_observations=[];ids=[]
    for index,(drive,delay,seed) in enumerate(jobs):
        m,arrays=run(drive,delay,seed);name=f'drive{drive}_lag{delay}_s{seed}'
        r=dict(name=name,drive=drive,delay=delay,seed=seed,metrics=m,policy_sha256=sha(POLICY))
        (folder/f'{name}.json').write_text(json.dumps(r,indent=2));np.savez_compressed(folder/f'{name}.npz',**arrays)
        records.append(r)
        if a.mode=='collect' and m['demonstration_accepted']:
            observations.extend(arrays['obs']);actions.extend(arrays['actions']);expert_observations.extend(arrays['expert_obs']);ids.extend([index]*600)
        print(json.dumps(dict(name=name,accepted=m['demonstration_accepted'],fell=m['fell'],vx=m['body_vx_mean_m_s'],distance=m['forward_displacement_m'],limit=m['joint_limit_violation_max_rad'])),flush=True)
    (folder/'matrix.json').write_text(json.dumps(records,indent=2))
    if a.mode=='probe':
        accepted=[r for r in records if r['drive'] is not None and r['metrics']['demonstration_accepted']]
        assert accepted,'No physically admitted expert; do not train on failed demonstrations'
        chosen=min(accepted,key=lambda r:abs(r['metrics']['body_vx_mean_m_s']-.2))
        (folder/'selection.json').write_text(json.dumps(dict(drive=chosen['drive'],probe=chosen['name']),indent=2))
    else:
        assert len(observations)>=1800,'Require at least three accepted complete demonstrations'
        path=OUT/'step_demonstrations.npz'
        np.savez_compressed(path,obs=np.asarray(observations,np.float32),actions=np.asarray(actions,np.float32),
            expert_obs=np.asarray(expert_observations,np.float32),trial_id=np.asarray(ids,np.int32))
        (OUT/'dataset.json').write_text(json.dumps(dict(observations=len(observations),accepted_trials=len(observations)//600,
            attempted_trials=len(jobs),sha256=sha(path),source_policy_sha256=sha(POLICY),
            teacher_command=drive,student_command=.2,semantics='Physical desired velocity .2; old policy internal command calibration is an explicit expert controller. Raw action history preserved.',
            contact_and_joint_trajectories='collect/*.npz; trajectory and actions are physical rollouts, not kinematic teleports.'),indent=2))


if __name__=='__main__':main()
