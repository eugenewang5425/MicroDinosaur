"""Training-label fit only; this does not establish physical task success."""
import json
import numpy as np
import torch
from corrective_common import OUT,ROOT,SOURCE,RUNS,PREVIOUS


def actor(checkpoint,obs):
    weights=torch.load(checkpoint,map_location='cpu',weights_only=False)['actor_state_dict']
    x=torch.as_tensor(obs);x=(x-weights['obs_normalizer._mean'])/(weights['obs_normalizer._std']+.01)
    with torch.no_grad():
        for i in (0,2,4,6):
            x=torch.nn.functional.linear(x,weights[f'mlp.{i}.weight'],weights[f'mlp.{i}.bias'])
            if i<6:x=torch.nn.functional.elu(x)
    return x.numpy()


def main():
    contract=json.loads((ROOT/'20260913_handoff/native_v07/contract.json').read_text())
    names=contract['action_names'];scale=np.broadcast_to(np.asarray(contract['action_scale']).reshape(-1),(19,))
    leg=[i for i,n in enumerate(names) if any(k in n for k in ('hip','knee','ankle'))]
    checkpoints={'source':SOURCE}
    for arm in ('control','corrective'):
        provenance=json.loads((RUNS/f'microdinosaur_corrective_{arm}/20260914_train_512x201/run_provenance.json').read_text())
        assert provenance['status']=='COMPLETE';checkpoints[arm]=provenance['final_checkpoint']
    results={}
    for kind,path in (('success',PREVIOUS/'step_demonstrations.npz'),('recovery',OUT/'recovery_demonstrations.npz'),('stop',OUT/'stop_demonstrations.npz')):
        with np.load(path) as z:obs=z['obs'];targets=z['actions']
        results[kind]={}
        for arm,checkpoint in checkpoints.items():
            error=actor(checkpoint,obs)-targets;physical=error*scale
            results[kind][arm]=dict(frames=len(obs),raw_action_mse=float(np.mean(error**2)),
                leg_target_rmse_rad=float(np.sqrt(np.mean(physical[:,leg]**2))),
                per_joint_target_rmse_rad={name:float(v) for name,v in zip(names,np.sqrt(np.mean(physical**2,axis=0)))})
    (OUT/'training_label_fit.json').write_text(json.dumps(dict(role='Training label error, not rollout performance',results=results),indent=2))
    print(json.dumps({kind:{arm:values['leg_target_rmse_rad'] for arm,values in arms.items()} for kind,arms in results.items()},indent=2))


if __name__=='__main__':main()
