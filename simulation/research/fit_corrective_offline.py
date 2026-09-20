"""Post-hoc isolation: fit executed labels without intervening PPO updates."""
import os
os.environ['WANDB_MODE']='disabled'
os.environ['HF_HUB_OFFLINE']='1'
import json
import argparse
import time
import numpy as np
import torch
import onnxruntime as ort
from mjlab_microduck.corrective_ppo import CorrectivePPO
from mjlab_microduck.export import ExportConfig,run_export
from corrective_common import OUT,RUNS,OLD,PREVIOUS
from evaluate_policy import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rehearsal',action='store_true');args=parser.parse_args()
    dest=OUT/('offline_rehearsal' if args.rehearsal else 'offline_probe');dest.mkdir(exist_ok=False)
    previous=json.loads((RUNS/'microdinosaur_corrective_corrective/20260914_train_512x201/run_provenance.json').read_text())
    source=OUT/'offline_probe/offline_actor.pt' if args.rehearsal else previous['final_checkpoint'];saved=torch.load(source,map_location='cpu',weights_only=False)
    plan=dict(post_hoc=True,reason='Corrective arm retains about 0.094 rad leg-target RMSE; isolate label fitting from PPO.',
        steps=1000,batch_size=1792 if args.rehearsal else 1536,learning_rate=1e-4,seed=53,source_checkpoint=str(source),source_sha256=sha(source),
        new_physics_training_transitions=0,normalizer='frozen',policy_dimensions='81-to-19',
        development_seeds=[61,62,63],selection='Final step 1000 only; diagnostic, no automatic promotion',
        quota='same corrective quota plus 256 crouch/return frames' if args.rehearsal else 'same corrective quota as paired study',started_unix=time.time())
    if args.rehearsal:plan['reason']='First offline fit omitted all nonzero crouch commands and regressed posture; add validated recorded posture rehearsal.'
    (dest/'plan.json').write_text(json.dumps(plan,indent=2))
    torch.manual_seed(53);torch.set_num_threads(2);device='cuda:0'
    net=torch.nn.Sequential(torch.nn.Linear(81,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),
        torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,19)).to(device)
    weights=saved['actor_state_dict'];net.load_state_dict({k.removeprefix('mlp.'):v for k,v in weights.items() if k.startswith('mlp.')})
    mean=weights['obs_normalizer._mean'].to(device);std=weights['obs_normalizer._std'].to(device)
    learner=object.__new__(CorrectivePPO);learner.device=device;learner.corrective_arm='corrective';learner.banks={}
    for name,path in (('old',OLD),('success',PREVIOUS/'step_demonstrations.npz'),('stop',OUT/'stop_demonstrations.npz'),('recovery',OUT/'recovery_demonstrations.npz')):
        with np.load(path) as z:
            obs=torch.as_tensor(z['obs'],device=device);actions=torch.as_tensor(z['actions'],device=device)
            if name=='old':
                standing=obs[:,63:66].abs().max(1).values<1e-6
                learner.banks['old_stand']=(obs[standing],actions[standing]);learner.banks['old_walk']=(obs[~standing],actions[~standing])
            elif name=='recovery':
                mask=torch.as_tensor(z['frontier'].astype(bool),device=device)
                learner.banks['frontier']=(obs[mask],actions[mask]);learner.banks['continuation']=(obs[~mask],actions[~mask])
            else:learner.banks[name]=(obs,actions)
    if args.rehearsal:
        with np.load(OUT/'posture_retention.npz') as z:learner.banks['posture']=(torch.as_tensor(z['obs'],device=device),torch.as_tensor(z['actions'],device=device))
    def fit_errors():
        with torch.no_grad():return {name:float(((net((obs-mean)/(std+.01))-target)**2).mean()) for name,(obs,target) in learner.banks.items()}
    before=fit_errors();optimizer=torch.optim.Adam(net.parameters(),lr=1e-4);losses=[]
    for i in range(1000):
        obs,target,_=learner.supervised_batch()
        if args.rehearsal:
            posture,posture_target=learner.banks['posture'];ids=torch.randint(len(posture),(256,),device=device)
            obs=torch.cat((obs,posture[ids]));target=torch.cat((target,posture_target[ids]))
        prediction=net((obs-mean)/(std+.01));loss=(prediction-target).square().mean()
        assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
        losses.append(float(loss.detach()))
        if (i+1)%100==0:print('offline step',i+1,'loss',losses[-1],flush=True)
    after=fit_errors()
    for key,value in net.state_dict().items():weights['mlp.'+key]=value.cpu()
    # Preserve a loadable official-export artifact but discard stale PPO moments.
    # Critic/common_step_counter are source metadata, not new PPO training.
    saved['optimizer_state_dict']['state']={}
    saved['infos']['offline_fit']=dict(steps=1000,normalizer_frozen=True,critic_not_trained=True,ppo_optimizer_reset=True)
    checkpoint=dest/'offline_actor.pt';torch.save(saved,checkpoint);torch.save(optimizer.state_dict(),dest/'offline_optimizer.pt')
    result=run_export('Mjlab-Velocity-Flat-MicroDinosaur',ExportConfig(checkpoint_file=str(checkpoint),onnx_file=str(dest/'candidate.onnx'),num_envs=1))
    session=ort.InferenceSession(str(result.onnx_path),providers=['CPUExecutionProvider'])
    samples=torch.cat([v[0][::max(1,len(v[0])//50)] for v in learner.banks.values()])
    # Official export may enable CUDA TF32; compare export to a CPU FP32 actor.
    samples=samples.cpu();net=net.cpu();mean=mean.cpu();std=std.cpu()
    with torch.no_grad():expected=net((samples-mean)/(std+.01)).numpy()
    actual=np.concatenate([session.run(None,{'obs':row[None]})[0] for row in samples.cpu().numpy()])
    error=float(abs(actual-expected).max());assert error<2e-5
    record=dict(status='COMPLETE',before_mse=before,after_mse=after,export_max_error=error,export_observations=len(samples),
        policy_sha256=sha(result.onnx_path),checkpoint_sha256=sha(checkpoint),steps=1000,unique_new_frames=3550+(len(learner.banks['posture'][0]) if args.rehearsal else 0),
        repeated_supervision_samples=plan['batch_size']*1000,new_physics_training_transitions=0,finished_unix=time.time())
    (dest/'result.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))


if __name__=='__main__':main()
