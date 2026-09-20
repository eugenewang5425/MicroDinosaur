"""Fit only verified position-action demonstrations; export with official normalization."""
import os
os.environ.update(WANDB_MODE='disabled',HF_HUB_OFFLINE='1')
from pathlib import Path
from copy import deepcopy
import json,time
import torch
import numpy as np
import onnxruntime as ort
from mjlab.tasks.registry import register_mjlab_task,load_runner_cls
from mjlab_microduck.export import run_export,ExportConfig
from squat_skill_cfg import build_config,OUT
from evaluate_policy import sha

dest=OUT/'imitation';dest.mkdir(exist_ok=False)
source=Path('D:/microduck_rl/logs/rsl_rl/velocity_microdinosaur/2026-09-13_13-46-57_velocity_microdinosaur/model_15500.pt')
saved=torch.load(source,map_location='cpu',weights_only=False)
weights=saved['actor_state_dict'];torch.manual_seed(915);torch.set_num_threads(2)
device='cuda:0'
net=torch.nn.Sequential(torch.nn.Linear(81,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),
    torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,19)).to(device)
net.load_state_dict({k.removeprefix('mlp.'):v for k,v in weights.items() if k.startswith('mlp.')})
mean=weights['obs_normalizer._mean'].to(device);std=weights['obs_normalizer._std'].to(device)
data=np.load(OUT/'squat_demonstrations.npz')
obs=torch.tensor(data['obs'],device=device);target=torch.tensor(data['actions'],device=device)
optimizer=torch.optim.Adam(net.parameters(),lr=1e-4)
def error():
    with torch.no_grad():return float((net((obs-mean)/(std+.01))-target).square().mean().sqrt())
before=error();losses=[]
for step in range(1500):
    ids=torch.randint(len(obs),(1024,),device=device);batch=obs[ids].clone()
    # A small observation augmentation prevents simply copying last_action.
    # These are augmented samples, not additional physical transitions.
    batch[:,44:63]+=torch.randn_like(batch[:,44:63])*.02
    loss=(net((batch-mean)/(std+.01))-target[ids]).square().mean()
    assert torch.isfinite(loss);optimizer.zero_grad();loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
    losses.append(float(loss.detach()))
    if (step+1)%250==0:print(step+1,losses[-1],flush=True)
after=error()
for key,value in net.state_dict().items():weights['mlp.'+key]=value.detach().cpu()
weights['distribution.std_param'].fill_(.08)
saved['optimizer_state_dict']['state']={}
saved['infos']['squat_imitation']=dict(steps=1500,normalizer_frozen=True,critic_not_trained=True,
    accepted_physical_frames=len(obs),augmentation_last_action_std_rad=.02)
checkpoint=dest/'model_15500.pt';torch.save(saved,checkpoint)
base,cfg=build_config(1,915);task='Mjlab-Independent-Squat-Imitation-MicroDinosaur'
register_mjlab_task(task,cfg.env,deepcopy(cfg.env),cfg.agent,load_runner_cls(base))
export=run_export(task,ExportConfig(checkpoint_file=str(checkpoint),onnx_file=str(dest/'candidate.onnx'),num_envs=1))
session=ort.InferenceSession(str(export.onnx_path),providers=['CPUExecutionProvider'])
sample=data['obs'][::max(1,len(obs)//100)]
net=net.cpu();mean=mean.cpu();std=std.cpu()
with torch.no_grad():expected=net((torch.tensor(sample)-mean)/(std+.01)).numpy()
actual=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in sample])
export_error=float(abs(actual-expected).max());assert export_error<2e-5
result=dict(status='COMPLETE',learning='Supervised imitation, not new PPO updates',steps=1500,
    physical_demonstration_frames=len(obs),sampled_supervision=1500*1024,before_rmse_rad=before,
    after_rmse_rad=after,export_max_error=export_error,checkpoint_sha256=sha(checkpoint),
    onnx_sha256=sha(export.onnx_path),source_sha256=sha(source),dataset_sha256=sha(OUT/'squat_demonstrations.npz'),
    source_script_sha256=sha(__file__),finished_unix=time.time())
(dest/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
