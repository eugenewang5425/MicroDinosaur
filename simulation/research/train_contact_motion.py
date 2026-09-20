"""Bounded official PPO continuation on an isolated contact-motion task."""
import os
os.environ.update(WANDB_MODE='disabled',HF_HUB_OFFLINE='1',CUDA_VISIBLE_DEVICES='0')
from pathlib import Path
from dataclasses import replace
from copy import deepcopy
import argparse,json,time,shutil
import torch
import numpy as np
import onnxruntime as ort
from mjlab.scripts.train import run_train
from mjlab.tasks.registry import register_mjlab_task,load_runner_cls
from mjlab_microduck.export import run_export,ExportConfig
from contact_motion_cfg import build_config,OUT,XML
from evaluate_policy import sha

p=argparse.ArgumentParser();p.add_argument('--skill',choices=['run','getup'],required=True)
p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--envs',type=int,default=64);p.add_argument('--iterations',type=int,default=5)
p.add_argument('--seed',type=int,default=914);p.add_argument('--flight-refine',action='store_true')
p.add_argument('--recovery-dense',action='store_true')
p.add_argument('--ground-tax-after-steps',type=int,default=None,
    help='Enable the body-on-floor tax only after this many env steps (ramped over 2M).')
p.add_argument('--body-clearance',action='store_true',
    help='Add the force sensor and dense non-foot floor-clearance reward.')
p.add_argument('--quiet-stand',action='store_true',
    help='Add the gated stillness reward for residual get-up wobble.')
p.add_argument('--yaw-hold',action='store_true',
    help='Penalise trunk yaw drift from the episode start heading.')
p.add_argument('--coach',action='store_true',
    help='Add the v7-HOME pose+hold demonstration reward (coach).')
p.add_argument('--sitfold-residual',action='store_true',
    help='坐撑示范+有界残差:折腿脚本(验证过 4/4)+±0.015rad 残差,按落地签名只对坐撑起点生效。')
a=p.parse_args()
out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
selected=json.loads((OUT/'selected_contact.json').read_text());plant=OUT/selected.get('plant_directory','plant')
task,cfg=build_config(a.skill,a.envs,a.seed,flight_refine=a.flight_refine,recovery_dense=a.recovery_dense,
    ground_tax_after_steps=a.ground_tax_after_steps,
    body_clearance=a.body_clearance,quiet_stand=a.quiet_stand,coach=a.coach,
    yaw_hold=a.yaw_hold,sitfold_residual=a.sitfold_residual);cfg=replace(cfg,enable_nan_guard=True)
cfg.agent.resume=True;cfg.agent.max_iterations=a.iterations;cfg.agent.save_interval=min(50,a.iterations)
cfg.agent.run_name=out.name;cfg.agent.algorithm.learning_rate=1e-4;cfg.agent.algorithm.schedule='fixed'
source=out.parent/('source_'+sha(a.checkpoint)[:12]+'_lr0.0001');source.mkdir(exist_ok=True)
dest=source/a.checkpoint.name
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
for group in saved['optimizer_state_dict']['param_groups']:group['lr']=1e-4
if not dest.exists():torch.save(saved,dest)
cfg.agent.load_run=source.name;cfg.agent.load_checkpoint=dest.name
runner=load_runner_cls(task);task='Mjlab-Contact-MicroDinosaur-'+a.skill
register_mjlab_task(task,cfg.env,deepcopy(cfg.env),cfg.agent,runner)
root=Path(__file__).parent;snapshot=out/'source_snapshot';snapshot.mkdir()
sources=[Path(__file__),root/'contact_motion_cfg.py',root/'squat_skill_cfg.py',root/'squat_plant.py',
    root/'jump_refine_cfg.py',root/'run_jump_cfg.py',root/'transition_refine_cfg.py',root/'compact_contact_cfg.py',
    Path('D:/microduck_rl/src/mjlab_microduck/tasks/mdp.py'),
    Path('D:/microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py'),
    Path('D:/microduck_rl/src/mjlab_microduck/head_imu_action.py'),
    Path('D:/microduck_rl/src/mjlab_microduck/calibrated_head_action.py'),
    Path('D:/microduck_rl/src/mjlab_microduck/imu_owned_head.py'),XML,OUT/'selected_contact.json',
    OUT/((('getup' if a.skill=='getup' else 'standing')+'_reset_bank')+selected.get('bank_suffix','')+'.json')]
if a.skill=='getup' and 'getup_bank' in selected:sources[-1]=OUT/selected['getup_bank']
for i,f in enumerate(sources):shutil.copy2(f,snapshot/f'{i}_{f.name}')
record=dict(status='RUNNING',skill=a.skill,envs=a.envs,iterations=a.iterations,seed=a.seed,
    flight_refine=a.flight_refine,
    recovery_dense=a.recovery_dense,sitfold_residual=a.sitfold_residual,
    source_checkpoint=str(a.checkpoint),source_sha256=sha(a.checkpoint),resume_sha256=sha(dest),
    plant_sha256=sha(plant/'nominal.mjb'),sources={str(f):sha(f) for f in sources},started_unix=time.time(),
    actor_dim=81,action_dim=19,action_semantics='Raw position offsets, positive-delay S288 envelope and head IMU, jaw +.04 and ankle +/-51deg operating bounds',
    physics_dt=.00125,policy_dt=.02,learning_rate=1e-4,external_assistance_during_learning=False,
    production_replaced=False,self_collision_release=False)
path=out/'run_provenance.json';path.write_text(json.dumps(record,indent=2))
try:
    run_train(task,cfg,out)
    final=max(out.glob('model_*.pt'),key=lambda f:int(f.stem.split('_')[-1]))
    result=run_export(task,ExportConfig(checkpoint_file=str(final),onnx_file=str(out/'candidate.onnx'),num_envs=1))
    session=ort.InferenceSession(str(result.onnx_path),providers=['CPUExecutionProvider'])
    output=session.run(None,{'obs':np.zeros((1,81),np.float32)})[0]
    assert output.shape==(1,19) and np.isfinite(output).all()
    record.update(status='COMPLETE',final_checkpoint=str(final),final_sha256=sha(final),onnx_sha256=sha(result.onnx_path))
except Exception as exc:record.update(status='FAILED',error=repr(exc));raise
finally:record['finished_unix']=time.time();path.write_text(json.dumps(record,indent=2))
