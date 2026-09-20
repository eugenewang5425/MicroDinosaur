"""Bounded independent PPO training through the official normalized exporter."""
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
from squat_skill_cfg import build_config,OUT,XML
from evaluate_policy import sha

p=argparse.ArgumentParser()
p.add_argument('--out',required=True,type=Path)
p.add_argument('--envs',type=int,default=64)
p.add_argument('--iterations',type=int,default=5)
p.add_argument('--seed',type=int,default=914)
p.add_argument('--residual',action='store_true')
p.add_argument('--checkpoint',type=Path,default=Path('D:/microduck_rl/logs/rsl_rl/velocity_microdinosaur/2026-09-13_13-46-57_velocity_microdinosaur/model_15500.pt'))
a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
task,cfg=build_config(a.envs,a.seed,residual=a.residual)
cfg=replace(cfg,enable_nan_guard=True)
cfg.agent.resume=True;cfg.agent.max_iterations=a.iterations;cfg.agent.save_interval=min(50,a.iterations)
cfg.agent.run_name=out.name;cfg.agent.algorithm.learning_rate=1e-4;cfg.agent.algorithm.schedule='fixed'
source=out.parent/('source_'+sha(a.checkpoint)[:12]+'_lr0.0001'+('_zero_residual' if a.residual else ''));source.mkdir(exist_ok=True)
dest=source/a.checkpoint.name
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
if a.residual:
    saved['actor_state_dict']['mlp.6.weight'].zero_()
    saved['actor_state_dict']['mlp.6.bias'].zero_()
    saved['actor_state_dict']['distribution.std_param'].fill_(.15)
    saved['optimizer_state_dict']['state']={}
for group in saved['optimizer_state_dict']['param_groups']:group['lr']=1e-4
if not dest.exists():torch.save(saved,dest)
cfg.agent.load_run=source.name;cfg.agent.load_checkpoint=dest.name
runner=load_runner_cls(task);task='Mjlab-Independent-Squat-MicroDinosaur'
register_mjlab_task(task,cfg.env,deepcopy(cfg.env),cfg.agent,runner)
snapshot=out/'source_snapshot';snapshot.mkdir()
root=Path(__file__).parent
sources=[Path(__file__),root/'squat_skill_cfg.py',root/'squat_plant.py',root/'jump_refine_cfg.py',
    root/'compact_contact_cfg.py',root/'run_jump_cfg.py',root/'transition_refine_cfg.py',
    root/'deep_crouch_cfg.py',root/'terrain_skill_cfg.py',root.parent/'microdinosaur/blend2mjcf.py',
    Path('D:/microduck_rl/src/mjlab_microduck/tasks/mdp.py'),
    Path('D:/microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py'),
    Path('D:/microduck_rl/src/mjlab_microduck/head_imu_action.py'),
    Path('D:/microduck_rl/src/mjlab_microduck/calibrated_head_action.py'),
    Path('D:/microduck_rl/src/mjlab_microduck/imu_owned_head.py'),
    OUT/'calibrated_reset_bank.json',OUT/'pose_references.json',XML]
for i,f in enumerate(sources):shutil.copy2(f,snapshot/f'{i}_{f.name}')
record=dict(status='RUNNING',task=task,envs=a.envs,iterations=a.iterations,seed=a.seed,
    source_checkpoint=str(a.checkpoint),source_sha256=sha(a.checkpoint),resume_sha256=sha(dest),
    plant_sha256=sha(OUT/'plant/nominal.mjb'),xml_sha256=sha(XML),
    source_hashes={str(f):sha(f) for f in sources},started_unix=time.time(),
    actor_dimensions=81,action_dimensions=19,learning_rate=1e-4,policy_dt=.02,physics_dt=.00125,
    motor_model='Provisional S288 bounds each physics substep, 11.1-12.6V, positive working-range delays',
    command='15-25mm smooth squat/hold/stand; 15% idle; 20% early return',
    reference='Reward-only joint/height target; no injected leg trajectory',
    runtime_adapter='Shared head IMU + ankle target +/-51deg, jaw target +0.04rad',
    nominal_sliding_friction=1.,friction_measured=False,production_replaced=False)
if a.residual:
    record.update(reference='Validated 25mm nominal joint trajectory plus learned leg balance residual',
        output_semantics='19 residual values; +/-0.015rad*tanh(action) on leg joints, others ignored; NOT raw position-policy output',
        initialization='Source hidden features and normalizer, zero final actor layer, reset optimizer, std=0.15',
        runtime_adapter='SquatResidualAction reference generator and bounds plus head IMU; CPU implementation in try_squat_reference.py')
path=out/'run_provenance.json';path.write_text(json.dumps(record,indent=2),encoding='utf-8')
try:
    run_train(task,cfg,out)
    final=max(out.glob('model_*.pt'),key=lambda f:int(f.stem.split('_')[-1]))
    export=run_export(task,ExportConfig(checkpoint_file=str(final),onnx_file=str(out/'candidate.onnx'),num_envs=1))
    session=ort.InferenceSession(str(export.onnx_path),providers=['CPUExecutionProvider'])
    output=session.run(None,{'obs':np.zeros((1,81),np.float32)})[0]
    assert output.shape==(1,19) and np.isfinite(output).all()
    record.update(status='COMPLETE',final_checkpoint=str(final),final_sha256=sha(final),
        onnx=str(export.onnx_path),onnx_sha256=sha(export.onnx_path))
except Exception as exc:
    record.update(status='FAILED',error=str(exc));raise
finally:
    record['finished_unix']=time.time();path.write_text(json.dumps(record,indent=2),encoding='utf-8')
