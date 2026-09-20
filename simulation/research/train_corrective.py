"""Bounded paired continuation, provenance capture, official normalized export."""
import os
os.environ['WANDB_MODE']='disabled'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time
import torch
import numpy as np
import onnxruntime as ort
from mjlab.scripts.train import run_train
from mjlab_microduck.export import ExportConfig,run_export
from corrective_cfg import build_config
from corrective_common import SOURCE,ROOT,OUT,PREVIOUS
from demonstration_nan_guard import install
from evaluate_policy import sha
from train_candidate import delay_contract


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('control','corrective'),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--envs',type=int,default=64)
    p.add_argument('--iterations',type=int,default=5);p.add_argument('--seed',type=int,default=47)
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    install(out/'nan_dumps');task,cfg=build_config(a.arm,a.envs,a.seed)
    cfg.agent.resume=True;cfg.agent.max_iterations=a.iterations;cfg.agent.save_interval=min(50,a.iterations)
    cfg.agent.run_name=out.name;cfg.agent.algorithm.learning_rate=3e-5;cfg.agent.algorithm.schedule='fixed'
    saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
    assert all(g['lr']==3e-5 for g in saved['optimizer_state_dict']['param_groups']);del saved
    source=out.parent/('source_'+sha(SOURCE)[:12]);source.mkdir(exist_ok=True)
    resume=source/SOURCE.name
    if not resume.exists():shutil.copy2(SOURCE,resume)
    assert sha(resume)==sha(SOURCE)
    cfg.agent.load_run=source.name;cfg.agent.load_checkpoint=resume.name;cfg=replace(cfg,enable_nan_guard=True)
    old=json.loads(SOURCE.with_name('run_provenance.json').read_text())
    sources={Path(p) for p in old['source_hashes']}
    sources.update({Path(__file__),ROOT/'corrective_cfg.py',ROOT/'corrective_common.py',ROOT/'collect_corrective.py',
        Path('D:/microduck_rl/src/mjlab_microduck/corrective_ppo.py'),OUT/'RECIPE.md',
        OUT/'stop_demonstrations.npz',OUT/'recovery_demonstrations.npz',PREVIOUS/'step_demonstrations.npz'})
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for i,path in enumerate(sorted(sources)):shutil.copy2(path,snapshot/f'{i}_{path.name}')
    r=dict(status='RUNNING',arm=a.arm,envs=a.envs,iterations=a.iterations,seed=a.seed,started_unix=time.time(),
        source_checkpoint=str(SOURCE),source_sha256=sha(SOURCE),optimizer_learning_rate=3e-5,
        source_hashes={str(p):sha(p) for p in sorted(sources)},delays=delay_contract(cfg),physics_dt=.00125,policy_dt=.02,
        anchor_steps=4,anchor_batch_size=1536,actor_dim=81,action_dim=19,head_runtime='yaw owned, unchanged')
    record=out/'run_provenance.json'
    def save():record.write_text(json.dumps(r,indent=2))
    save()
    try:
        run_train(task,cfg,out)
        final=max(out.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
        result=run_export(task,ExportConfig(checkpoint_file=str(final),onnx_file=str(out/'candidate.onnx'),num_envs=1))
        session=ort.InferenceSession(str(result.onnx_path),providers=['CPUExecutionProvider'])
        y=session.run(None,{'obs':np.zeros((1,81),np.float32)})[0];assert y.shape==(1,19) and np.isfinite(y).all()
        r.update(status='COMPLETE',final_checkpoint=str(final),final_sha256=sha(final),
            onnx=str(result.onnx_path),onnx_sha256=sha(result.onnx_path),finished_unix=time.time())
    except Exception as exc:r.update(status='FAILED',error=str(exc));raise
    finally:save()


if __name__=='__main__':main()
