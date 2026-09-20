"""v2 纠正轮训练:源 = 最新纠正组 checkpoint,数据 = v2 库,配额 = CorrectivePPOv2。

单臂(不再配 control 对照):本轮目的是推进线路而非对照实验,验收门不变。
先 64×5 冒烟,再 512×201 正式(与上一轮同预算)。
"""
import os
os.environ['WANDB_MODE'] = 'disabled'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
from mjlab_microduck.export import ExportConfig, run_export
from stair_climb_cfg import build_config
from corrective_common import OUT, ROOT
from demonstration_nan_guard import install
from evaluate_policy import sha
from train_candidate import delay_contract

import os as _os
SOURCE = Path(_os.environ.get('STAIR_SOURCE',
    'D:/microduck_rl/logs/rsl_rl/microdinosaur_stair_climb/20260918_s1b_256x250/model_16899.pt'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--envs', type=int, default=64)
    p.add_argument('--iterations', type=int, default=5)
    p.add_argument('--seed', type=int, default=47)
    p.add_argument('--save-interval', type=int, default=10,
                   help='存档间隔(迭代),默认 10;崩溃多发生在 30-70 迭代,'
                        '间隔 50 会导致崩溃时无 checkpoint 可续')
    p.add_argument('--difficulty', type=float, default=None,
                   help='固定台阶难度 d: 每级高 h=5+5d mm (2.0->15mm, 3.0->20mm)')
    a = p.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    install(out / 'nan_dumps')
    from stair_climb_cfg import DEFAULT_DIFFICULTY
    task, cfg = build_config(a.envs, a.seed,
                             DEFAULT_DIFFICULTY if a.difficulty is None else a.difficulty)
    cfg.agent.resume = True
    cfg.agent.max_iterations = a.iterations
    cfg.agent.save_interval = min(a.save_interval, a.iterations)
    cfg.agent.run_name = out.name
    cfg.agent.algorithm.learning_rate = 3e-5
    cfg.agent.algorithm.schedule = 'fixed'
    saved = torch.load(SOURCE, map_location='cpu', weights_only=False)
    assert all(g['lr'] == 3e-5 for g in saved['optimizer_state_dict']['param_groups'])
    del saved
    source = out.parent / ('source_' + sha(SOURCE)[:12])
    source.mkdir(exist_ok=True)
    resume = source / SOURCE.name
    if not resume.exists():
        shutil.copy2(SOURCE, resume)
    assert sha(resume) == sha(SOURCE)
    cfg.agent.load_run = source.name
    cfg.agent.load_checkpoint = resume.name
    cfg = replace(cfg, enable_nan_guard=True)
    sources = {Path(__file__), ROOT / 'stair_climb_cfg.py', ROOT / 'corrective_cfg_v2.py', ROOT / 'corrective_rough_cfg.py', ROOT / 'corrective_common.py',
               ROOT / 'collect_corrective_v2.py', ROOT / 'probe_latest_failures.py',
               ROOT / 'collect_corrective.py',
               Path('D:/microduck_rl/src/mjlab_microduck/corrective_ppo.py'),
               OUT / 'RECIPE_V2.md', OUT / 'probe_latest.json',
               OUT / 'stop_demonstrations_v2.npz',
               OUT / 'recovery_demonstrations_v2.npz',
               OUT.parent / '20260914_demonstrations' / 'step_demonstrations.npz'}
    snapshot = out / 'source_snapshot'
    snapshot.mkdir()
    for i, path in enumerate(sorted(sources)):
        shutil.copy2(path, snapshot / f'{i}_{path.name}')
    r = dict(status='RUNNING', arm='stair_climb', envs=a.envs,
             iterations=a.iterations, seed=a.seed, started_unix=time.time(),
             source_checkpoint=str(SOURCE), source_sha256=sha(SOURCE),
             optimizer_learning_rate=3e-5,
             source_hashes={str(p): sha(p) for p in sorted(sources)},
             delays=delay_contract(cfg), physics_dt=.00125, policy_dt=.02,
             anchor_steps=4, anchor_batch_size=1536, actor_dim=81, action_dim=19,
             task='上高台阶模组: 地形=台阶课程(up20%+flat50%+down10%+uneven20%)',
             single_variable='foot_clearance/foot_swing_height target_height 0.02->0.04',
             datasets=dict(stop='stop_demonstrations_v2.npz',
                           recovery='recovery_demonstrations_v2.npz',
                           success='step_demonstrations.npz(上一轮)'),
             quota='128 stand/384 walk/384 stop/256 success/192 frontier/192 continuation')
    record = out / 'run_provenance.json'

    def save():
        record.write_text(json.dumps(r, indent=2))
    save()
    try:
        run_train(task, cfg, out)
        final = max(out.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[-1]))
        result = run_export(task, ExportConfig(checkpoint_file=str(final),
                                               onnx_file=str(out / 'candidate.onnx'),
                                               num_envs=1))
        session = ort.InferenceSession(str(result.onnx_path),
                                       providers=['CPUExecutionProvider'])
        y = session.run(None, {'obs': np.zeros((1, 81), np.float32)})[0]
        assert y.shape == (1, 19) and np.isfinite(y).all()
        r.update(status='COMPLETE', final_checkpoint=str(final),
                 final_sha256=sha(final), onnx=str(result.onnx_path),
                 onnx_sha256=sha(result.onnx_path), finished_unix=time.time())
    except Exception as exc:
        r.update(status='FAILED', error=str(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
