"""Flamingo(单脚站立)训练:官方配方移植,姿态内出生+保持。

源 = v2 纠正组 checkpoint(会站稳的身体);配置 = flamingo_cfg(官方奖励集)。
可用 STAIR_SOURCE 环境变量覆盖源 checkpoint。
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
from flamingo_cfg import build_config
from corrective_common import OUT, ROOT
from demonstration_nan_guard import install
from evaluate_policy import sha
from train_candidate import delay_contract

SOURCE = Path(os.environ.get('STAIR_SOURCE',
    'D:/microduck_rl/logs/rsl_rl/microdinosaur_corrective_v2'
    '/20260917_train_512x201/model_16650.pt'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--envs', type=int, default=512)
    p.add_argument('--iterations', type=int, default=400)
    p.add_argument('--seed', type=int, default=47)
    p.add_argument('--save-interval', type=int, default=10)
    p.add_argument('--standing-prob', type=float, default=0.,
                   help='出生时直立姿势比例(stage-2 循环训练用)')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--push', type=float, default=0.0,
                   help='抗干扰推力幅度(m/s), 0=零干扰')
    p.add_argument('--pose', type=float, default=0.0,
                   help='固定姿态: +1=只前伸, -1=只后折, 0=双姿态随机')
    p.add_argument('--tilt-threshold', type=float, default=.40)
    p.add_argument('--tilt-weight', type=float, default=4.)
    a = p.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)   # 允许续跑/重入同一目录
    install(out / 'nan_dumps')
    # --pose fixed: 前伸/后折拆成两个专用模型(用户 2026-09-20 指示),
    # +1 = 只训前伸(出生与奖励目标恒前伸), -1 = 只训后折, 0 = 双姿态随机
    task, cfg = build_config(a.envs, a.seed, standing_prob=a.standing_prob,
                             push=a.push, fix_pose=a.pose,
                             tilt_threshold=a.tilt_threshold,
                             tilt_weight=a.tilt_weight)
    cfg.agent.resume = True
    cfg.agent.max_iterations = a.iterations
    cfg.agent.save_interval = min(a.save_interval, a.iterations)
    cfg.agent.run_name = out.name
    cfg.agent.algorithm.learning_rate = a.lr
    cfg.agent.algorithm.schedule = 'fixed'
    saved = torch.load(SOURCE, map_location='cpu', weights_only=False)
    for g in saved['optimizer_state_dict']['param_groups']:
        g['lr'] = a.lr
    del saved
    source = out.parent / ('source_' + sha(SOURCE)[:12])
    source.mkdir(exist_ok=True)
    resume = source / SOURCE.name
    if not resume.exists():
        shutil.copy2(SOURCE, resume)
    cfg.agent.load_run = source.name
    cfg.agent.load_checkpoint = resume.name
    cfg = replace(cfg, enable_nan_guard=True)
    sources = {Path(__file__), ROOT / 'flamingo_cfg.py', ROOT / 'flamingo_mdp.py',
               ROOT / 'corrective_cfg_v2.py', ROOT / 'corrective_rough_cfg.py',
               ROOT / 'corrective_common.py',
               Path('D:/microduck_rl/src/mjlab_microduck/corrective_ppo.py'),
               OUT / 'RECIPE_V2.md', OUT / 'probe_latest.json'}
    snapshot = out / 'source_snapshot'
    snapshot.mkdir()
    for i, path in enumerate(sorted(sources)):
        shutil.copy2(path, snapshot / f'{i}_{path.name}')
    r = dict(status='RUNNING', arm='flamingo', envs=a.envs,
             iterations=a.iterations, seed=a.seed, started_unix=time.time(),
             source_checkpoint=str(SOURCE), source_sha256=sha(SOURCE),
             optimizer_learning_rate=a.lr,
             source_hashes={str(p): sha(p) for p in sorted(sources)},
             delays=delay_contract(cfg), physics_dt=.00125, policy_dt=.02,
             actor_dim=81, action_dim=19,
             task='Flamingo 单脚站立: 官方配方移植(姿态内出生+保持)')
    record = out / 'run_provenance.json'

    def save():
        record.write_text(json.dumps(r, indent=2))
    save()
    try:
        run_train(task, cfg, out)
        final = max(out.glob('model_*.pt'), key=lambda f: int(f.stem.split('_')[-1]))
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
