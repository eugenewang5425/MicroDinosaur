"""Bounded local continuation on an explicit CAD-derived candidate model.

Uses the installed mjlab training loop and exporter; no network logger,
external model upload, force-kill supervisor, or replacement of a release ONNX.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ['WANDB_MODE'] = 'disabled'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import mjlab
import mujoco
import numpy as np
import onnxruntime as ort
from mjlab.scripts.train import TrainConfig, run_train
from mjlab_microduck.export import ExportConfig, run_export


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def portable_path(path):
    """Record a repository-relative path, or only a filename for external input."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return f"external/{resolved.name}"


def delay_contract(cfg):
    fields = ('delay_min_lag', 'delay_max_lag', 'delay_hold_prob', 'delay_update_period')
    return {
        'actuators': [{key: getattr(act, key) for key in fields}
                      for act in cfg.env.scene.entities['robot'].articulation.actuators],
        'actor': {name: {key: getattr(term, key) for key in fields}
                  for name, term in cfg.env.observations['actor'].terms.items()},
    }


def include_zero_delay(cfg):
    """Extend delay support only; retain maxima, resampling and all other fields."""
    articulation = cfg.env.scene.entities['robot'].articulation
    articulation.actuators = tuple(dataclasses.replace(act, delay_min_lag=0)
                                  for act in articulation.actuators)
    for name in ('joint_pos', 'joint_vel'):
        cfg.env.observations['actor'].terms[name].delay_min_lag = 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xml', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--iterations', type=int, default=5)
    p.add_argument('--envs', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=None)
    p.add_argument('--experiment-note', default='')
    p.add_argument('--action-alpha', type=float, default=None)
    p.add_argument('--delay-coverage', choices=('original', 'include-zero'), default='original')
    p.add_argument('--hardware-profile', choices=('legacy', 's288-protocol'), default='legacy')
    p.add_argument('--reward-recipe', choices=('current', 'v7'), default='current')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--head-gaze-weight', type=float, default=None,
                   help='Optional signed weight for neck angular-velocity cost; <= 0')
    args = p.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to reuse a run directory: {out}')
    out.mkdir(parents=True)
    task = 'Mjlab-Velocity-Flat-MicroDinosaur'
    cfg = TrainConfig.from_task(task)
    cfg.agent.seed = args.seed
    cfg.env.seed = args.seed
    cfg.env.scene.num_envs = args.envs
    cfg.env.scene.entities['robot'].spec_fn = lambda: mujoco.MjSpec.from_file(args.xml)
    if args.reward_recipe == 'v7':
        for name in ('lean_drift', 'contact_timing'):
            cfg.env.rewards.pop(name, None)
    if args.hardware_profile == 's288-protocol':
        from mjlab_microduck.s288_profile import apply_s288_protocol
        apply_s288_protocol(cfg.env)
    if args.head_gaze_weight is not None:
        if not np.isfinite(args.head_gaze_weight) or args.head_gaze_weight > 0:
            raise ValueError('head-gaze-weight must be finite and nonpositive')
        # Keep the term and all camera pose targets; isolate only its weight.
        cfg.env.rewards['head_gaze'].weight = args.head_gaze_weight
    delay_before = delay_contract(cfg)
    if args.delay_coverage == 'include-zero':
        include_zero_delay(cfg)
    if args.action_alpha is not None:
        if not 0.0 < args.action_alpha <= 1.0:
            raise ValueError('action-alpha must be in (0, 1]')
        cfg.env.actions['joint_pos'].lp_alpha = args.action_alpha
    cfg.agent.max_iterations = args.iterations
    cfg.agent.save_interval = min(50, args.iterations)
    cfg.agent.logger = 'tensorboard'
    cfg.agent.upload_model = False
    cfg.agent.resume = True
    # mjlab treats load_run as a regex over siblings of the new run. Pin a
    # hash-named source copy there; an absolute Windows path is not accepted.
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    source_store = out.parent / ('source_' + source_hash[:12])
    source_store.mkdir(exist_ok=True)
    source_copy = source_store / checkpoint.name
    if source_copy.exists():
        assert hashlib.sha256(source_copy.read_bytes()).hexdigest() == source_hash
    else:
        shutil.copy2(checkpoint, source_copy)
    cfg.agent.load_run = source_store.name
    cfg.agent.load_checkpoint = checkpoint.name
    cfg.agent.experiment_name = 'microdinosaur_v07_calibration'
    cfg.agent.run_name = out.name
    if args.learning_rate is not None:
        cfg.agent.algorithm.learning_rate = args.learning_rate
        cfg.agent.algorithm.schedule = 'fixed'
    cfg = dataclasses.replace(cfg, enable_nan_guard=True)
    recorded_args = {**vars(args), 'xml': portable_path(args.xml),
                     'checkpoint': portable_path(checkpoint), 'out': portable_path(out)}
    provenance = {**recorded_args, 'source_checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  'model_xml_sha256': hashlib.sha256(Path(args.xml).read_bytes()).hexdigest(),
                  'same_81D_actor_contract': True, 'head_IMU_actor_input': False,
                  'algorithm_changes': 'none' if args.learning_rate is None else 'fixed learning rate',
                  'action_filter_alpha_new': cfg.env.actions['joint_pos'].lp_alpha,
                  'effective_head_gaze_weight': cfg.env.rewards['head_gaze'].weight,
                  'head_world_gaze_weight': cfg.env.rewards['head_world_gaze'].weight,
                  'delay_contract_before': delay_before,
                  'delay_contract_after': delay_contract(cfg),
                  'microdino_probe_environment': os.environ.get('MICRODINO_PROBE', ''),
                  'status': 'RUNNING'}
    # Freeze the effective source for later A/B review. YAML preserves weights
    # and parameters but cannot preserve the implementation of reward callables.
    source_root = Path(__import__('mjlab_microduck').__file__).parent
    snapshot = out / 'source_snapshot'
    snapshot.mkdir()
    source_files = [source_root / name for name in (
        'tasks/mdp.py', 'tasks/microduck_velocity_env_cfg.py', 'tasks/__init__.py',
        'robot/microdinosaur_constants.py', 'action_filter.py', 'tasks/symmetry_microdinosaur.py',
        's288_protocol.py', 's288_profile.py')]
    source_files.append(Path(__file__))
    provenance['source_files'] = []
    for i, path in enumerate(source_files):
        dest = snapshot / f'{i}_{path.name}'
        shutil.copy2(path, dest)
        provenance['source_files'].append({'path': portable_path(path), 'snapshot': portable_path(dest),
                                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    record = out / 'run_provenance.json'
    record.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    run_train(task, cfg, out)
    checkpoints = sorted(out.glob('model_*.pt'), key=lambda path: int(path.stem.split('_')[1]))
    if not checkpoints:
        raise RuntimeError('Training finished without a checkpoint')
    final = checkpoints[-1]
    # Official export API includes the learned observation normalizer. Geometry
    # is immaterial for constructing the unchanged 81D/19D actor graph.
    export = run_export(task, ExportConfig(checkpoint_file=str(final),
                        onnx_file=str(out / 'candidate.onnx'), num_envs=1))
    session = ort.InferenceSession(str(export.onnx_path), providers=['CPUExecutionProvider'])
    y = session.run(None, {'obs': np.zeros((1, 81), dtype=np.float32)})[0]
    assert y.shape == (1, 19) and np.isfinite(y).all()
    provenance.update(status='COMPLETE', final_checkpoint=portable_path(final),
                      onnx=portable_path(export.onnx_path), finite_output_check=True,
                      final_checkpoint_sha256=hashlib.sha256(final.read_bytes()).hexdigest())
    record.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == '__main__':
    main()
