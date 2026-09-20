"""Bounded official PPO continuation and normalized ONNX export."""
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
import mujoco
import numpy as np
import onnxruntime as ort
from mjlab.scripts.train import run_train
from mjlab_microduck.export import ExportConfig, run_export
from terrain_skill_cfg import build_config, XML
from evaluate_policy import sha
from train_candidate import delay_contract


def equal_checkpoint_values(a, b):
    if torch.is_tensor(a): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal_checkpoint_values(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(equal_checkpoint_values(v, w) for v, w in zip(a, b))
    if isinstance(a, np.ndarray): return np.array_equal(a, b)
    return a == b


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skill', choices=('terrain', 'crouch', 'deep_crouch', 'transition_refine','imu_owned_terrain','run','jump','jump_refine','foot_flight'), required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--envs', type=int, default=64)
    p.add_argument('--iterations', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--learning-rate', type=float, default=None)
    args = p.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    if args.skill in ('jump_refine','foot_flight'):
        from research_nan_guard import install
        install(out/'nan_dumps')
        if args.skill == 'foot_flight':
            from foot_flight_cfg import build_config as build_refined_jump
        else:
            from jump_refine_cfg import build_config as build_refined_jump
        task,cfg=build_refined_jump(args.envs,args.seed)
    elif args.skill in ('run','jump'):
        from research_nan_guard import install
        install(out/'nan_dumps')
        from run_jump_cfg import build_config as build_motion
        task,cfg=build_motion(args.skill,args.envs,args.seed)
    elif args.skill == 'imu_owned_terrain':
        from research_nan_guard import install
        install(out/'nan_dumps')
        from terrain_lane_cfg import build_config as build_lanes
        task,cfg=build_lanes(args.envs,args.seed)
    elif args.skill == 'transition_refine':
        from transition_refine_cfg import build_config as build_refine
        task, cfg = build_refine(args.envs, args.seed)
    elif args.skill == 'deep_crouch':
        from deep_crouch_cfg import build_config as build_deep
        task, cfg = build_deep(args.envs, args.seed)
    else:
        task, cfg = build_config(args.skill, args.envs, args.seed)
    cfg.agent.resume = True; cfg.agent.max_iterations = args.iterations
    cfg.agent.save_interval = min(50, args.iterations); cfg.agent.run_name = out.name
    # PPO.load restores optimizer groups, including LR. Match the advertised
    # fixed LR to the resumed optimizer rather than logging a fictitious value.
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    rates = {float(g['lr']) for g in saved['optimizer_state_dict']['param_groups']}
    assert len(rates) == 1, 'Multiple optimizer rates need an explicit recipe'
    source_learning_rate = rates.pop()
    resumed_learning_rate = args.learning_rate if args.learning_rate is not None else source_learning_rate
    assert resumed_learning_rate > 0
    cfg.agent.algorithm.learning_rate = resumed_learning_rate
    cfg.agent.algorithm.schedule = 'fixed'
    source = out.parent/('source_'+sha(args.checkpoint)[:12]+f'_lr{resumed_learning_rate:.9g}'); source.mkdir(exist_ok=True)
    dest = source/args.checkpoint.name
    if args.learning_rate is not None:
        # Derived resume checkpoint: change optimizer LR, preserve all model,
        # normalizer, momentum and progress state. The source is never edited.
        for group in saved['optimizer_state_dict']['param_groups']:
            group['lr'] = resumed_learning_rate
        if not dest.exists(): torch.save(saved, dest)
        check = torch.load(dest, map_location='cpu', weights_only=False)
        assert all(g['lr'] == resumed_learning_rate for g in check['optimizer_state_dict']['param_groups'])
        assert equal_checkpoint_values(saved, check), 'Derived checkpoint differs from the exact intended state'
        del check
    else:
        if not dest.exists(): shutil.copy2(args.checkpoint, dest)
        assert sha(dest) == sha(args.checkpoint)
    del saved
    cfg.agent.load_run = source.name; cfg.agent.load_checkpoint = dest.name
    cfg = replace(cfg, enable_nan_guard=True)
    snapshot = out/'source_snapshot'; snapshot.mkdir()
    sources = [Path(__file__), Path(__file__).with_name('terrain_skill_cfg.py'),
        Path('D:/microduck_rl/src/mjlab_microduck/tasks/mdp.py'),
        Path('D:/microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py'),
        Path('D:/microduck_rl/src/mjlab_microduck/tasks/__init__.py'),
        Path('D:/microduck_rl/src/mjlab_microduck/s288_profile.py'),
        Path('D:/microduck_rl/src/mjlab_microduck/s288_protocol.py'), XML]
    if args.skill in ('deep_crouch','transition_refine','imu_owned_terrain','run','jump','jump_refine','foot_flight'):
        sources.extend([Path(__file__).with_name('deep_crouch_cfg.py'),
            Path('D:/microduck_rl/src/mjlab_microduck/head_imu_action.py'),
            Path('D:/microduck_rl/src/mjlab_microduck/head_attitude_torch.py'),
            Path(__file__).parent/'20260914_fold_recovery/fold_geometry.json'])
    if args.skill in ('transition_refine','imu_owned_terrain','run','jump','jump_refine','foot_flight'):
        sources.extend([Path(__file__).with_name('transition_refine_cfg.py'),
            Path('D:/microduck_rl/src/mjlab_microduck/calibrated_head_action.py'),
            Path('D:/microduck_rl/src/mjlab_microduck/retention_ppo.py'),
            Path(__file__).parent/'20260914_transition_refine/calibrated_reset_bank.json',
            Path(__file__).parent/'20260914_transition_refine/v7_anchor.npz'])
    if args.skill == 'imu_owned_terrain':
        sources.extend([Path(__file__).with_name('terrain_lane_cfg.py'),
            Path(__file__).with_name('research_nan_guard.py'),
            Path('D:/microduck_rl/src/mjlab_microduck/imu_owned_head.py')])
    if args.skill in ('run','jump','jump_refine','foot_flight'):
        sources.extend([Path(__file__).with_name('run_jump_cfg.py'),
            Path(__file__).with_name('research_nan_guard.py'),
            Path('D:/microduck_rl/src/mjlab_microduck/imu_owned_head.py'),
            Path(__file__).parent/'20260914_terrain_skills/planned_motion_contacts/audit.json'])
    if args.skill in ('jump_refine','foot_flight'):
        sources.append(Path(__file__).with_name('jump_refine_cfg.py'))
    if args.skill == 'foot_flight':
        sources.extend([Path(__file__).with_name('foot_flight_cfg.py'),
            Path(__file__).parent/'20260914_foot_flight/deep_targets.json',
            Path(__file__).parent/'20260914_foot_flight/selection.json'])
    for i, path in enumerate(sources): shutil.copy2(path, snapshot/f'{i}_{path.name}')
    provenance = dict(status='RUNNING', skill=args.skill, task=task, envs=args.envs, iterations=args.iterations,
        seed=args.seed, source_checkpoint=str(args.checkpoint), source_sha256=sha(args.checkpoint),
        xml_sha256=sha(XML), started_unix=time.time(), physics_dt=.00125, policy_dt=.02,
        delays=delay_contract(cfg), nominal_kp=7, nominal_kd=.8,
        head_control_in_gpu_training=args.skill in ('deep_crouch','transition_refine','imu_owned_terrain','run','jump','jump_refine','foot_flight'), external_head_control_required_in_evaluation=True,
        optimizer_learning_rate=resumed_learning_rate,
        source_optimizer_learning_rate=source_learning_rate, resume_checkpoint=str(dest), resume_sha256=sha(dest),
        learning_rate_source='Explicit derived optimizer override' if args.learning_rate is not None else 'Inherited optimizer',
        notes='Fixed 81D actor; terrain ray reward-only; body pose z command is slot 9 of 18 commands',
        source_hashes={str(path): sha(path) for path in sources})
    if args.skill in ('transition_refine','imu_owned_terrain'):
        provenance.update(teacher_checkpoint=cfg.agent.algorithm.teacher_checkpoint,
            teacher_sha256=sha(cfg.agent.algorithm.teacher_checkpoint),anchor_steps=cfg.agent.algorithm.anchor_steps,
            initialization='Calibrated nominal v7 physics/sensor snapshot with randomized world yaw; DR unchanged',
            head_reference='Same navigation yaw and neutral IMU attitude for controller and reward')
    if args.skill == 'imu_owned_terrain':
        provenance.update(head_ownership='Yaw uses delayed encoder baseline; pitch/roll retain policy baseline plus IMU feedback',
            terrain='50% flat, 25% ascending and 25% descending 5-10mm physical steps; 0.2/0.35/0.55m/s terrain command buckets')
    if args.skill in ('run','jump','jump_refine','foot_flight'):
        provenance.update(head_ownership='Yaw encoder baseline; pitch/roll policy baseline plus shared head IMU',
            terrain='Flat only; current-CAD conservative floor-contact proxies; terminate on nonfoot support',
            teacher_loss=False,phase='Independent specialist; no modification to production policy',
            source_pushes=False,kinematic_or_impulse_takeoff_assistance=False)
    if args.skill == 'jump_refine':
        provenance.update(motor_envelope='Provisional four-quadrant bounds applied every physics substep; voltage uniformly 11.1-12.6V, inherited 0.48-0.60Nm effort caps',
            takeoff_reward='Mass-weighted whole-robot COM velocity, not trunk velocity',
            joint_margin_rad=.10,joint_limit_termination_rad=.02,
            command_ranges=dict(crouch_m=[.025,.03],extension_m=[.008,.015],preparation_s=[1.2,1.8],extension_s=[.4,.8]))
    if args.skill == 'foot_flight':
        provenance.update(motor_envelope='Same per-substep provisional S288 bounds as jump_refine; 11.1-12.6V',
            jump_objective='Whole-foot minimum >5mm for 60ms and >=10mm peak in the same interval; no positive COM launch reward',
            physics_substep_feet_measurement=True,crouch_curriculum='30 to 50mm over 240 updates, then 50mm, with 2mm episode jitter',
            airborne_tuck_pose='50mm fold IK with unchanged 3 degree joint margin',
            successful_scripted_jump_demonstrations=0,failed_scripts_not_used_as_success_targets=True)
    record = out/'run_provenance.json'
    record.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    try:
        run_train(task, cfg, out)
        files = sorted(out.glob('model_*.pt'), key=lambda x: int(x.stem.split('_')[-1]))
        final = files[-1]
        result = run_export(task, ExportConfig(checkpoint_file=str(final), onnx_file=str(out/'candidate.onnx'), num_envs=1))
        session = ort.InferenceSession(str(result.onnx_path), providers=['CPUExecutionProvider'])
        output = session.run(None, {'obs': np.zeros((1, 81), dtype=np.float32)})[0]
        assert output.shape == (1, 19) and np.isfinite(output).all()
        provenance.update(status='COMPLETE', final_checkpoint=str(final), onnx=str(result.onnx_path),
            final_sha256=sha(final), onnx_sha256=sha(result.onnx_path), finished_unix=time.time())
    except Exception as exc:
        provenance.update(status='FAILED', error=str(exc)); raise
    finally:
        record.write_text(json.dumps(provenance, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
