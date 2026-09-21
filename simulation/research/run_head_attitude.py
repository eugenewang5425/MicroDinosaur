"""Frozen confirmation matrix: no training, no parameter changes after seed 0."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import time
import numpy as np
from head_attitude_sim import HeadExperiment, HEAD_TRACE_COLUMNS
from head_attitude import HeadConfig
from heading_sim import ImuTransportConfig, TRACE_COLUMNS
from hardware_sim import HardwareCase
from run_heading_stable_start import CALIBRATOR, POLICY
from evaluate_policy import sha

ROOT = Path(__file__).parent
OUT = ROOT/'20260914_head_attitude'
POLICIES = {'v7': CALIBRATOR, 's42_no_neck': POLICY}


def config_for(mode):
    config = HeadConfig()
    if mode == 'filter_only':
        config = replace(config, kp=(0., 0., 0.), kd=(0., 0., 0.))
    return config


def make_experiment(policy, mode, hardware=None, transport=None, head_motion='hold'):
    return HeadExperiment(ROOT/'20260913_handoff/native_v07', POLICIES[policy],
        hardware or HardwareCase(physics_dt=.00125), transport=transport,
        head_enabled=mode != 'off', head_config=config_for(mode), head_motion=head_motion)


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'plan.json').exists(), 'Never overwrite a frozen confirmation matrix'
    for name in ('trials', 'traces', 'source_snapshot'):
        (OUT/name).mkdir(exist_ok=True)
    source_names = ['head_attitude.py', 'head_attitude_sim.py', 'run_head_attitude.py', 'test_head_attitude.py',
        'imu_heading.py', 'heading_sim.py', 'run_heading_stable_start.py', 'hardware_sim.py',
        'evaluate_policy.py', 'evaluate_gaze_ablation.py']
    sources = {name: sha(ROOT/name) for name in source_names}
    for name in source_names:
        shutil.copyfile(ROOT/name, OUT/'source_snapshot'/name)
    jobs = []
    def add(case, policy, seed, scenario, mode, hardware=None, transport=None, head_motion='hold'):
        jobs.append(dict(case=case, policy=policy, seed=seed, scenario=scenario, mode=mode,
            seconds=8 if scenario == 'stand' else 12, hardware=asdict(hardware or HardwareCase(physics_dt=.00125)),
            transport=asdict(transport or ImuTransportConfig()), head_motion=head_motion))
    for policy in POLICIES:
        for seed in (1, 2, 3):
            for mode in ('off', 'filter_only', 'imu'):
                add('nominal', policy, seed, 'straight', mode)
            for scenario in ('stand', 'left_then_hold', 'right_then_hold'):
                for mode in ('off', 'imu'):
                    add('nominal', policy, seed, scenario, mode)
    probes = [
        ('physics_0p625ms', HardwareCase(physics_dt=.000625), ImuTransportConfig()),
        ('motor_5ms', HardwareCase(physics_dt=.00125, command_ms=5), ImuTransportConfig()),
        ('motor_15ms', HardwareCase(physics_dt=.00125, command_ms=15), ImuTransportConfig()),
        ('position_40ms', HardwareCase(physics_dt=.00125, position_ms=40), ImuTransportConfig()),
        ('imu_40_60ms', HardwareCase(physics_dt=.00125), ImuTransportConfig(period_ms=40, latency_ms=60)),
        ('imu_outage_200ms', HardwareCase(physics_dt=.00125), ImuTransportConfig(outage_duration_s=.2)),
        ('imu_residual_0p5deg_s', HardwareCase(physics_dt=.00125), ImuTransportConfig(residual_z_bias_deg_s=.5)),
        ('head_mass_110pct', HardwareCase(physics_dt=.00125, head_mass_scale=1.1), ImuTransportConfig())]
    for label, hardware, transport in probes:
        for mode in ('off', 'imu'):
            add(label, 'v7', 1, 'straight', mode, hardware, transport)
    for motion in ('yaw_scan', 'pitch_scan', 'roll_scan'):
        add(motion, 'v7', 1, 'straight', 'imu', head_motion=motion)
    plan = {'development': 'v7 seed 0, 11 recorded trials; 100 ms head target filter chosen before confirmation',
        'head_config': asdict(HeadConfig()), 'sources': sources, 'jobs': jobs,
        'primary': 'nominal straight, 3 paired initial states; heading/amplitude/yaw-rate/pitch/speed',
        'causal_control': 'filter_only has identical filter/limits but zero head IMU feedback gains',
        'policy_sha256': {label: sha(path) for label, path in POLICIES.items()},
        'plant_sha256': sha(ROOT/'20260913_handoff/native_v07/nominal.mjb'),
        'production_sha256': sha(ROOT.parent/'policies/head_candidate.onnx')}
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    state = dict(status='RUNNING', completed=0, rejected=0, total=len(jobs), started_unix=time.time())
    records = []
    def save():
        (OUT/'status.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    save()
    try:
        for job in jobs:
            key = '__'.join(str(job[k]) for k in ('case', 'policy', 'seed', 'scenario', 'mode'))
            e = make_experiment(job['policy'], job['mode'], HardwareCase(**job['hardware']),
                                ImuTransportConfig(**job['transport']), job['head_motion'])
            try:
                m, trace = e.run(job['scenario'], 'imu', job['seed'], job['seconds'], True)
            except ValueError as exc:
                if not str(exc).startswith('Calibration '):
                    raise
                record = dict(job=job, status='CALIBRATION_REJECTED', error=str(exc))
                state['rejected'] += 1
            else:
                record = dict(job=job, status='COMPLETE', metrics=m)
                np.savez_compressed(OUT/'traces'/f'{key}.npz', trace=trace, columns=np.array(TRACE_COLUMNS),
                    head=np.array(e.head_rows), head_columns=np.array(HEAD_TRACE_COLUMNS),
                    gaze=np.asarray(e.sim.gaze_trace)[-round(job['seconds']/e.sim.model.opt.timestep):],
                    physical=np.array(e.head_physics))
            (OUT/'trials'/f'{key}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            records.append(record); state['completed'] = len(records); save()
            print(json.dumps(dict(completed=len(records), key=key, status=record['status'],
                yaw_rms=record.get('metrics', {}).get('camera_heading_error_rms_deg'),
                yaw_rate=record.get('metrics', {}).get('camera_yaw_rate_error_rms_deg_s'),
                fell=record.get('metrics', {}).get('fell'))), flush=True)
            assert all(sha(ROOT/name) == digest for name, digest in sources.items()), 'Frozen source changed'
        (OUT/'matrix.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
        state.update(status='COMPLETE', finished_unix=time.time()); save()
    except Exception as exc:
        state.update(status='FAILED', error=str(exc)); save(); raise


if __name__ == '__main__':
    main()
