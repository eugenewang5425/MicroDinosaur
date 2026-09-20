"""Small declared development set: v7 seed 0 only; keep every trial."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
from head_attitude_sim import HeadExperiment
from head_attitude import HeadConfig
from hardware_sim import HardwareCase
from evaluate_policy import sha

ROOT = Path(__file__).parent
OUT = ROOT/'20260914_head_attitude/development'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = HeadConfig(nominal_target_filter_tau_s=0.)
    configurations = [('off', False, base), ('p1', True, base),
        ('p2', True, replace(base, kp=(.8, 1.5, 3.))),
        ('p3', True, replace(base, kp=(1., 2., 4.), kd=(.02, .04, .06))),
        ('no_predict', True, replace(base, prediction_limit_s=0.))]
    sources = {p.name: sha(p) for p in [ROOT/'head_attitude.py', ROOT/'head_attitude_sim.py', ROOT/'evaluate_policy.py']}
    (OUT/'plan.json').write_text(json.dumps({'seed': 0, 'policy': 'v7', 'scenario': 'straight',
        'seconds': 12, 'configurations': [(n, on, asdict(c)) for n, on, c in configurations],
        'sources': sources}, indent=2), encoding='utf-8')
    for name, enabled, config in configurations:
        dest = OUT/f'{name}.json'
        assert not dest.exists(), 'Preserve development evidence'
        e = HeadExperiment(ROOT/'20260913_handoff/native_v07', ROOT/'20260913_handoff/v7_reference.onnx',
            HardwareCase(physics_dt=.00125), head_enabled=enabled, head_config=config)
        m, a = e.run('straight', 'imu', 0, 12., True)
        dest.write_text(json.dumps(m, indent=2), encoding='utf-8')
        np.savez_compressed(OUT/f'{name}.npz', trace=a, head=np.array(e.head_rows))
        print(json.dumps({'case': name, **{k: m[k] for k in ('camera_heading_error_rms_deg',
            'camera_heading_detrended_rms_deg', 'camera_yaw_rate_error_rms_deg_s', 'camera_pitch_rms_deg',
            'camera_roll_rms_deg', 'body_vx_mean_m_s', 'heading_error_rms_deg', 'fell')}}), flush=True)


if __name__ == '__main__':
    main()
