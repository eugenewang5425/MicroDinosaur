"""Record observed learning signals without inferring a causal reward ablation."""
import json
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
OUT = ROOT / '20260914_foot_flight'
RUN = Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_foot_flight/20260914_train_512x401')


def main():
    event = EventAccumulator(str(RUN), size_guidance={'scalars': 0})
    event.Reload()
    result = {}
    for tag in event.Tags()['scalars']:
        if not any(name in tag for name in ['foot_clearance', 'foot_duration', 'foot_goal',
                'air_tuck', 'true_foot', 'deep_pose', 'hop_prepare', 'hop_land', 'Train/mean_reward']):
            continue
        if tag.endswith('/time'):
            continue
        records = event.Scalars(tag)
        values = np.array([r.value for r in records])
        assert len(values) == 401 and np.isfinite(values).all()
        positive = np.flatnonzero(values > 1e-10)
        result[tag] = dict(count=len(records), positive_updates=len(positive),
            last_positive_update=int(positive[-1] + 1) if len(positive) else None,
            last10_mean=float(values[-10:].mean()), max=float(values.max()),
            blocks=[dict(start=i+1, end=min(i+50,len(values)), mean=float(values[i:i+50].mean()))
                    for i in range(0,len(values),50)])
    result['interpretation'] = {
        'observed': 'Strict goal reward is zero in all 401 updates. Sparse clearance persists late; final 10 updates have zero clearance, duration, tuck and landing rewards.',
        'hypothesis': 'Scheduled deepening plus dense supported-pose rewards may favor grounded crouch/extension before launch coordination is learned.',
        'causal_ablation_performed': False,
        'airborne_tuck_learned': False,
        'next_curriculum': 'Require repeatable true foot flight before increasing crouch depth; validate takeoff-to-tuck timing through unchanged physical actuation.',
        'elastic_energy_storage_validated': False,
    }
    a = np.load(OUT / 'script_probes/008.npz')['physics']
    pre = a[(a[:, 0] >= 1.3) & (a[:, 0] < 1.8)]
    result['reference_hold'] = dict(source='script_probes/008.npz',
        interval_s=[1.3, 1.8], requested_depth_mm=50,
        actual_depth_mm=float((a[0, 1] - pre[:, 1].mean()) * 1000),
        root_height_mm=float(pre[:, 1].mean() * 1000),
        both_supported_fraction=float(((pre[:, 5] > .1) & (pre[:, 6] > .1)).mean()),
        max_tilt_deg=float(pre[:, 8].max()), max_joint_excess_rad=float(pre[:, 13].max()),
        following_jump_succeeded=False, learned_policy=False)
    (OUT / 'learning_diagnostic.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['interpretation'], indent=2))


if __name__ == '__main__':
    main()
