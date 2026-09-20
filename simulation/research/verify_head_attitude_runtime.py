"""Explicit follow-up: nominal equivalence and release on slow IMU data."""
import json
from pathlib import Path
import numpy as np
from head_attitude_runtime import make_operating_experiment
from heading_sim import ImuTransportConfig
from evaluate_policy import sha

OUT = Path(__file__).parent/'20260914_head_attitude/runtime_followup'


def main():
    OUT.mkdir(exist_ok=False)
    jobs = [(policy, case, mode) for policy in ('v7', 's42_no_neck')
        for case in ('nominal', 'slow') for mode in ('filter_only', 'imu')]
    jobs += [('v7', 'outage', 'imu')]
    (OUT/'plan.json').write_text(json.dumps(dict(reason='Frozen 40ms/60ms transport increased yaw rate; release feedback above 40ms age',
        jobs=jobs, unchanged_nominal_gains=True, runtime_source_sha256=sha(Path(__file__).parent/'head_attitude_runtime.py')), indent=2), encoding='utf-8')
    rows = {}; traces = {}; verdicts = {}
    for policy, case, mode in jobs:
        transport = (ImuTransportConfig(period_ms=40, latency_ms=60) if case == 'slow' else
                     ImuTransportConfig(outage_duration_s=.2) if case == 'outage' else ImuTransportConfig())
        e = make_operating_experiment(policy, mode, transport=transport)
        m, a = e.run('straight', 'imu', 1, 12., True)
        name = '__'.join((policy, case, mode)); rows[name] = m; traces[name] = a
        (OUT/f'{name}.json').write_text(json.dumps(m, indent=2), encoding='utf-8')
        np.savez_compressed(OUT/f'{name}.npz', trace=a, head=np.array(e.head_rows))
        if case == 'nominal':
            original = json.loads((OUT.parent/'trials'/f'nominal__{policy}__1__straight__{mode}.json').read_text())['metrics']
            current = json.loads(json.dumps(m)); current.pop('head_config'); original.pop('head_config')
            assert original == current
            with np.load(OUT.parent/'traces'/f'nominal__{policy}__1__straight__{mode}.npz') as old:
                np.testing.assert_array_equal(a, old['trace'])
            verdicts[name] = 'All nominal metrics and trajectory unchanged (age threshold only)'
        if case == 'slow' and mode == 'imu':
            np.testing.assert_array_equal(a, traces[f'{policy}__slow__filter_only'])
            assert m['head_feedback_stale_fraction'] == 1.
            verdicts[name] = 'Head IMU correction disabled; trajectory equals filter-only fallback'
        assert not m['fell'] and m['head_target_slew_max_rad_s'] <= 4.00001
        print(json.dumps(dict(trial=name, camera_rms=m['camera_heading_error_rms_deg'],
            yaw_rate=m['camera_yaw_rate_error_rms_deg_s'], stale=m['head_feedback_stale_fraction'])), flush=True)
    report = dict(status='PASS', completed=len(rows), falls=0, nominal_equivalence=True,
        slow_fallback_equivalence=True, verdicts=verdicts)
    (OUT/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
