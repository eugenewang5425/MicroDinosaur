"""Independent result, bounds, paired controls and startup/transient audit."""
import json
from pathlib import Path
import numpy as np
from evaluate_policy import sha

ROOT = Path(__file__).parent
OUT = ROOT/'20260914_head_attitude'


def main():
    plan = json.loads((OUT/'plan.json').read_text())
    state = json.loads((OUT/'status.json').read_text())
    assert state['status'] == 'COMPLETE'
    records = [json.loads(p.read_text()) for p in sorted((OUT/'trials').glob('*.json'))]
    assert len(records) == len(plan['jobs'])
    assert all(sha(ROOT/name) == digest == sha(OUT/'source_snapshot'/name) for name, digest in plan['sources'].items())
    assert sha('D:/microduck_rl/microdinosaur_p2.onnx') == plan['production_sha256']
    for label, path in [('v7', ROOT/'20260913_handoff/v7_reference.onnx'),
                        ('s42_no_neck', Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_v07_calibration/20260914_gaze_train_s42_no_neck_cost_1024x101/candidate.onnx'))]:
        assert sha(path) == plan['policy_sha256'][label]
    def key(job):
        return '__'.join(str(job[k]) for k in ('case', 'policy', 'seed', 'scenario', 'mode'))
    by_key = {key(r['job']): r for r in records}
    assert set(by_key) == {key(j) for j in plan['jobs']}
    improvements = {}; settled_stand = {}; scans = {}
    for row in records:
        if row['status'] != 'COMPLETE':
            continue
        job = row['job']; m = row['metrics']
        a = np.load(OUT/'traces'/f'{key(job)}.npz')
        head, trace, gaze, physical = [a[k] for k in ('head', 'trace', 'gaze', 'physical')]
        assert all(np.isfinite(x).all() for x in (head, trace, gaze, physical))
        assert m['command_wz_max_rad_s'] <= .70001
        assert m['head_joint_torque_peak_nm'] <= .60001
        if job['mode'] != 'off':
            assert m['head_target_slew_max_rad_s'] <= 4.00001
            assert np.all(np.max(abs(head[:, 4:7]), axis=0) <= np.array([.35, .5, .18])+1e-7)
        if job['case'] == 'nominal' and job['scenario'] == 'straight' and job['mode'] == 'imu':
            base = by_key[key({**job, 'mode': 'off'})]['metrics']
            filt = by_key[key({**job, 'mode': 'filter_only'})]['metrics']
            assert m['calibration'] == base['calibration'] == filt['calibration']
            primary = ('camera_heading_error_rms_deg', 'camera_heading_detrended_rms_deg', 'camera_yaw_rate_error_rms_deg_s')
            improvements[key(job)] = {k: dict(off=base[k], filter_only=filt[k], imu=m[k],
                better_than_both=bool(m[k] < min(base[k], filt[k]))) for k in primary}
        if job['case'] == 'nominal' and job['scenario'] == 'stand':
            settled = gaze[:, 0] >= 8.  # Skip first two seconds AFTER calibration.
            settled_stand[key(job)] = dict(camera_yaw_rate_after_2s_rms_deg_s=float(np.rad2deg(np.sqrt(np.mean(gaze[settled, 7]**2)))),
                camera_yaw_periodic_after_2s_rms_deg=float(np.rad2deg(np.std(np.unwrap(gaze[settled, 5])))))
        if job['head_motion'] != 'hold':
            start = gaze[0, 4]; t = gaze[:, 0]-6.
            plateau = (t >= 4)&(t <= 5.5)
            late = (t >= 10)
            values = dict(yaw=np.unwrap(gaze[:, 5])-start, pitch=-gaze[:, 6], roll=physical[:, 1])
            axis = job['head_motion'].split('_')[0]; value = values[axis]
            scans[axis] = dict(requested_deg={'yaw': float(np.rad2deg(.25)), 'pitch': float(np.rad2deg(.12)), 'roll': float(np.rad2deg(.10))}[axis],
                plateau_mean_deg=float(np.rad2deg(np.mean(value[plateau]))),
                returned_mean_deg=float(np.rad2deg(np.mean(value[late]))))
    cli = json.loads((OUT/'cli_check/result.json').read_text())['metrics']
    assert cli == by_key['nominal__v7__1__straight__imu']['metrics']
    test_bytes = (OUT/'tests.log').read_bytes()
    tests = test_bytes.decode('utf-16' if test_bytes.startswith(b'\xff\xfe') else 'utf-8')
    assert 'Ran 14 tests' in tests and 'OK' in tests
    result = dict(status='PASS', attempts=len(records), completed=sum(r['status'] == 'COMPLETE' for r in records),
        rejected=sum(r['status'] != 'COMPLETE' for r in records), falls=sum(r.get('metrics', {}).get('fell', False) for r in records),
        frozen_sources_match=True, production_and_reference_policies_unchanged=True, all_traces_finite=True,
        head_torque_and_command_bounds_pass=True, tests_passed=14, cli_matches_confirmation=True,
        primary_improvements=improvements, stand_after_transient=settled_stand, explicit_head_commands=scans)
    (OUT/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
