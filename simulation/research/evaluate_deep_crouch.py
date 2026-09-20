"""Matched CPU 12-second cycles with verified v7 warmup and both IMU loops."""
import argparse
import json
from pathlib import Path
import numpy as np
from terrain_skill_eval import TerrainSkillExperiment
from hardware_sim import HardwareCase
from heading_sim import WARMUP_SECONDS
from evaluate_policy import sha

OUT = Path(__file__).parent/'20260914_deep_crouch'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', required=True, type=Path)
    p.add_argument('--label', required=True)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--delay-coverage', action='store_true')
    a = p.parse_args()
    dest = OUT/'evaluation'/a.label; dest.mkdir(parents=True, exist_ok=False)
    cases = [(depth, 'stand', 10) for depth in (0, 20, 30, 40)]
    cases += [(0, 'straight', 10), (30, 'slow', 10)]
    if a.delay_coverage: cases += [(depth, 'stand', lag) for lag in (5, 15) for depth in (30, 40)]
    records = []
    for depth, scenario, lag in cases:
        for seed in range(1, a.seeds+1):
            name = f'd{depth}_{scenario}_lag{lag}_s{seed}'
            e = TerrainSkillExperiment(a.policy, posture=f'crouch{depth}' if depth else 'none',
                walking_speed=.2 if scenario == 'slow' else None,
                hardware_case=HardwareCase(physics_dt=.00125, command_ms=lag))
            try:
                m, trace = e.run('straight' if scenario == 'slow' else scenario, 'imu', seed, 12., True)
                full = np.asarray(e.sim.gaze_trace); t = full[:, 0]-WARMUP_SECONDS
                initial = full[(t>=.5)&(t<1.8)]; plateau = full[(t>=4.2)&(t<6.5)]; returned = full[(t>=9.5)&(t<12.)]
                initial_height = float(initial[:, 3].mean())
                actual_drop = (initial_height-float(plateau[:, 3].mean()))*1000
                return_error = (float(returned[:, 3].mean())-initial_height)*1000
                hold_drift = float(np.linalg.norm(returned[-1, 1:3]-returned[0, 1:3])/(returned[-1, 0]-returned[0, 0]))
                # Cycle acceptance requires depth AND stable return, not only no fall.
                success = not m['fell'] and abs(actual_drop-depth)<=6 and abs(return_error)<=5
                if scenario == 'stand': success = success and hold_drift <= .01
                m.update(initial_height_mm=initial_height*1000, actual_drop_mm=actual_drop,
                    return_height_error_mm=return_error, returned_drift_m_s=hold_drift, cycle_success=bool(success))
                record = dict(status='COMPLETE', name=name, depth_mm=depth, scenario=scenario, motor_delay_ms=lag,
                    seed=seed, policy_sha256=sha(a.policy), metrics=m)
                np.savez_compressed(dest/(name+'.npz'), trace=trace, gaze=full, commands=np.array(e.command_rows),
                    head_controls=np.asarray(e.head_rows), head_physics=np.asarray(e.head_physics))
            except ValueError as exc:
                record = dict(status='FAILED', name=name, error=str(exc), policy_sha256=sha(a.policy))
            records.append(record)
            (dest/(name+'.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
            print(json.dumps({k: record.get('metrics', {}).get(k) for k in ('actual_drop_mm', 'return_height_error_mm', 'returned_drift_m_s', 'cycle_success', 'fell')} | {'name': name}), flush=True)
    (dest/'matrix.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
