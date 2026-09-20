"""One bounded simulation with the frozen head IMU controller; no hardware IO."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from head_attitude_sim import HEAD_TRACE_COLUMNS
from heading_sim import TRACE_COLUMNS
from run_head_attitude import POLICIES
from head_attitude_runtime import make_operating_experiment
from evaluate_policy import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', choices=POLICIES, default='v7')
    parser.add_argument('--head-mode', choices=('off', 'filter_only', 'imu'), default='imu')
    parser.add_argument('--scenario', choices=('stand', 'straight', 'left_then_hold', 'right_then_hold', 's_turn'), default='straight')
    parser.add_argument('--head-motion', choices=('hold', 'yaw_scan', 'pitch_scan', 'roll_scan'), default='hold')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--seconds', type=float, default=12.)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 120 or args.seed < 0:
        parser.error('seconds must be 1..120; seed must be nonnegative')
    if args.out.exists():
        parser.error('Use a new output directory to preserve previous evidence')
    e = make_operating_experiment(args.policy, args.head_mode, head_motion=args.head_motion)
    metrics, trace = e.run(args.scenario, 'imu', args.seed, args.seconds, True)
    args.out.mkdir(parents=True)
    record = {'metrics': metrics, 'policy_sha256': sha(POLICIES[args.policy]),
              'head_mode': args.head_mode, 'hardware': asdict(e.sim.case), 'transport': asdict(e.transport)}
    (args.out/'result.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    np.savez_compressed(args.out/'trace.npz', trace=trace, columns=np.array(TRACE_COLUMNS),
        head=np.array(e.head_rows), head_columns=np.array(HEAD_TRACE_COLUMNS))
    print(json.dumps({k: metrics[k] for k in ('camera_heading_error_rms_deg',
        'camera_heading_detrended_rms_deg', 'camera_yaw_rate_error_rms_deg_s', 'body_vx_mean_m_s', 'fell')}, indent=2))


if __name__ == '__main__':
    main()
