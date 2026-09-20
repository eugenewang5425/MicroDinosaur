"""Explicit frozen-policy filter interventions; these are not matched retrained policies."""
import argparse
import json
from pathlib import Path
import numpy as np
from delay_matrix import TracedSim, trial

parser = argparse.ArgumentParser()
parser.add_argument('--plant', required=True)
parser.add_argument('--onnx', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
results = []
for alpha in (.9, .7, .5, .3):
    for command_lag, obs_lag in ((0, 0), (0, 1), (2, 0), (2, 1)):
        sim = TracedSim(args.plant, args.onnx, command_lag, obs_lag)
        sim.contract['action_filter']['alpha_new'] = alpha
        row = {'alpha_new': alpha, 'command_ms': command_lag*5,
               'position_ms': obs_lag*20, 'velocity_ms': obs_lag*20,
               'policy_retrained_for_this_filter': alpha == .9, 'trials': {}}
        for name, vx, seconds in [('stand', 0., 5), ('forward', .55, 6)]:
            command = np.zeros(18); command[0] = vx
            row['trials'][name] = [trial(sim, command, seed, seconds) for seed in range(3)]
        results.append(row)
        print(alpha, command_lag*5, obs_lag*20,
              'head', round(np.mean([t['head_angular_speed_rms_deg_s'] for t in row['trials']['stand']]), 2),
              'yaw', round(np.mean([abs(t['yaw_drift_deg']) for t in row['trials']['stand']]), 2),
              'vx', round(np.mean([t['vx_body_m_s'] for t in row['trials']['forward']]), 3),
              'dy', round(np.mean([abs(t['lateral_displacement_mm']) for t in row['trials']['forward']]), 1), flush=True)
Path(args.out).write_text(json.dumps(results, indent=2), encoding='utf-8')
