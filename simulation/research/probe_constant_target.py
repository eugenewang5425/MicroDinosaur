"""Diagnostic only: does standing jitter persist without changing policy targets?"""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate_policy import Sim, run_trial

parser = argparse.ArgumentParser()
parser.add_argument('--plant', required=True)
parser.add_argument('--onnx', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
reference = Sim(args.plant, args.onnx, command_lag=2)
actions = []
for i in range(350):
    reference.step(np.zeros(18))
    if i >= 250:
        actions.append(reference.raw.copy())
mean_action = np.mean(actions, axis=0).astype(np.float32)


class ConstantPolicy:
    def __init__(self, session, action):
        self.schema = session.get_inputs()
        self.action = action[None]

    def get_inputs(self):
        return self.schema

    def run(self, *_args, **_kwargs):
        return [self.action.copy()]


results = []
for lag in (0, 2):
    for label, action in [('active_policy', None), ('fixed_HOME', np.zeros(19, dtype=np.float32)),
                          ('fixed_mean_standing_target', mean_action)]:
        sim = Sim(args.plant, args.onnx, command_lag=lag)
        if action is not None:
            sim.session = ConstantPolicy(sim.session, action)
        row = run_trial(sim, np.zeros(18), seconds=5)
        row.update(mode=label, command_ms=lag*5, feedback_ms=0)
        results.append(row)
Path(args.out).write_text(json.dumps({'mean_raw_target': mean_action.tolist(), 'trials': results,
    'scope': 'One unperturbed flat-ground initial condition; constant targets are a diagnostic, not a walking policy.'}, indent=2), encoding='utf-8')
for r in results:
    print(r['mode'], r['command_ms'], {k: r[k] for k in ('head_angular_speed_rms_deg_s', 'tilt_max_deg', 'fell')})
