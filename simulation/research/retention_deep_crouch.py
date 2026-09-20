"""Retain 10mm steps and lifted-foot turning; keep all cases including falls."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate_skill_turns import run
from terrain_skill_eval import TerrainSkillExperiment
from evaluate_policy import sha
from evaluate_deep_crouch import OUT


def main():
    p = argparse.ArgumentParser(); p.add_argument('--policy', required=True, type=Path)
    a = p.parse_args(); dest = OUT/'retention'; dest.mkdir(exist_ok=False)
    records = []
    for scenario in ('stand', 'left_then_hold', 'right_then_hold'):
        for seed in (1, 2, 3):
            metrics, trace = run(a.policy, scenario, seed)
            name = f'{scenario}_s{seed}'
            record = dict(kind='turns', name=name, metrics=metrics, policy_sha256=sha(a.policy))
            records.append(record); np.savez_compressed(dest/(name+'.npz'), feet=trace)
            (dest/(name+'.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
            print(json.dumps(record), flush=True)
    for seed in (1, 2, 3):
        e = TerrainSkillExperiment(a.policy, terrain='steps_10')
        metrics, trace = e.run('straight', 'imu', seed, 12., True)
        name = f'steps10_s{seed}'
        record = dict(kind='terrain', name=name, metrics=metrics, policy_sha256=sha(a.policy))
        records.append(record); np.savez_compressed(dest/(name+'.npz'), trace=trace, gaze=np.array(e.sim.gaze_trace))
        (dest/(name+'.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(record), flush=True)
    (dest/'matrix.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
