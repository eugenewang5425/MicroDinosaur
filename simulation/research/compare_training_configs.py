"""Audit saved training parameters; report every difference, never just weights."""
import argparse
import hashlib
import json
from pathlib import Path
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--control', required=True)
parser.add_argument('--treatment', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
control, treatment = Path(args.control), Path(args.treatment)
differences = []


def compare(a, b, path):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            compare(a.get(key), b.get(key), path + '.' + key)
    elif a != b:
        differences.append({'path': path, 'control': a, 'treatment': b})


for name in ('agent.yaml', 'env.yaml'):
    compare(yaml.load((control/'params'/name).read_text(), Loader=yaml.BaseLoader),
            yaml.load((treatment/'params'/name).read_text(), Loader=yaml.BaseLoader), name)
ca = json.loads((control/'run_provenance.json').read_text())
tr = json.loads((treatment/'run_provenance.json').read_text())
allowed = {'agent.yaml.run_name'}
unexpected = [d for d in differences if d['path'] not in allowed]
report = {'parameter_differences': differences, 'unexpected_parameter_differences': unexpected,
          'same_source_checkpoint': ca['source_checkpoint_sha256'] == tr['source_checkpoint_sha256'],
          'same_robot_xml': ca['model_xml_sha256'] == tr['model_xml_sha256'],
          'same_iterations': ca['iterations'] == tr['iterations'],
          'same_num_envs': ca['envs'] == tr['envs'],
          'reward_implementation_change': 'per-environment history reset and deterministic head HOME',
          'implementation_diff': 'reward_implementation.patch'}
assert not unexpected, unexpected
assert all(report[k] for k in ('same_source_checkpoint', 'same_robot_xml', 'same_iterations', 'same_num_envs'))
report['status'] = 'PASS'
Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
