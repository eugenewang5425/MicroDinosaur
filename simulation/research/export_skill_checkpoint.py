"""Official normalized export for a named, already-written checkpoint."""
import os
os.environ['WANDB_MODE'] = 'disabled'
os.environ['HF_HUB_OFFLINE'] = '1'
import argparse
from pathlib import Path
import json
from mjlab_microduck.export import ExportConfig, run_export
from evaluate_policy import sha


def main():
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    before = sha(args.checkpoint)
    result = run_export('Mjlab-Velocity-Flat-MicroDinosaur', ExportConfig(checkpoint_file=str(args.checkpoint),
        onnx_file=str(args.out/'candidate.onnx'), num_envs=1))
    assert sha(args.checkpoint) == before
    (args.out/'provenance.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint), checkpoint_sha256=before,
        onnx_sha256=sha(result.onnx_path), official_normalized_export=True), indent=2), encoding='utf-8')


if __name__ == '__main__': main()
