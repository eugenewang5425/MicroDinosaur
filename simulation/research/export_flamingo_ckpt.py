"""从训练中途的 checkpoint 导出 ONNX(flamingo 配置没注册任务名, 只能传 task 对象)。

用法:
  python export_flamingo_ckpt.py --checkpoint 20260919_dualpose/model_16780.pt \
      --out 20260919_dualpose/mid_16780
"""
import os
os.environ['WANDB_MODE'] = 'disabled'
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
import argparse
from pathlib import Path

import json

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from mjlab_microduck.export import ExportConfig, run_export  # noqa: E402
from flamingo_cfg import build_config  # noqa: E402
from evaluate_policy import sha  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    before = sha(a.checkpoint)
    task, cfg = build_config(envs=1, seed=47)
    result = run_export(task, ExportConfig(checkpoint_file=str(a.checkpoint),
                                           onnx_file=str(a.out / 'candidate.onnx'),
                                           num_envs=1))
    assert sha(a.checkpoint) == before, '导出不该改动 checkpoint'
    (a.out / 'provenance.json').write_text(json.dumps(dict(
        checkpoint=str(a.checkpoint), checkpoint_sha256=before,
        onnx_sha256=sha(result.onnx_path)), indent=2), encoding='utf-8')
    print(f'[export] {result.onnx_path}', flush=True)


if __name__ == '__main__':
    main()
