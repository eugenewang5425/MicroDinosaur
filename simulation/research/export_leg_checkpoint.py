"""把单脚站立(或台阶)训练的 checkpoint 导出成 ONNX,供 CPU 评估使用。

为什么需要独立脚本:两个训练脚本只在 run_train 正常返回后才导出;而本机
MuJoCo Warp 会偶发 Non-finite(守护脚本已能自动续训),用最新 checkpoint
先验收不需要等整轮跑完。
"""
import os
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import argparse
from copy import deepcopy
from pathlib import Path
import sys

from mjlab.tasks.registry import register_mjlab_task, load_runner_cls
from mjlab_microduck.export import run_export, ExportConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--kind', choices=('single_leg', 'stair'), default='single_leg')
    ap.add_argument('--target-height', type=float, default=.025)
    ap.add_argument('--lift-foot', default='left')
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    if a.kind == 'single_leg':
        from single_leg_cfg import build_config
        task, cfg = build_config(1, 47, target_height=a.target_height,
                                 lift_foot=a.lift_foot)
    else:
        from stair_climb_cfg import build_config
        task, cfg = build_config(1, 47)
    runner = load_runner_cls(task)
    name = 'Mjlab-Export-SingleLeg'
    register_mjlab_task(name, cfg.env, deepcopy(cfg.env), cfg.agent, runner)
    result = run_export(name, ExportConfig(checkpoint_file=str(a.checkpoint),
                                           onnx_file=str(a.output), num_envs=1))
    print(f'[export] {result.onnx_path}')


if __name__ == '__main__':
    main()
