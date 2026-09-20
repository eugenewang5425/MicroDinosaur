"""Frozen identities for the corrective demonstration study."""
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_corrective'
PREVIOUS=ROOT/'20260914_demonstrations'
RUNS=Path('D:/microduck_rl/logs/rsl_rl')
SOURCE=RUNS/'microdinosaur_demo_demonstration/20260914_train_512x101/model_16250.pt'
STUDENT=SOURCE.with_name('candidate.onnx')
STOP_TEACHER=RUNS/'microdinosaur_demo_archive/20260914_train_512x101/candidate.onnx'
OLD=ROOT/'20260914_transition_refine/v7_anchor.npz'

