"""Guard against the get-up envelope drifting between training and the evaluator.

These two paths disagreed once already: the evaluator enforced ankle +/-51 deg and
jaw +0.04 but no neck bound, while the CAD clearance gate requires neck_pitch to
stay under ~38.5 deg. The released candidate then drove neck_pitch to 56.8 deg and
pushed the head-roll case 6839 mm^3 into the battery without the gate noticing.

Run this before trusting any get-up acceptance matrix.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mjlab_microduck.tasks import recovery_bounds  # noqa: E402


def main():
    from mjlab_microduck.tasks import mdp
    checks = {
        'NECK_BATTERY_CLEARANCE_DEG': (mdp.NECK_BATTERY_CLEARANCE_DEG, recovery_bounds.NECK_PITCH_MAX_DEG),
        'NECK_BATTERY_TERMINATE_DEG': (mdp.NECK_BATTERY_TERMINATE_DEG, recovery_bounds.NECK_BATTERY_TERMINATE_DEG),
    }
    ok = True
    for name, (training, evaluator) in checks.items():
        match = float(training) == float(evaluator)
        ok &= match
        print(f'{"OK  " if match else "FAIL"} {name}: training={training} evaluator={evaluator}')

    # The numpy envelope and the training clamp must select the same joint rows.
    names = ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
             'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll', 'jaw_hinge',
             'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle',
             'tail_yaw', 'tail_pitch', 'arm_l', 'arm_r']
    raw = np.zeros((2, 19))
    raw[0, names.index('left_ankle')] = 5.
    raw[1, names.index('left_ankle')] = -5.
    raw[0, names.index('right_ankle')] = 5.
    raw[0, names.index('neck_pitch')] = 5.
    raw[0, names.index('jaw_hinge')] = -5.
    out = recovery_bounds.apply_numpy(raw, names)
    expect = {
        'left_ankle': np.deg2rad(recovery_bounds.ANKLE_LIMIT_DEG),
        'right_ankle': np.deg2rad(recovery_bounds.ANKLE_LIMIT_DEG),
        'neck_pitch': np.deg2rad(recovery_bounds.NECK_PITCH_MAX_DEG),
        'jaw_hinge': recovery_bounds.JAW_TARGET_RAD,
        'left_knee': 0.,
    }
    for joint, want in expect.items():
        got = out[0, names.index(joint)]
        match = np.isclose(got, want)
        ok &= bool(match)
        print(f'{"OK  " if match else "FAIL"} apply_numpy {joint}: {got:.6f} expected {want:.6f}')
    # Values inside the envelope must pass through untouched.
    inside = np.zeros((1, 19))
    inside[0, names.index('neck_pitch')] = np.deg2rad(20.)
    unchanged = np.isclose(recovery_bounds.apply_numpy(inside, names)[0, names.index('neck_pitch')],
                           np.deg2rad(20.))
    ok &= bool(unchanged)
    print(f'{"OK  " if unchanged else "FAIL"} in-envelope neck_pitch is preserved')
    print('CONSISTENT' if ok else 'INCONSISTENT')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
