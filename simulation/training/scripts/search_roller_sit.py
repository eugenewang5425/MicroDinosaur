#!/usr/bin/env python3
"""Search for a STABLE seated keyframe on the roller model (CPU, no GPU).

The walker SIT keyframe (flat feet, knee ±1.35) collapses on rollers: the
blades+wheels cannot brace the way a flat sole does (measured: 100° tilt, 5%
survival). This script sweeps symmetric leg poses under a firmware-strength PD
hold and reports which ones settle upright — the candidate SIT keyframe for the
roller sit-stand task.

Criteria (same bar as the walker keyframe, per AGENTS.md): tilt < 20° after a
4 s hold from a small noisy drop, on the majority of seeds.

Usage:
    uv run python scripts/search_roller_sit.py [--seeds 6] [--top 12]
"""

import argparse
import itertools
import math

import mujoco
import numpy as np

XML = "src/mjlab_microduck/robot/microduck/scene_rollers.xml"
KP = 200.0          # firmware PD strength (XML default 0.6 is limp)
TIMESTEP = 0.002
HOLD_S = 4.0
TILT_OK = 20.0      # degrees


def build_model():
    model = mujoco.MjModel.from_xml_path(XML)
    model.opt.timestep = TIMESTEP
    model.actuator_gainprm[:, 0] = KP
    model.actuator_biasprm[:, 1] = -KP
    return model


def set_symmetric_legs(model, data, hip_pitch, knee, ankle, hip_roll=0.0):
    """Apply a symmetric leg pose (left/right mirrored) by joint name."""
    values = {
        "left_hip_pitch": hip_pitch, "left_knee": knee, "left_ankle": ankle,
        "left_hip_roll": hip_roll,
        "right_hip_pitch": -hip_pitch, "right_knee": -knee, "right_ankle": -ankle,
        "right_hip_roll": -hip_roll,
    }
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name in values:
            data.qpos[model.jnt_qposadr[j]] = values[name]


def trial(model, data, hip_pitch, knee, ankle, hip_roll, z0, seed):
    rng = np.random.default_rng(seed)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[2] = z0
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    set_symmetric_legs(model, data, hip_pitch, knee, ankle, hip_roll)
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name and not name.startswith("passive_"):
            data.qpos[model.jnt_qposadr[j]] += rng.normal(0.0, 0.03)
    for a in range(model.nu):
        j = model.actuator_trnid[a, 0]
        data.ctrl[a] = data.qpos[model.jnt_qposadr[j]]
    mujoco.mj_forward(model, data)
    for _ in range(int(HOLD_S / TIMESTEP)):
        mujoco.mj_step(model, data)
    w = min(1.0, abs(float(data.qpos[3])))
    return 2.0 * math.degrees(math.acos(w)), float(data.qpos[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    model = build_model()
    data = mujoco.MjData(model)

    # Symmetric leg-pose grid: hip_pitch (fwd flexion), knee, ankle. Sign
    # convention follows the walker keyframes (hip_pitch negative = flexed).
    grid = itertools.product(
        [-0.2, -0.5, -0.8, -1.1, -1.4],     # hip_pitch
        [0.4, 0.8, 1.2, 1.5],               # knee
        [0.0, 0.3, 0.6, 0.9],               # ankle
        [0.0, 0.2],                         # hip_roll (splay)
    )
    results = []
    for hip_pitch, knee, ankle, hip_roll in grid:
        tilts = []
        zs = []
        for seed in range(args.seeds):
            t, z = trial(model, data, hip_pitch, knee, ankle, hip_roll,
                         z0=0.075, seed=seed)
            tilts.append(t)
            zs.append(z)
        tilts = np.array(tilts)
        results.append((float((tilts < TILT_OK).mean()), float(np.median(tilts)),
                        float(np.median(zs)), hip_pitch, knee, ankle, hip_roll))

    results.sort(key=lambda r: (-r[0], r[1]))
    print(f"{'survive':>8} {'med tilt':>9} {'med z':>8}  hip_pitch  knee  ankle  hip_roll")
    print("-" * 66)
    for ok, mt, mz, hp, kn, an, hr in results[:args.top]:
        print(f"{ok*100:7.0f}% {mt:8.1f}° {mz:8.4f}  {hp:9.2f} {kn:5.2f} {an:6.2f} {hr:8.2f}")


if __name__ == "__main__":
    main()
