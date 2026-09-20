#!/usr/bin/env python3
"""Verify the SIT keyframe on the ROLLER model (CPU, no GPU).

The walker SIT keyframe (knee ±1.35, hip_pitch ∓0.4079, ankle 0) was verified
on the wheel-less model. On rollers the blades+wheels change the contact
geometry, so the seated pose must be re-verified: does it settle (low tilt),
and at what trunk height?

Usage:
    uv run python scripts/measure_roller_sit.py [--sweep]
"""

import argparse
import math

import mujoco
import numpy as np

SIT_BY_NAME = {
    "left_hip_roll": 0.0, "left_hip_pitch": -0.4079, "left_knee": 1.35, "left_ankle": 0.0,
    "right_hip_roll": 0.0, "right_hip_pitch": 0.4079, "right_knee": -1.35, "right_ankle": 0.0,
}
HOME_BY_NAME = {
    "left_hip_yaw": 0.0, "left_hip_roll": -0.0873, "left_hip_pitch": -0.4579,
    "left_knee": -0.0049, "left_ankle": 0.4530,
    "neck_pitch": 0.3491, "head_pitch": 0.3491, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0873, "right_hip_pitch": 0.4579,
    "right_knee": 0.0049, "right_ankle": -0.4530,
}


def set_joints_by_name(model, data, by_name):
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name in by_name:
            data.qpos[model.jnt_qposadr[j]] = by_name[name]


def tilt_deg(quat) -> float:
    w = min(1.0, abs(float(quat[0])))
    return 2.0 * math.degrees(math.acos(w))


def run(model, data, by_name, z0, noise_std=0.0, seconds=4.0, seed=0, kp=None):
    """Settle the pose under the XML position actuators (firmware-PD stand-in).

    The raw XML gains (kp=0.6) are far too soft to hold a pose; the real robot
    and training run a firmware PD at kp≈200. ``kp`` overwrites the gain so the
    test reflects a held pose, not a limp collapse.
    """
    rng = np.random.default_rng(seed)
    if kp is not None:
        model.actuator_gainprm[:, 0] = kp
        model.actuator_biasprm[:, 1] = -kp
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[2] = z0
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    set_joints_by_name(model, data, by_name)
    if noise_std > 0:
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and not name.startswith("passive_"):
                data.qpos[model.jnt_qposadr[j]] += rng.normal(0.0, noise_std)
    # Hold every servo at its (possibly noisy) current angle: ctrl = qpos target.
    for a in range(model.nu):
        j = model.actuator_trnid[a, 0]
        data.ctrl[a] = data.qpos[model.jnt_qposadr[j]]
    mujoco.mj_forward(model, data)
    n = int(seconds / model.opt.timestep)
    for _ in range(n):
        mujoco.mj_step(model, data)
    return tilt_deg(data.qpos[3:7]), float(data.qpos[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="src/mjlab_microduck/robot/microduck/scene_rollers.xml")
    ap.add_argument("--sweep", action="store_true", help="noisy-reset sweep instead of single run")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    model.opt.timestep = 0.002

    if not args.sweep:
        for label, pose, z0 in (("HOME/stand", HOME_BY_NAME, 0.138),
                                ("SIT keyframe", SIT_BY_NAME, 0.075)):
            tilt, z = run(model, data, pose, z0, kp=200.0)
            print(f"{label:<14} settled trunk z={z:.4f}  tilt={tilt:6.1f}°  "
                  f"{'STABLE' if tilt < 20 else 'TIPPED'}")
    else:
        # Noisy-reset sweep: fraction of starts that stay upright (the check the
        # walker SIT keyframe passed at 95-100%).
        for label, pose, z0 in (("HOME/stand", HOME_BY_NAME, 0.138),
                                ("SIT keyframe", SIT_BY_NAME, 0.075)):
            tilts = []
            zs = []
            for seed in range(20):
                t, z = run(model, data, pose, z0, noise_std=0.05, seed=seed, kp=200.0)
                tilts.append(t)
                zs.append(z)
            tilts = np.array(tilts)
            ok = float((tilts < 20).mean() * 100)
            print(f"{label:<14} tilt<20°: {ok:5.1f}%   median tilt {np.median(tilts):5.1f}°   "
                  f"median z {np.median(zs):.4f}")


if __name__ == "__main__":
    main()
