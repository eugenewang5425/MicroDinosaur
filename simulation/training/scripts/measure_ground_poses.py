#!/usr/bin/env python3
"""Measure the trunk height of each ground pose by exact kinematics (CPU, no GPU).

For each pose (face-down, face-up, side-left, side-right, sitting, standing) we
place the robot in that orientation, run ``mj_forward`` and find the lowest
vertex of every colliding geom. Setting that lowest vertex to z=0 gives the
trunk-origin height the pose rests at — the number the reset function should
use as its z target.

Usage:
    uv run python scripts/measure_ground_poses.py [--xml PATH] [--joints home|sit]
"""

import argparse
import math

import mujoco
import numpy as np


def _geom_world_min_z(model, data, g) -> float:
    """Lowest world-frame z of geom g's collision surface (mesh-exact where possible)."""
    pos = data.geom_xpos[g]
    mat = data.geom_xmat[g].reshape(3, 3)
    gtype = model.geom_type[g]

    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        vid = model.geom_dataid[g]
        va, vn = model.mesh_vertadr[vid], model.mesh_vertnum[vid]
        verts = model.mesh_vert[va:va + vn]
        return float((pos + verts @ mat.T)[:, 2].min())

    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        half = model.geom_size[g]
        corners = np.array([[sx * half[0], sy * half[1], sz * half[2]]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        return float((pos + corners @ mat.T)[:, 2].min())

    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(pos[2] - model.geom_size[g][0])

    if gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        r, half_len = model.geom_size[g][0], model.geom_size[g][1]
        axis = mat[:, 2]  # local z
        return float(pos[2] - abs(axis[2]) * half_len - r)

    return float(pos[2])


def lowest_collision_z(model, data) -> float:
    lo = math.inf
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue  # visual-only geom
        lo = min(lo, _geom_world_min_z(model, data, g))
    return lo


def set_pose(model, data, quat, joint_pos=None, free_z=0.5):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[2] = free_z
    data.qpos[3:7] = quat
    if joint_pos is not None:
        # qpos columns: free (7) then one per hinge joint, in model joint order.
        n = min(len(joint_pos), model.nq - 7)
        data.qpos[7:7 + n] = joint_pos[:n]
    mujoco.mj_forward(model, data)


def settle_height(model, data, quat, joints, free_z, seconds=3.0, dt=0.002):
    """Drop the pose from free_z, simulate until it rests, return (mean_z, std_z, settled)."""
    set_pose(model, data, quat, joints, free_z=free_z)
    n = int(seconds / dt)
    model.opt.timestep = dt
    zs = []
    for i in range(n):
        mujoco.mj_step(model, data)
        if i > n // 2:  # second half = settled window
            zs.append(float(data.qpos[2]))
    zs = np.array(zs)
    return float(zs.mean()), float(zs.std()), float(zs.std()) < 0.002


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="src/mjlab_microduck/robot/microduck/scene_rollers.xml")
    ap.add_argument("--free-z", type=float, default=0.5, help="initial trunk z (only the delta matters)")
    ap.add_argument("--settle", action="store_true", help="simulate the drop instead of pure kinematics")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    s = 2.0 ** -0.5

    # Joint angles from the scene keyframes (model joint order, free joint first).
    def key_qpos(name):
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
        if kid < 0:
            raise SystemExit(f"keyframe {name!r} not found in {args.xml}")
        return np.array(model.key_qpos[kid][7:])

    home = key_qpos("STAND")
    sit = key_qpos("SIT")

    poses = {
        "face-down (prone)": (np.array([s, 0.0, s, 0.0]), home),          # +90° pitch
        "face-up (supine)": (np.array([s, 0.0, -s, 0.0]), home),          # -90° pitch
        "side-right down": (np.array([s, s, 0.0, 0.0]), home),            # +90° roll
        "side-left down": (np.array([s, -s, 0.0, 0.0]), home),            # -90° roll
        "sitting": (np.array([1.0, 0.0, 0.0, 0.0]), sit),
        "standing": (np.array([1.0, 0.0, 0.0, 0.0]), home),
    }

    print(f"model: {args.xml}  (nq={model.nq}, njnt={model.njnt})")
    if args.settle:
        print(f"{'pose':<20} {'settled trunk z':>16} {'std':>9}  settled?")
        print("-" * 56)
        for name, (quat, joints) in poses.items():
            mz, sz, ok = settle_height(model, data, quat, joints, args.free_z)
            print(f"{name:<20} {mz:>16.4f} {sz:>9.5f}  {'yes' if ok else 'NO (still moving)'}")
    else:
        print(f"{'pose':<20} {'lowest geom z (rel. trunk)':>26} {'→ trunk height':>16}")
        print("-" * 66)
        for name, (quat, joints) in poses.items():
            set_pose(model, data, quat, joints, free_z=args.free_z)
            lo = lowest_collision_z(model, data)
            trunk_z = args.free_z - lo
            print(f"{name:<20} {lo:>26.4f} {trunk_z:>16.4f}")


if __name__ == "__main__":
    main()
