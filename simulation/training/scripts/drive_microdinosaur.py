"""Interactive keyboard drive for the DuckRex-Tail policy.

Runs the trained 16-DOF tail policy inside a MuJoCo window; the window owns the
keyboard focus, so no terminal is needed while driving.

Keys (hold to keep moving; release to stand):
    W / S        forward / backward     (lin x  +/- 0.30 m/s)
    A / D        strafe left / right    (lin y  +/- 0.20 m/s)
    Q / E        turn left / right      (yaw    +/- 0.80 rad/s)
    Space        stop (zero command)
    R            reset the robot
    Esc          quit

The policy runs in the exact training environment (warp on CUDA, BAM actuators,
domain randomization OFF in play mode).  A separate CPU scene mirrors the robot
state each step for the window renderer.
"""

import argparse
import glob
import sys
import time
import traceback
from dataclasses import asdict

import numpy as np
import torch

import mujoco
from mujoco.glfw import glfw

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import (
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
)

TASK = "Mjlab-Velocity-Flat-MicroDinosaur"
# 16-DOF snapshot of the tail model — matches the trained checkpoint
# (model_12500).  The main model in robot/microdinosaur/ has since grown two
# arm joints (18 DOF) and has no trained policy yet; switch SCENE_XML and the
# robot spec back once an 18-DOF policy exists.
ROBOT16_XML = "src/mjlab_microduck/robot/microdinosaur16/robot_microdinosaur.xml"
SCENE_XML = "src/mjlab_microduck/robot/microdinosaur16/scene_microdinosaur.xml"

CMD_VX = 0.30
CMD_VY = 0.20
CMD_WY = 0.80


def find_latest_checkpoint() -> str:
    ckpts = glob.glob("logs/rsl_rl/velocity_microdinosaur/*/model_*.pt")
    assert ckpts, "no checkpoints found under logs/rsl_rl/velocity_microdinosaur/"
    return sorted(ckpts, key=lambda p: int(p.split("model_")[-1].split(".")[0]))[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint .pt (default: latest in velocity_microdinosaur logs)")
    ap.add_argument("--envs", type=int, default=1)
    args = ap.parse_args()

    # --- training-environment side (policy runs here) -----------------------
    # Build the env directly on the 16-DOF snapshot (the registered task now
    # points at the 18-DOF model, which has no trained checkpoint yet).
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab_microduck.robot.microduck_constants import (
        FULL_COLLISION,
        HOME_FRAME,
        make_bam_actuator,
    )
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    def _get_robot16_spec() -> mujoco.MjSpec:
        return mujoco.MjSpec.from_file(ROBOT16_XML)

    robot16 = EntityCfg(
        spec_fn=_get_robot16_spec,
        init_state=HOME_FRAME,
        collisions=(FULL_COLLISION,),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                make_bam_actuator(r"^(?!passive_|tail_).*"),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^tail_.*",),
                    stiffness=2.0,
                    damping=0.06,
                    effort_limit=0.226,
                    armature=0.002,
                    frictionloss=0.01,
                ),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
    )

    cfg = make_microduck_velocity_env_cfg(
        play=True,
        robot_cfg=robot16,
        reset_z=(0.115, 0.125),
        pose_exclude_extra="|.*tail.*",
    )
    cfg.scene.num_envs = args.envs
    raw = ManagerBasedRlEnv(cfg, device="cuda:0")
    agent_cfg = load_rl_cfg(TASK)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(TASK)(env, asdict(agent_cfg), device="cuda:0")
    ckpt = args.checkpoint or find_latest_checkpoint()
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    print(f"[drive] loaded {ckpt}", flush=True)

    robot = raw.scene["robot"]
    cm = raw.command_manager

    def force_cmd(vx: float, vy: float, wy: float) -> None:
        term = cm.get_term("twist")
        buf = (getattr(term, "_command", None)
               or getattr(term, "_commands", None)
               or getattr(term, "command", None))
        buf[:, 0] = vx
        buf[:, 1] = vy
        buf[:, 2] = wy

    obs, _ = env.reset()

    # --- render side: CPU mirror of the same robot --------------------------
    # The generated XML currently carries a couple of degenerate (zero-eigenvalue)
    # inertias on small parts; MuJoCo refuses to compile those.  Floor the
    # diagonal before compiling the mirror (display-only; the training env
    # compiles through its own pipeline and is unaffected).
    spec = mujoco.MjSpec.from_file(SCENE_XML)

    def _all_bodies(body, out):
        out.append(body)
        for ch in body.bodies:
            _all_bodies(ch, out)

    _bodies = []
    _all_bodies(spec.worldbody.bodies[0], _bodies)
    for _b in _bodies:
        _fi = _b.fullinertia
        if _fi is not None and min(_fi[:3]) <= 0.0:
            _fi = np.array(_fi, dtype=float)
            _fi[:3] = np.maximum(_fi[:3], 1e-6)
            _b.fullinertia = _fi
    mirror = spec.compile()
    mdata = mujoco.MjData(mirror)

    glfw.init()
    window = glfw.create_window(1280, 800, "DuckRex-Tail 驾驶 (窗口焦点按键)", None, None)
    if not window:
        raise RuntimeError("failed to create glfw window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = (0.0, 0.0, 0.115)
    cam.azimuth = 145
    cam.elevation = -14
    cam.distance = 0.75
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    scene = mujoco.MjvScene(mirror, maxgeom=20000)
    ctx = mujoco.MjrContext(mirror, mujoco.mjtFontScale.mjFONTSCALE_150)

    keys = set()

    def key_cb(w, key, scancode, act, mods):
        if act == glfw.PRESS:
            keys.add(key)
        elif act == glfw.RELEASE:
            keys.discard(key)

    glfw.set_key_callback(window, key_cb)

    prev = 0.0
    try:
        _run_loop(window, glfw, cam, opt, scene, ctx, mirror, mdata,
                  robot, keys, obs, env, policy, force_cmd)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        glfw.terminate()
    print("[drive] window closed", flush=True)
    sys.exit(0)


def _run_loop(window, glfw, cam, opt, scene, ctx, mirror, mdata,
              robot, keys, obs, env, policy, force_cmd):
    def cmd_from_keys():
        vx = vy = wy = 0.0
        if glfw.KEY_W in keys:
            vx += CMD_VX
        if glfw.KEY_S in keys:
            vx -= CMD_VX
        if glfw.KEY_A in keys:
            vy += CMD_VY
        if glfw.KEY_D in keys:
            vy -= CMD_VY
        if glfw.KEY_Q in keys:
            wy += CMD_WY
        if glfw.KEY_E in keys:
            wy -= CMD_WY
        return vx, vy, wy

    prev = 0.0
    while not glfw.window_should_close(window):
        if glfw.KEY_ESCAPE in keys:
            break

        now = time.time()
        if now - prev >= 0.02:  # 50 Hz policy rate
            prev = now
            if glfw.KEY_R in keys:
                obs, _ = env.reset()
                keys.discard(glfw.KEY_R)
            else:
                vx, vy, wy = cmd_from_keys()
                force_cmd(vx, vy, wy)
                o = obs["actor"] if isinstance(obs, dict) else obs
                with torch.no_grad():
                    act = policy(o)
                obs, _, _, _ = env.step(act)

            # sync mirror
            pos = robot.data.root_link_pos_w[0].cpu().numpy()
            quat = robot.data.root_link_quat_w[0].cpu().numpy()
            jpos = robot.data.joint_pos[0].cpu().numpy()
            mdata.qpos[:3] = pos
            mdata.qpos[3:7] = quat
            mdata.qpos[7:7 + len(jpos)] = jpos
            mujoco.mj_forward(mirror, mdata)

            vx, vy, wy = cmd_from_keys()
            mujoco.mjv_updateScene(mirror, mdata, opt, None, cam,
                                   mujoco.mjtCatBit.mjCAT_ALL, scene)
            w0, h0 = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, w0, h0)
            mujoco.mjr_render(viewport, scene, ctx)
            label = (f"vx={vx:+.2f}  vy={vy:+.2f}  wy={wy:+.2f}\n"
                     "W/S 前后   A/D 左右   Q/E 转向   Space 停   R 重置   Esc 退出")
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150,
                               mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport,
                               label, "", ctx)
            glfw.swap_buffers(window)

        glfw.poll_events()

    if glfw.window_should_close(window):
        return


if __name__ == "__main__":
    main()
