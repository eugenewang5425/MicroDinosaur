"""MicroDinosaur interactive checker -- self-rendered window (NO MuJoCo native viewer).

Highlights
  * Offscreen render (mujoco.Renderer) -> tkinter window, resizable.  Click the
    image once to give it focus; keys ALSO work from the terminal (both input
    channels are polled, so whichever window has focus drives it).
  * Real joint dynamics: all 19 joints as Unitree S288 position servos
    (kp / kv / +/-0.6 Nm, all adjustable live), full-collision model + floor.
  * Live console for gait amplitude / speed / servo stiffness.
  * ASCII-only console output: this machine's console is a GBK code page that
    mangles CJK, so the terminal stays ASCII and the on-screen overlay renders
    text with a TrueType font (so Chinese shows correctly inside the image).

Keys (click the image first; the terminal also works):
  Arrow keys      forward / back / strafe left / strafe right
  A / E           turn left / right in place
  Space           clear all commands
  1 / 2           tail wag  on / off
  3               arm wave  on / off
  4               jaw open  / close
  5               crouch    on / off
  6 / 7           head up   / head down
  R               reset        Q or Esc  quit

Live console (also shown in the overlay):
  + / -           action amplitude (gait size)       0.2 .. 1.6
  ] / [           velocity command amplitude (speed) 0.1 .. 1.0
  K / L           servo stiffness kp                 1.0 .. 30
  J / H           servo damping kv                   0.0 .. 1.0
  . / ,           solver iterations                 5 .. 120
  C               cycle camera view
  P               print parameters to the terminal

Usage (from the repo root):
  uv run scripts/microdinosaur_play.py --onnx microdinosaur_check.onnx --model gc
  uv run scripts/microdinosaur_play.py --onnx p.onnx --no-window --outline out.mp4
"""

import argparse
import math
import os
import queue
import sys
import threading
import time

import numpy as np
import onnxruntime as ort
import mujoco

# ------------------------------------------------------------------ constants
JOINT_ORDER = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "jaw_hinge",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    "tail_yaw", "tail_pitch", "arm_l", "arm_r",
]
HOME = {
    "left_hip_yaw": 0.0, "left_hip_roll": -0.0873, "left_hip_pitch": -0.4579,
    "left_knee": -0.0049, "left_ankle": 0.4530,
    "neck_pitch": 0.3491, "head_pitch": 0.3491, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0873, "right_hip_pitch": 0.4579,
    "right_knee": 0.0049, "right_ankle": -0.4530,
    "tail_yaw": 0.0, "tail_pitch": 0.0, "arm_l": 0.0, "arm_r": 0.0, "jaw_hinge": 0.0,
    "tail_pitch_cmd": 0.0,   # 尾巴抬起指令(1.57 = 竖直向上)
}
# command block inside the 81-dim actor obs (measured against the env):
#   [63:66] twist  [66:70] head  [70:76] body  [76:78] arm  [78:80] tail  [80] jaw
# the local 18-dim cmd vector uses: twist[0:3] head[3:7] body[7:13] arm[13:15]
#                                  tail[15:17] jaw[17]

DEFAULT_XML = {
    # "fine" is the Phase-2 walk model the main task trains on (was pointing at
    # the pre-rename microdinosaur_tail path, which no longer exists).
    # "v07" is the 2026-09-13 CAD re-export (672-entry mass ledger, 1.098 kg,
    # head-IMU mount) that the calibration candidate was trained on.
    "gc": r"src\mjlab_microduck\robot\microdinosaur_gc\robot_microdinosaur_gc.xml",
    "fine": r"src\mjlab_microduck\robot\microdinosaur\robot_microdinosaur.xml",
    "p1": r"src\mjlab_microduck\robot\microdinosaur_p1\robot_microdinosaur_p1.xml",
    "v07": r"src\mjlab_microduck\robot\microdinosaur_v07\robot_microdinosaur_v07.xml",
}

SCENE_ASSETS = """<asset>
  <texture type="skybox" name="sky" builtin="gradient" rgb1="0.32 0.46 0.66"
           rgb2="0.03 0.05 0.08" width="256" height="1536"/>
  <texture type="2d" name="grid" builtin="checker" mark="edge"
           rgb1="0.24 0.32 0.42" rgb2="0.13 0.19 0.27" markrgb="0.72 0.76 0.8"
           width="300" height="300"/>
  <material name="gridmat" texture="grid" texuniform="true" texrepeat="8 8"
            reflectance="0.12"/>
</asset>
"""


def build_model(xml_path, kp=6.0, kv=0.25, floor=True, iters=40, foot_contact="condim6"):
    """S288 position servos on all 19 joints + checkered floor + sky."""
    spec = mujoco.MjSpec.from_file(xml_path)

    def config(a):
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.dyntype = mujoco.mjtDyn.mjDYN_NONE
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = kp
        a.biasprm[1] = -kp
        a.biasprm[2] = -kv
        a.forcerange[:] = [-0.6, 0.6]
        a.ctrllimited = False

    have = set()
    for a in spec.actuators:
        jn = a.target if isinstance(a.target, str) else ""
        if jn in JOINT_ORDER:
            config(a)
            have.add(jn)
    for jn in JOINT_ORDER:
        if jn not in have:
            config(spec.add_actuator(name=jn, target=jn))

    # Foot contact model must match training exactly, otherwise the window shows
    # a different plant than the policy was trained on: mjlab applies the
    # CollisionCfg at env build only, so the plain XML still carries the old
    # condim=3 feet (no torsional friction, free rolling).
    try:
        # condim 6 (torsion + rolling) is 67x slower per iteration in MJX/warp
        # (194 s vs 2.9 s at 4096 envs) while plain MuJoCo CPU cost is the same,
        # so the production choice is condim=4 (torsion only, 2.8 s/iter).
        want_condim = {"condim6": 6, "condim4": 4}.get(foot_contact, 0)
        if not want_condim:
            raise RuntimeError("legacy contact model requested")
        from mjlab_microduck.robot.microdinosaur_constants import (
            MICRODINOSAUR_FOOT_FRICTION,
        )

        for g in spec.geoms:
            if g.name in ("left_foot_collision", "right_foot_collision"):
                g.condim = want_condim
                g.friction = list(MICRODINOSAUR_FOOT_FRICTION)
    except Exception as exc:  # keep the viewer usable if the import fails
        print(f"[play] 脚底接触参数未能应用({exc});用的是 XML 里的旧值")

    xml_text = spec.to_xml()
    if floor:
        scene = SCENE_ASSETS
        xml_text = xml_text.replace("<worldbody>", scene + "<worldbody>", 1)
        extra = ('<geom name="check_floor" type="plane" size="25 25 0.05" '
                 'pos="0 0 0" material="gridmat"/>'
                 '<light pos="0 0 3" dir="0 0 -1" directional="true" '
                 'diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>')
        xml_text = xml_text.replace("<worldbody>", "<worldbody>" + extra, 1)
    path = os.path.join(os.path.dirname(os.path.abspath(xml_path)), "_check_scene.xml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml_text)
    model = mujoco.MjModel.from_xml_path(path)
    model.opt.timestep = 0.005
    model.opt.iterations = iters
    model.opt.ls_iterations = max(2 * iters, 20)
    try:
        model.vis.global_.azimuth = 135
        model.vis.global_.elevation = -18
    except Exception:
        pass
    return model


class Sim:
    def __init__(self, xml, onnx, kp=6.0, kv=0.25, iters=40, foot_contact="condim6",
                 lp_alpha=0.9, max_delta=0.6, cmd_lag=1, obs_lag=1):
        self.sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
        self.model = build_model(xml, kp=kp, kv=kv, iters=iters, foot_contact=foot_contact)
        self.lp_alpha = float(lp_alpha)
        self.max_delta = max_delta
        # Training runs WITH bus/actuator delays (command 5-15 ms, joint
        # feedback 20-40 ms).  Evaluating the policy without them distorts its
        # behaviour badly (2026-09-14 audit: the calibration candidate spun in
        # place at zero delay and walked 0.28 m/s with 10/20 ms delays), so the
        # window replicates them: commands and the proprioceptive obs block
        # (imu ang-vel, joint pos/vel, last action) are delayed by whole policy
        # steps (20 ms each at the 50 Hz loop).  cmd_lag/obs_lag = 0 disables.
        import collections as _collections
        self._cmd_q = _collections.deque(maxlen=cmd_lag + 1) if cmd_lag > 0 else None
        self._g_q = _collections.deque(maxlen=obs_lag + 1) if obs_lag > 0 else None
        self._jq_q = _collections.deque(maxlen=obs_lag + 1) if obs_lag > 0 else None
        self._applied = np.zeros(19, dtype=np.float32)
        self._last_applied = np.zeros(19, dtype=np.float32)
        self.data = mujoco.MjData(self.model)
        self.jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                     for j in JOINT_ORDER]
        self.jadr = [self.model.jnt_qposadr[j] for j in self.jids]
        self.dofadr = [self.model.jnt_dofadr[j] for j in self.jids]
        self.imu = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.imu_adr = self.model.sensor_adr[self.imu] if self.imu >= 0 else 0
        self.trunk = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.home = np.array([HOME[j] for j in JOINT_ORDER], dtype=np.float32)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0, 0, 0.117182]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        for jn, v in HOME.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = v
                self.data.ctrl[self.model.actuator(jn).id] = v
        self.last_action = np.zeros(19, dtype=np.float32)
        # Match the training action term's episode reset. These arrays hold
        # HOME-relative targets; retaining them leaks the previous trial into
        # the next standing/turning/walking measurement.
        self._applied.fill(0.0)
        self._last_applied.fill(0.0)
        # The delay shift registers must clear with the episode too.
        for q in (self._cmd_q, self._g_q, self._jq_q):
            if q is not None:
                q.clear()
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _delay_push(buf, value):
        buf.append(value)
        return buf[0]

    def obs(self, cmd):
        d, m = self.data, self.model
        qpos = np.array([d.qpos[a] for a in self.jadr], dtype=np.float32)
        qvel = np.array([d.qvel[a] for a in self.dofadr], dtype=np.float32)
        g = d.sensordata[self.imu_adr:self.imu_adr + 3].astype(np.float32)
        if self._g_q is not None:
            g = self._delay_push(self._g_q, g)
        joint = np.concatenate([qpos - self.home, qvel, self.last_action])
        if self._jq_q is not None:
            joint = self._delay_push(self._jq_q, joint)
        R = d.xmat[self.trunk].reshape(3, 3)
        grav = (R.T @ np.array([0, 0, -1], dtype=np.float32)).astype(np.float32)
        return np.concatenate([g, grav, joint, cmd]).astype(np.float32)[None, :]

    def step(self, cmd, action_scale=1.0):
        if self._cmd_q is not None:
            cmd = self._delay_push(self._cmd_q, np.array(cmd, dtype=np.float32))
        action = np.asarray(self.sess.run(None, {"obs": self.obs(cmd)})[0][0],
                            dtype=np.float32)
        # last_action stays the UNFILTERED policy output, matching the training
        # observation (mjlab's last_action is the raw output too).
        self.last_action = action.copy()
        # Training's amplitude randomization scales the processed target,
        # not the raw action fed back to the next observation.
        action = action * action_scale
        # Same action low-pass + slew limit as training:
        # applied = 0.1 * previous + 0.9 * target, once per 20 ms;
        # then limit each target change to 0.6 rad (not a hardware speed cap).
        # Deployment
        # MUST include this, otherwise the policy is applied unfiltered and the
        # authority it learned to rely on no longer matches.
        if self.lp_alpha > 0.0:
            self._applied = self._applied + self.lp_alpha * (action - self._applied)
        else:
            self._applied = action
        if self.max_delta is not None:
            d = np.clip(self._applied - self._last_applied, -self.max_delta, self.max_delta)
            self._applied = self._last_applied + d
        self._last_applied = self._applied.copy()
        for i, jn in enumerate(JOINT_ORDER):
            self.data.ctrl[self.model.actuator(jn).id] = float(self._applied[i]) + HOME[jn]
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)

    def set_gains(self, kp=None, kv=None):
        for i in range(self.model.nu):
            if kp is not None:
                self.model.actuator_gainprm[i, 0] = kp
                self.model.actuator_biasprm[i, 1] = -kp
            if kv is not None:
                self.model.actuator_biasprm[i, 2] = -kv

    def render(self, cam, size):
        p = self.data.xpos[self.trunk]
        cam.lookat[:] = [p[0], p[1], max(p[2], 0.09)]
        if getattr(self, "_size", None) != size:
            self.model.vis.global_.offwidth = max(size[0], 640)
            self.model.vis.global_.offheight = max(size[1], 480)
            if hasattr(self, "_ren"):
                self._ren.close()
            self._ren = mujoco.Renderer(self.model, height=size[1], width=size[0])
            self._size = size
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        self._ren.update_scene(self.data, camera=cam, scene_option=opt)
        return self._ren.render()


class KeyReader:
    """Terminal key reader (Windows msvcrt) so keys work without window focus."""

    _WIN = {"H": "up", "P": "down", "K": "left", "M": "right"}

    def __init__(self):
        self.q = queue.Queue()
        self.ok = False
        try:
            import msvcrt
            self.msvcrt = msvcrt
            self.ok = True
        except ImportError:
            self.msvcrt = None

    def start(self):
        if self.ok:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            if not self.msvcrt.kbhit():
                time.sleep(0.01)
                continue
            ch = self.msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                self.q.put(self._WIN.get(self.msvcrt.getwch(), ""))
            elif ch == " ":
                self.q.put("space")
            else:
                self.q.put(ch.lower())

    def drain(self):
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=None,
                    help="policy ONNX (or use --ckpt / --latest to pick a checkpoint)")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint .pt path; exported to a temp ONNX automatically")
    ap.add_argument("--latest", action="store_true",
                    help="use the newest checkpoint of --experiment")
    ap.add_argument("--experiment", default="velocity_microdinosaur",
                    help="experiment dir under logs/rsl_rl for --latest")
    ap.add_argument("--record", default=None,
                    help="record the whole session to this mp4 (ASCII path)")
    ap.add_argument("--record-fps", type=int, default=30)
    ap.add_argument("--model", default="gc", choices=list(DEFAULT_XML))
    ap.add_argument("--xml", default=None)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--outline", default=None)
    ap.add_argument("--size", default="960x720")
    ap.add_argument("--kp", type=float, default=7.0,
                    help="servo stiffness (N.m/rad).  P2 nominal is 7.0 (S288 "
                         "protocol-nominal, bench ID pending).  For the Phase-1 "
                         "policy use --kp 1.5 (its transfer limit)")
    ap.add_argument("--kv", type=float, default=0.4)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--lp-alpha", type=float, default=0.9,
                    help="动作低通 EMA 系数(与训练一致; 0=关闭)")
    ap.add_argument("--max-delta", type=float, default=0.6,
                    help="单步动作变化上限 rad(与训练一致; 负数=关闭)")
    ap.add_argument("--no-filter", action="store_true",
                    help="关闭动作滤波, 用于对比'有/无滤波'的手感差异")
    ap.add_argument("--cmd-lag", type=int, default=1,
                    help="指令延迟(策略步数, 1步=20ms); 训练带 5-15ms 指令延迟, 0=关闭")
    ap.add_argument("--obs-lag", type=int, default=1,
                    help="本体感觉反馈延迟(策略步数, 1步=20ms); 训练带 20-40ms 反馈延迟, 0=关闭")
    args = ap.parse_args()
    xml = args.xml or DEFAULT_XML[args.model]

    # ---- resolve the policy: --onnx, --ckpt or --latest ------------------
    import glob as _glob
    import subprocess as _sp
    import tempfile

    ckpt = args.ckpt
    if args.latest and not ckpt:
        cands = []
        for run in _glob.glob(os.path.join("logs", "rsl_rl", args.experiment, "*")):
            for f in _glob.glob(os.path.join(run, "model_*.pt")):
                cands.append((int(os.path.basename(f).split("_")[1].split(".")[0]), f))
        if cands:
            ckpt = max(cands)[1]
            print(f"[play] newest checkpoint: {ckpt}")
    if ckpt and not args.onnx:
        tmp = os.path.join(tempfile.gettempdir(), "microdinosaur_play_tmp.onnx")
        task = "Mjlab-Velocity-Flat-MicroDinosaur"
        print(f"[play] exporting {os.path.basename(ckpt)} -> {tmp}")
        _sp.run(["uv", "run", "scripts/export.py", task,
                 "--checkpoint-file", ckpt, "--onnx-file", tmp], check=False)
        args.onnx = tmp
    if not args.onnx:
        raise SystemExit("need --onnx, or --ckpt <model_*.pt>, or --latest")
    print(f"[play] policy: {args.onnx}")
    W, H = (int(x) for x in args.size.lower().split("x"))

    sim = Sim(xml, args.onnx, kp=args.kp, kv=args.kv, iters=args.iters,
              lp_alpha=(0.0 if args.no_filter else args.lp_alpha),
              max_delta=(None if args.no_filter or args.max_delta < 0 else args.max_delta),
              cmd_lag=max(0, args.cmd_lag), obs_lag=max(0, args.obs_lag))
    st = dict(vx=0.0, vy=0.0, wz=0.0, wag=False, wave=False, jaw=False, crouch=False,
              head=0.0, action_scale=1.0, cmd_scale=1.0, kp=args.kp, kv=args.kv,
              iters=args.iters, view=0, tail_pitch_cmd=0.0,
              run=False, fold=False)
    cmd = np.zeros(18, dtype=np.float32)

    def build_cmd(t):
        cmd[:] = 0.0
        s = st["cmd_scale"]
        cmd[0] = st["vx"] * s
        cmd[1] = st["vy"] * s
        cmd[2] = st["wz"] * s
        cmd[4] = st["head"]
        if st["crouch"]:
            cmd[9] = -0.03
        # 折叠蹲: body-z command pushed BEYOND the trained +-4 cm band.  The
        # roadmap lists 蹲 as NOT yet trained (roadmap item 3), so -6.5 cm is an
        # out-of-distribution probe -- exactly what this window is for.  With
        # --model fine/v07 only the FEET collide with the floor, so a deep fold
        # can sink the trunk through the ground: use --model gc (15 ground
        # collision bodies) to see real body-ground contact.
        if st["fold"]:
            cmd[9] = -0.065
        # 跑步: fixed 0.8 m/s forward command = 1.45x the trained lin_vel_max
        # (0.55 m/s).  Also out-of-distribution; the window shows how the gait
        # degrades past the trained envelope (roadmap item 2: retrain planned).
        if st["run"]:
            cmd[0] = max(cmd[0], 0.8)
        if st["wave"]:
            cmd[13] = cmd[14] = 0.9
        if st["wag"]:
            cmd[15] = 0.5 * math.sin(2 * math.pi * 0.8 * t)
        # Tail pitch has its own slider now, so "how vertical" is directly
        # tunable: 0 = HOME (level, pointing back), 1.57 = straight up, joint
        # limit 1.75 rad (100 deg).  Commanded every step (not only while
        # wagging) so the tail can be held up while walking.
        cmd[16] = st["tail_pitch_cmd"]
        if st["jaw"]:
            cmd[17] = 0.35
        return cmd

    def osm():
        return (f"cmd vx={st['vx']*st['cmd_scale']:+.2f} vy={st['vy']*st['cmd_scale']:+.2f} "
                f"wz={st['wz']*st['cmd_scale']:+.2f} | amp x{st['action_scale']:.2f} "
                f"spd x{st['cmd_scale']:.2f} kp {st['kp']:.1f} kv {st['kv']:.2f} "
                f"it {st['iters']} | filt {sim.lp_alpha:.2f}/{sim.max_delta} | tail {'ON' if st['wag'] else 'off'} "
                f"arm {'ON' if st['wave'] else 'off'} jaw {'ON' if st['jaw'] else 'off'} "
                f"crouch {'ON' if st['crouch'] else 'off'} "
                f"run {'ON' if st['run'] else 'off'} fold {'ON' if st['fold'] else 'off'} "
                f"lag {1 if sim._cmd_q else 0}/{1 if sim._g_q else 0}")

    def apply(name):
        if name in ("q", "escape"):
            return False
        if name == "r":
            sim.reset()
        elif name == "up":
            st["vx"] = 0.3
        elif name == "down":
            st["vx"] = -0.3
        elif name == "left":
            st["vy"] = 0.2
        elif name == "right":
            st["vy"] = -0.2
        elif name == "a":
            st["wz"] = 0.8
        elif name == "e":
            st["wz"] = -0.8
        elif name == "space":
            st.update(vx=0.0, vy=0.0, wz=0.0, wag=False, wave=False)
        elif name == "1":
            st["wag"] = True
        elif name == "2":
            st["wag"] = False
        elif name == "3":
            st["wave"] = not st["wave"]
        elif name == "4":
            st["jaw"] = not st["jaw"]
        elif name == "5":
            st["crouch"] = not st["crouch"]
        elif name == "6":
            st["head"] = min(st["head"] + 0.15, 0.6)
        elif name == "7":
            st["head"] = max(st["head"] - 0.15, -0.6)
        elif name in ("plus", "equal"):
            st["action_scale"] = round(min(st["action_scale"] + 0.1, 1.6), 2)
        elif name in ("minus", "underscore"):
            st["action_scale"] = round(max(st["action_scale"] - 0.1, 0.2), 2)
        elif name == "bracketright":
            st["cmd_scale"] = round(min(st["cmd_scale"] + 0.1, 1.0), 2)
        elif name == "bracketleft":
            st["cmd_scale"] = round(max(st["cmd_scale"] - 0.1, 0.1), 2)
        elif name == "k":
            st["kp"] = min(round(st["kp"] + 2.0, 1), 30.0); sim.set_gains(kp=st["kp"])
        elif name == "l":
            st["kp"] = max(round(st["kp"] - 2.0, 1), 1.0); sim.set_gains(kp=st["kp"])
        elif name == "j":
            st["kv"] = min(round(st["kv"] + 0.05, 2), 1.0); sim.set_gains(kv=st["kv"])
        elif name == "h":
            st["kv"] = max(round(st["kv"] - 0.05, 2), 0.0); sim.set_gains(kv=st["kv"])
        elif name == "period":
            st["iters"] = min(st["iters"] + 10, 120); sim.model.opt.iterations = st["iters"]
        elif name == "comma":
            st["iters"] = max(st["iters"] - 10, 5); sim.model.opt.iterations = st["iters"]
        elif name == "c":
            st["view"] = (st["view"] + 1) % 4
        elif name == "p":
            print("params:", st)
        print("  " + osm())
        return True

    rec_frames = []

    if args.no_window:
        print("[self-test] stand 1s -> forward 3s -> turn 2s")
        frames = []
        sim.reset()
        for phase, (vx, wz, n) in (("stand", (0.0, 0.0, 50)),
                                   ("forward", (0.3, 0.0, 150)),
                                   ("turn", (0.0, 0.8, 100))):
            for k in range(n):
                st["vx"], st["wz"] = vx, wz
                sim.step(build_cmd(k * 0.02), st["action_scale"])
                if args.outline:
                    cam = mujoco.MjvCamera()
                    cam.azimuth, cam.elevation, cam.distance = 135, -18, 0.9
                    frames.append(sim.render(cam, (W, H)))
            print(f"  {phase:8s} vx={sim.data.qvel[0]:+.3f} wz={sim.data.qvel[5]:+.3f} "
                  f"z={sim.data.qpos[2]:+.4f}")
        if args.outline and frames:
            import imageio
            imageio.mimsave(args.outline, frames, fps=50)
            print("wrote", args.outline)
        return

    # ---- window mode: optionally record every rendered frame ------------
    if args.record:
        print(f"[play] recording session to {args.record} "
              f"({args.record_fps} fps) -- press Q to finish and write the file")

    import tkinter as tk
    from PIL import Image, ImageDraw, ImageFont, ImageTk

    FONT_PATH = None
    for cand in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
                 r"C:\Windows\Fonts\segoeui.ttf"):
        if os.path.exists(cand):
            FONT_PATH = cand
            break
    osd_font = ImageFont.truetype(FONT_PATH, 22) if FONT_PATH else None
    gui_font = ("Microsoft YaHei", 13)
    big_font = ("Microsoft YaHei", 15, "bold")

    root = tk.Tk()
    root.title("MicroDinosaur checker - mouse buttons work directly; click image for keys")
    img = ImageTk.PhotoImage(Image.new("RGB", (W, H), (24, 24, 28)))
    label = tk.Label(root, image=img, bd=0, highlightthickness=0)
    label.pack(fill="both", expand=True)
    holder = {"img": img}

    panel = tk.Frame(root)
    panel.pack(fill="x")

    def row(n):
        f = tk.Frame(panel)
        f.grid(row=n, column=0, sticky="w")
        return f

    def btn(parent, text, fn, w=7):
        b = tk.Button(parent, text=text, width=w, font=gui_font, command=fn)
        b.pack(side="left", padx=2, pady=2)
        return b

    r1 = row(0)
    btn(r1, "前进 ↑", lambda: st.update(vx=0.3, vy=0.0))
    btn(r1, "后退 ↓", lambda: st.update(vx=-0.3, vy=0.0))
    btn(r1, "左移 ←", lambda: st.update(vy=0.2, vx=0.0))
    btn(r1, "右移 →", lambda: st.update(vy=-0.2, vx=0.0))
    btn(r1, "左转 ↺", lambda: st.update(wz=0.8))
    btn(r1, "右转 ↻", lambda: st.update(wz=-0.8))
    btn(r1, "停 ■", lambda: st.update(vx=0.0, vy=0.0, wz=0.0))
    btn(r1, "重置", lambda: sim.reset(), w=6)

    r2 = row(1)
    tail_btn = btn(r2, "甩尾巴", lambda: st.update(wag=not st["wag"]))
    arm_btn = btn(r2, "挥手", lambda: st.update(wave=not st["wave"]))
    jaw_btn = btn(r2, "张嘴", lambda: st.update(jaw=not st["jaw"]))
    crouch_btn = btn(r2, "蹲探针", lambda: st.update(crouch=not st["crouch"], fold=False))
    fold_btn = btn(r2, "深蹲探针", lambda: st.update(fold=not st["fold"], crouch=False))
    run_btn = btn(r2, "冲刺探针", lambda: st.update(run=not st["run"]))
    btn(r2, "头抬", lambda: st.update(head=min(st["head"] + 0.15, 0.6)))
    btn(r2, "头低", lambda: st.update(head=max(st["head"] - 0.15, -0.6)))
    btn(r2, "切视角", lambda: st.update(view=(st["view"] + 1) % 4))

    # ---- 专项技能检查(第三行) ----------------------------------------
    # 蹲/折叠蹲/跑步/起身 都是独立专项模型(另一条训练线的产物), 不在当前
    # 走路策略里; 且它们的忠实检查需要各自训练时的 1.25ms 物理 + 电机包络 +
    # 头部 IMU 外环栈 —— 这个窗口不复制该栈, 所以这里直接打开由该栈渲染的
    # 检查视频(与验收条件一致), 而不是在本窗口的走路物理里假跑。
    SKILL_VIDEOS = [
        ("蹲·专项25mm", r"D:\项目\miro_dinosaur\research\20260914_contact_motion\squat_contact_optimized.mp4"),
        ("折叠蹲·脚本40mm", r"D:\项目\miro_dinosaur\research\20260914_requested_fold\deeper_fold_probe.mp4"),
        ("折叠40/70对比", r"D:\项目\miro_dinosaur\research\20260914_fold_recovery\videos\fold_comparison.mp4"),
        ("跑步·专家0.6", r"D:\项目\miro_dinosaur\research\20260914_contact_motion\run_final\v0.6_c10_s941.mp4"),
        ("起身·专项", r"D:\项目\miro_dinosaur\research\20260914_contact_motion\getup_dense_final\back_c10_s941.mp4"),
    ]

    def open_video(path):
        if os.path.exists(path):
            os.startfile(path)
        else:
            print("缺视频文件:", path)

    r4 = row(3)
    for text, path in SKILL_VIDEOS:
        btn(r4, text, lambda p=path: open_video(p), w=11)

    r3 = tk.Frame(panel)
    r3.grid(row=2, column=0, sticky="w")
    sliders = {}

    def slider(parent, key, lo, hi, res, text, fmt="{:.2f}"):
        f = tk.Frame(parent)
        f.pack(side="left", padx=6)
        tk.Label(f, text=text, font=gui_font).pack(anchor="w")
        val = tk.Label(f, text=fmt.format(st[key]), font=big_font, fg="#0a6")
        val.pack(anchor="w")

        def on_change(v, k=key, l=val, fm=fmt):
            st[k] = float(v)
            l.configure(text=fm.format(float(v)))
            if k in ("kp", "kv"):
                sim.set_gains(kp=st["kp"], kv=st["kv"])
            elif k == "iters":
                sim.model.opt.iterations = int(st["iters"])

        sc = tk.Scale(f, from_=lo, to=hi, resolution=res, orient="horizontal",
                      length=200, font=gui_font, showvalue=False, command=on_change)
        sc.set(st[key])
        sc.pack()
        sliders[key] = sc

    slider(r3, "action_scale", 0.2, 1.6, 0.05, "动作幅度(步态大小)")
    slider(r3, "cmd_scale", 0.1, 1.0, 0.05, "速度指令(走多快)")
    slider(r3, "kp", 0.5, 20.0, 0.5, "伺服刚度 kp (P2 名义 7)", "{:.1f}")
    slider(r3, "kv", 0.0, 1.0, 0.05, "伺服阻尼 kv")
    slider(r3, "iters", 5, 120, 5, "求解器迭代", "{:.0f}")
    slider(r3, "tail_pitch_cmd", 0.0, 1.75, 0.05, "尾巴抬起(1.57=竖直)")

    status = tk.Label(panel, text="", font=("Consolas", 12), anchor="w", justify="left")
    status.grid(row=4, column=0, sticky="w", padx=4)
    tk.Label(panel, anchor="w", font=gui_font, fg="#666",
             text="键盘(先点一下画面): 方向键移动 A/E转向 空格停 1/2甩尾 3挥手 4嘴 "
                  "5蹲探针 8冲刺探针 9深蹲探针 6/7头 +/-幅度 [/]速度 K/L刚度 J/H阻尼 ,/.迭代 C视角 R重置 Q退出 "
                  "| 第三行按钮=专项技能检查视频(蹲/折叠蹲/跑步/起身是独立专项模型, 不是本窗口策略)"
             ).grid(row=5, column=0, sticky="w", padx=4)

    def sync_buttons():
        tail_btn.configure(relief="sunken" if st["wag"] else "raised")
        arm_btn.configure(relief="sunken" if st["wave"] else "raised")
        jaw_btn.configure(relief="sunken" if st["jaw"] else "raised")
        crouch_btn.configure(relief="sunken" if st["crouch"] else "raised")
        fold_btn.configure(relief="sunken" if st["fold"] else "raised")
        run_btn.configure(relief="sunken" if st["run"] else "raised")

    def key(name):
        if name in ("q", "escape"):
            return False
        if name == "r":
            sim.reset()
        elif name == "up":
            st.update(vx=0.3)
        elif name == "down":
            st.update(vx=-0.3)
        elif name == "left":
            st.update(vy=0.2)
        elif name == "right":
            st.update(vy=-0.2)
        elif name == "a":
            st.update(wz=0.8)
        elif name == "e":
            st.update(wz=-0.8)
        elif name == "space":
            st.update(vx=0.0, vy=0.0, wz=0.0, wag=False, wave=False)
        elif name == "1":
            st.update(wag=True)
        elif name == "2":
            st.update(wag=False)
        elif name == "3":
            st.update(wave=not st["wave"])
        elif name == "4":
            st.update(jaw=not st["jaw"])
        elif name == "5":
            st.update(crouch=not st["crouch"], fold=False)
        elif name == "8":
            st.update(run=not st["run"])
        elif name == "9":
            st.update(fold=not st["fold"], crouch=False)
        elif name == "6":
            st.update(head=min(st["head"] + 0.15, 0.6))
        elif name == "7":
            st.update(head=max(st["head"] - 0.15, -0.6))
        elif name in ("plus", "equal"):
            st.update(action_scale=round(min(st["action_scale"] + 0.1, 1.6), 2))
        elif name in ("minus", "underscore"):
            st.update(action_scale=round(max(st["action_scale"] - 0.1, 0.2), 2))
        elif name == "bracketright":
            st.update(cmd_scale=round(min(st["cmd_scale"] + 0.1, 1.0), 2))
        elif name == "bracketleft":
            st.update(cmd_scale=round(max(st["cmd_scale"] - 0.1, 0.1), 2))
        elif name == "k":
            st.update(kp=min(round(st["kp"] + 2.0, 1), 30.0)); sim.set_gains(kp=st["kp"])
        elif name == "l":
            st.update(kp=max(round(st["kp"] - 2.0, 1), 1.0)); sim.set_gains(kp=st["kp"])
        elif name == "j":
            st.update(kv=min(round(st["kv"] + 0.05, 2), 1.0)); sim.set_gains(kv=st["kv"])
        elif name == "h":
            st.update(kv=max(round(st["kv"] - 0.05, 2), 0.0)); sim.set_gains(kv=st["kv"])
        elif name == "period":
            st.update(iters=min(st["iters"] + 10, 120)); sim.model.opt.iterations = int(st["iters"])
        elif name == "comma":
            st.update(iters=max(st["iters"] - 10, 5)); sim.model.opt.iterations = int(st["iters"])
        elif name == "c":
            st.update(view=(st["view"] + 1) % 4)
        elif name == "p":
            print("params:", st)
        for k, sc in sliders.items():
            if abs(sc.get() - st[k]) > 1e-6:
                sc.set(st[k])
        print("  " + osm())
        return True

    KEYMAP = {"Up": "up", "Down": "down", "Left": "left", "Right": "right",
              "space": "space", "Escape": "escape",
              "plus": "plus", "equal": "equal", "minus": "minus", "underscore": "underscore",
              "bracketright": "bracketright", "bracketleft": "bracketleft",
              "period": "period", "comma": "comma"}

    def on_key(ev):
        name = KEYMAP.get(ev.keysym, "")
        if not name:
            name = ev.char.lower() if ev.char else ""
        if name and not key(name):
            root.quit()

    root.bind_all("<Key>", on_key)
    root.bind("<Button-1>", lambda e: root.focus_force())
    root.lift()
    root.focus_force()
    kbd = KeyReader()
    kbd.start()

    views = [(135, -18, 0.9), (90, -8, 0.9), (180, -25, 1.3), (135, -55, 1.5)]
    t0 = time.time()

    def loop():
        for name in kbd.drain():
            if name and not key(name):
                root.quit()
        t = time.time() - t0
        try:
            sim.step(build_cmd(t), st["action_scale"])
        except Exception as exc:
            print("sim error (reset):", exc)
            sim.reset()
        az, el, dist = views[st["view"]]
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        frame = sim.render(cam, (max(label.winfo_width(), 360), max(label.winfo_height(), 260)))
        im = Image.fromarray(frame)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 30], fill=(0, 0, 0))
        d.text((8, 4), osm(), fill=(255, 220, 120), font=osd_font)
        status.configure(text=osm())
        sync_buttons()
        if args.record:
            rec_frames.append(frame.copy())
        holder["img"] = ImageTk.PhotoImage(im)
        label.configure(image=holder["img"])
        root.after(15, loop)

    print("=== MicroDinosaur checker ===")
    print("  MOUSE: buttons + sliders in the window (no keyboard focus needed)")
    print("  KEYBOARD: click the image once, then keys work (terminal also works)")
    print(f"  servos kp={st['kp']} kv={st['kv']} iters={st['iters']} (--kp --kv --iters)")
    print("  tip: P2 nominal kp=7 (S288).  The Phase-1 policy needs --kp 1.5")
    try:
        loop()
        root.mainloop()
    finally:
        if args.record and rec_frames:
            import imageio
            imageio.mimsave(args.record, rec_frames, fps=args.record_fps)
            print(f"wrote {args.record}  ({len(rec_frames)} frames)")


if __name__ == "__main__":
    main()
