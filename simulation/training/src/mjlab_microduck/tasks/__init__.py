from mjlab_microduck.train_hook import maybe_submit_to_hf_jobs

# `train <task> ... --hf-jobs` submits to HF Jobs and exits here, before any
# of the cfg imports below: this module is what mjlab's plugin loader pulls
# in, and it is the only train path no install order can take from us (see
# train_hook.py). A no-op without the flag.
maybe_submit_to_hf_jobs()

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        # resolve_symmetry_config injects _env into train_cfg["algorithm"]["symmetry_cfg"]
        # in-place, sharing the same dict object with self.alg.symmetry.  Replace the
        # train_cfg reference with a copy that omits _env so dump_yaml can serialize the
        # config (MjSpec is not picklable), without touching the PPO's internal reference.
        alg = train_cfg.get("algorithm", {})
        sym = alg.get("symmetry_cfg") if isinstance(alg, dict) else None
        if isinstance(sym, dict) and "_env" in sym:
            alg["symmetry_cfg"] = {k: v for k, v in sym.items() if k != "_env"}


from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)
from .microduck_standup_env_cfg import (
    make_microduck_standup_env_cfg,
    MicroduckStandUpRlCfg,
)
from .microduck_velstand_env_cfg import (
    make_microduck_velstand_env_cfg,
    MicroduckVelStandRlCfg,
)
from .microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
    MicroduckGroundPickRlCfg,
)
from .microduck_ball_kick_env_cfg import (
    make_microduck_ball_kick_env_cfg,
    MicroduckBallKickRlCfg,
)
from .microduck_sitstand_env_cfg import (
    make_microduck_sitstand_env_cfg,
    MicroduckSitStandRlCfg,
)
from .microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
    MicroduckRollersRlCfg,
)
from .microduck_velocity_swizzle_env_cfg import (
    make_microduck_velocity_swizzle_env_cfg,
    MicroduckSwizzleRlCfg,
)
from .microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
    MicroduckRollerCrouchRlCfg,
)
from .microduck_roller_slope_env_cfg import (
    make_microduck_roller_slope_env_cfg,
    MicroduckRollerSlopeRlCfg,
)
from .microduck_roller_standup_env_cfg import (
    make_microduck_roller_standup_env_cfg,
    MicroduckRollerStandUpRlCfg,
)
from .microduck_roller_sitstand_env_cfg import (
    make_microduck_roller_sitstand_env_cfg,
    MicroduckRollerSitStandRlCfg,
)
from .microduck_figure8_rollers_env_cfg import (
    make_microduck_figure8_rollers_env_cfg,
    MicroduckFigure8RlCfg,
)
from .microduck_stride_rollers_env_cfg import (
    make_microduck_stride_rollers_env_cfg,
    MicroduckStrideRlCfg,
)
from .microduck_spin_env_cfg import (
    make_microduck_spin_env_cfg,
    MicroduckSpinRlCfg,
)
from .microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
    MicroduckRouladeRlCfg,
)
from .backlash import make_backlash_variant

# Standard velocity task
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# VelStand — walking + fall recovery + body pose control in one policy.
register_mjlab_task(
    task_id="Mjlab-VelStand-Flat-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-VelStand-Rough-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Stand-up task — robot starts inverted (lying on back) and must stand up
register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-StandUp-Rough-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(rough=True),
    play_env_cfg=make_microduck_standup_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# SitStand task — commanded sit ↔ stand in one policy, gently, head commandable
register_mjlab_task(
    task_id="Mjlab-SitStand-Flat-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SitStand-Rough-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Ground-pick task — crouch, touch the ground with the mouth tip, return to stand
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# BallKick task — kick a 70mm/15g ball forward hard with the right foot from a
# standing start (flat terrain only — a ball on rough terrain is another task).
register_mjlab_task(
    task_id="Mjlab-BallKick-Flat-MicroDuck",
    env_cfg=make_microduck_ball_kick_env_cfg(),
    play_env_cfg=make_microduck_ball_kick_env_cfg(play=True),
    rl_cfg=MicroduckBallKickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(rough=True),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller skate velocity task (passive-wheel model; historical task id kept)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_velocity_rollers_env_cfg(),
    play_env_cfg=make_microduck_velocity_rollers_env_cfg(play=True),
    rl_cfg=MicroduckRollersRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller SWIZZLE task — clean classic swizzle (symmetric, feet grounded).
register_mjlab_task(
    task_id="Mjlab-Velocity-Swizzle-MicroDuck",
    env_cfg=make_microduck_velocity_swizzle_env_cfg(),
    play_env_cfg=make_microduck_velocity_swizzle_env_cfg(play=True),
    rl_cfg=MicroduckSwizzleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerCrouch-Flat-MicroDuck",
    env_cfg=make_microduck_roller_crouch_env_cfg(),
    play_env_cfg=make_microduck_roller_crouch_env_cfg(play=True),
    rl_cfg=MicroduckRollerCrouchRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerSlope-Flat-MicroDuck",
    env_cfg=make_microduck_roller_slope_env_cfg(),
    play_env_cfg=make_microduck_roller_slope_env_cfg(play=True),
    rl_cfg=MicroduckRollerSlopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller STANDUP — se relever sur rollers (policy dédiée, départ au sol).
register_mjlab_task(
    task_id="Mjlab-RollerStandUp-Flat-MicroDuck",
    env_cfg=make_microduck_roller_standup_env_cfg(),
    play_env_cfg=make_microduck_roller_standup_env_cfg(play=True),
    rl_cfg=MicroduckRollerStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller SITSTAND — s'asseoir / se relever sur rollers (posture commandée).
# Keyframe assis RE-VÉRIFIÉ sur le modèle rollers (le keyframe du marcheur
# s'effondre : les lames ne peuvent pas servir d'appui comme une semelle plate).
register_mjlab_task(
    task_id="Mjlab-SitStand-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_roller_sitstand_env_cfg(),
    play_env_cfg=make_microduck_roller_sitstand_env_cfg(play=True),
    rl_cfg=MicroduckRollerSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller STRIDE — vraie poussée de patinage (abduction latérale alternée).
register_mjlab_task(
    task_id="Mjlab-Stride-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_stride_rollers_env_cfg(),
    play_env_cfg=make_microduck_stride_rollers_env_cfg(play=True),
    rl_cfg=MicroduckStrideRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller FIGURE-8 — tracer des 8 en patinant (cible mobile dans le repère robot).
register_mjlab_task(
    task_id="Mjlab-Figure8-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_figure8_rollers_env_cfg(),
    play_env_cfg=make_microduck_figure8_rollers_env_cfg(play=True),
    rl_cfg=MicroduckFigure8RlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Spin task — rotation rapide sur place, sur rollers (slot ground-pick).
register_mjlab_task(
    task_id="Mjlab-Spin-Flat-MicroDuck",
    env_cfg=make_microduck_spin_env_cfg(),
    play_env_cfg=make_microduck_spin_env_cfg(play=True),
    rl_cfg=MicroduckSpinRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roulade — forward roll over the flat head top, land back on the feet.
register_mjlab_task(
    task_id="Mjlab-Roulade-Flat-MicroDuck",
    env_cfg=make_microduck_roulade_env_cfg(),
    play_env_cfg=make_microduck_roulade_env_cfg(play=True),
    rl_cfg=MicroduckRouladeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# MicroDinosaur: the same gait task on the MicroDinosaur model with its 2-DOF balance
# tail ACTUATED (SC09 yaw + pitch) — 16 actions instead of 14, so it has its own
# observation/action contract (no longer interchangeable with the microduck
# runtime's [1,61] -> [1,14] ONNX).  The tail is excluded from the pose reward
# so the policy is free to swing it for attitude control.
from mjlab_microduck.robot.microdinosaur_constants import (
    MICRODINOSAUR_GC_ROBOT_CFG,
    MICRODINOSAUR_ROBOT_CFG,
)

# Own experiment dir so the tail runs don't mix with the microduck walk runs.
import dataclasses as _dc

# Symmetry augmentation (mirror loss) for MicroDinosaur: verified physically against
# the model (mirroring the robot state reproduces the mirrored 81-D observation
# with 0.0 error), which addresses left-right gait asymmetry at large stride.
from mjlab_microduck.tasks.symmetry_microdinosaur import SYMMETRY_CFG as MICRODINOSAUR_SYMMETRY

# --- A/B probe switches (temporary debugging aid) ---------------------------
# MICRODINO_PROBE isolates ONE recipe change at a time against the same
# warm-start checkpoint, because the CLI cannot override these fields (tyro
# cannot parse the `dict | None` symmetry field).  Combine with commas, e.g.
#   MICRODINO_PROBE=nosym,nodr,nostill,ang2
# Unset = the production recipe.  Remove this block once the recipe is settled.
import os as _os

MICRODINO_PROBE = _os.environ.get("MICRODINO_PROBE", "")
MICRODINOSAUR_SYMMETRY_ENABLED = "nosym" not in MICRODINO_PROBE


def _microdinosaur_rl(experiment: str):
    return _dc.replace(
        MicroduckRlCfg,
        experiment_name=experiment,
        run_name=experiment,
        algorithm=_dc.replace(
            MicroduckRlCfg.algorithm,
            symmetry_cfg=(MICRODINOSAUR_SYMMETRY if MICRODINOSAUR_SYMMETRY_ENABLED else None),
        ),
    )


MicroDinosaurRlCfg = _microdinosaur_rl("velocity_microdinosaur")

# --- MicroDinosaur variants (post-training on the walk policy) ---
# ---------------------------------------------------------------- servo model
# S288 servo uncertainty, PROVISIONAL until bench identification (2026-09-12
# decision: official-protocol nominal model now, measured values later).
#   kp / kd : scale factors around the nominal stiffness/damping.  Kept narrow
#             on purpose -- the spec explicitly warns that wide randomization
#             does NOT guarantee a stiffness-insensitive policy.
#   effort  : <1.0 models falling output due to battery sag, current limit and
#             temperature (the 0.6 N.m stall figure is not sustainable).
# kd band re-centred 2026-09-13: the old (0.7, 1.3) around kd=0.4 only ever saw
# 0.28-0.52, so the controller never explored the damping that the sweep shows
# fixes the jitter (0.8).  Now (0.62, 1.25) x 0.8 = 0.50-1.00 rad/s units.
MICRODINOSAUR_SERVO_DR = {"kp": (0.7, 1.3), "kd": (0.62, 1.25), "effort": (0.8, 1.0)}

# Zero-command stillness penalty -- DISABLED 2026-09-12 with measured evidence.
# Measured with scripts/microdinosaur_measure.py, same plant (fine model,
# kp=7/kv=0.4), same warm start model_13500:
#   warm start (no term):      yaw drift +0.79 deg / 5 s, forward 0.430 m/s
#   trained WITH the term:     yaw drift -33.7 deg / 5 s, forward 0.295 m/s
# The term did not just fail to fix standing drift, it came WITH a 31% forward
# speed drop and asymmetric foot chatter (L/R peak counts 97/61 vs 124/124).
# The premise was also wrong: the pre-term policy already stands still, so the
# earlier -22/-26 deg measurement must have come from a different policy/build.
# Re-enable only with a formulation that is validated on the drift measurement,
# not on the training reward.
MICRODINOSAUR_STAND_STILL = 0.0

# Gait-amplitude randomization: the policy is trained across stride sizes so the
# left-right coordination holds at large amplitude (observed failure: right foot
# lifts 5.6 mm vs left 3.5 mm, forward speed collapses above scale ~1.2).
MICRODINOSAUR_ACTION_SCALE_DR = (0.90, 1.15)

# Turn-tracking weight (2.0 -> 2.8 when the user reported turning is slow:
# track_yaw error stayed ~1.6 rad/s against a +/-0.9 command).
MICRODINOSAUR_ANG_TRACK = 2.8

if "nostill" in MICRODINO_PROBE:
    MICRODINOSAUR_STAND_STILL = 0.0

# Zero-command FOOT-SLIP penalty.  mjlab's feet_slip is gated the OTHER way (it
# prices slip only while the command is non-zero), so at standstill the planted
# feet grind for free: measured 2026-09-12 on the fine model at kp=7/kv=0.4, the
# trained policies rotate in place at +14 deg/s with a foot lift of only
# 0.7-0.9 mm -- not marching, twisting.  Cost is slip_velocity^2, so a 0.1 m/s
# grind costs 0.01 per step: weight 20 = -0.2/s while grinding.
MICRODINOSAUR_STAND_SLIP = 30.0

if "noslip" in MICRODINO_PROBE:
    MICRODINOSAUR_STAND_SLIP = 0.0

# Head-gaze stability (2026-09-13 user: walk with a steady head / stable view;
# balance micro-adjustment is the TAIL's and the legs' job).  Always active --
# not gated by command -- because the view must be stable while walking too.
MICRODINOSAUR_HEAD_GAZE = 0.2
# World-frame head target (the user's "头部为预设角度, 视角稳定"): the head must
# hold a commanded world orientation, so the neck works AGAINST the trunk and
# the head can no longer be used to pay for balance -- that goes to the tail,
# trunk and legs.
MICRODINOSAUR_HEAD_WORLD = 1.5

# Action low-pass + slew limit (lp_alpha, max_delta rad/step).
# lp_alpha=0.5 at the 50 Hz control loop ~ 4 Hz cutoff, the value used in the
# biped locomotion stack cited in action_filter.py; max_delta 0.08 rad/step =
# 4 rad/s equals roughly double the policy's measured mean action jerk (0.04),
# i.e. it only clips the worst spikes.  Measured motivation: jitter/rotation
# scale with action amplitude and kp and shrink with kd -- the policy's own
# high-frequency output, not a solver or servo-ring problem.
# (lp_alpha, max_delta).  MEASURED SPECTRUM of this robot's walking (2026-09-13,
# /d/tmp/gait_spectrum.py on the v5 policy): the GAIT lives at 2.2-2.5 Hz -- 93%
# of the foot-height energy and 79% of the trunk yaw-rate energy sit below 4 Hz
# -- while the jitter energy is spread thin above 8 Hz (5.4% of the yaw rate).
# The first setting (0.5 = ~4 Hz at the 50 Hz control loop) therefore sat ON the
# gait: only 13% amplitude attenuation but ~30 deg of phase lag at 2.3 Hz, which
# wrecked the left-right timing (forward 0.230 -> 0.086 m/s, step height
# 21.7 -> 13.6 mm) while still not removing the chatter.
# Now 0.9 (~7.2 Hz, ~18 deg lag at the gait) with the SLEW LIMIT as the primary
# tool: the limit has no frequency-dependent lag and is amplitude-selective, so
# it clips the jitter spikes and passes the step cycle.
# (lp_alpha, max_delta).  SIZED FROM MEASUREMENT, not guessed -- the first two
# attempts were wrong in an instructive way:
#   * 0.08 rad/step (= 4 rad/s) looked like a gentle jitter filter and was
#     actually a WALK DISABLE.  Measured per-step command demand while walking
#     (measure script phase C, rad/step): healthy unfiltered policy 0.262 mean,
#     P95 0.552, P99 0.711, max 0.811 -- a 2.3 Hz gait with ~1 rad swings needs
#     ~14 rad/s of command rate, so 96% of steps were clipped.  The policy then
#     learned to fight the limit (raw output pushed to 0.63 rad/step peaks) and
#     the gait collapsed: forward 0.230 -> 0.085 m/s, step height 21.7 -> 13.1 mm.
#   * raising only the EMA cutoff (0.5 -> 0.9) changed nothing, which proved the
#     LIMIT was the binding constraint, not the low-pass.
# Also note the gait's demand (0.26-0.55) and the jitter amplitude overlap, so at
# a 50 Hz control rate an amplitude limit cannot separate them; the limit is now
# only a spike catcher and kd (0.8) is the real jitter/slip lever.
MICRODINOSAUR_ACTION_FILTER = (0.9, 0.6)

# Lateral-wander penalties, third attempt, now pricing quantities the POLICY CAN
# SEE (see the mdp docstrings for the two failed |vy| attempts: base linear
# velocity is critic-only, so |vy| rewards were not actionable).
#   lean_drift:     |EMA(lateral lean)| from projected gravity.  A 2 deg
#                   persistent lean ~ 0.035 rad -> 0.035*W per step.  W=4 gives
#                   ~0.14/s, comparable to one tracking term.
#   contact_timing: |EMA(contact_L) - EMA(contact_R)|, the duty-cycle imbalance
#                   measured at 23% vs 32% on the v7 gait.  Same weight.
MICRODINOSAUR_LATERAL_DRIFT = 4.0
# Zero-command joint fidget, small on purpose (see stand_jitter_penalty doc);
# the 2026-09-12 collapse was a much broader term at 1.5.
MICRODINOSAUR_STAND_JITTER = 0.4

if "nogaze" in MICRODINO_PROBE:
    MICRODINOSAUR_HEAD_GAZE = 0.0
    MICRODINOSAUR_HEAD_WORLD = 0.0
if "nojitter" in MICRODINO_PROBE:
    MICRODINOSAUR_STAND_JITTER = 0.0


def _microdinosaur_post(cfg):
    """Behavior overrides for the 2026-09-13 user requests.

    - Tail freed for balance: tracking weight 1.5 -> 0.5, std 0.5 -> 0.8.  A
      tightly tracked tail cannot serve as the inertial damper the user asked
      for ("身体的平衡尽量用尾巴和腿部来微调").
    - Head: tighter pose hold (std 0.5 -> 0.3) on top of the head_gaze term.
    - Trunk: body_ang_vel -0.05 -> -0.15 (head_camera is trunk-mounted, so a
      stable view starts with a quiet trunk).
    """
    # Tail command tracking back to 1.0 (was dropped to 0.3 on 09-13 to free the
    # tail for balance): the user now wants the tail RAISED to vertical on
    # command and held there while walking ("尾巴可以竖着走"), which needs the
    # tracking term to be strong enough to follow the command.  Balance work
    # stays with the legs/trunk; the tail's own inertia still contributes.
    if "tail_pose_tracking" in cfg.rewards:
        cfg.rewards["tail_pose_tracking"].weight = 1.0
        cfg.rewards["tail_pose_tracking"].params["std"] = 0.8
    cur = cfg.curriculum.get("tail_track_weight")
    if cur is not None and "weight_stages" in cur.params:
        cur.params["weight_stages"] = [
            {**st, "weight": 1.0 if st["weight"] >= 1.5 else st["weight"]}
            for st in cur.params["weight_stages"]
        ]
    # head_pose_tracking becomes a POSTURE regularizer (keep the neck near a
    # sane configuration); the objective is the world-frame target below.
    if "head_pose_tracking" in cfg.rewards:
        cfg.rewards["head_pose_tracking"].params["std"] = 0.3
        cfg.rewards["head_pose_tracking"].weight = 0.8
    # body_ang_vel: TRIED -0.15 on 2026-09-13 and REVERTED.  Measured on the
    # v3 policy (800 iters from the condim6 baseline, same eval):
    #   head world ang vel  161 -> 85 deg/s   (the win, but it came from the
    #                                         head_gaze term + head std 0.3)
    #   trunk world ang vel  86 -> 78 deg/s   (barely moved)
    #   standing drift      +6.3 -> +18.1 deg / 5 s
    #   turning achievement  83/48% -> 52/45%
    #   forward 0.55 cmd     0.337 -> 0.209 m/s  (-38%)
    # Braking the trunk costs the rotations the robot needs to BALANCE (gait
    # trunk sway, standing micro-corrections) without buying much trunk
    # quietness -- the head work is done by the head terms.  Kept at -0.05.
    if "body_ang_vel" in cfg.rewards:
        cfg.rewards["body_ang_vel"].weight = -0.05
    # Linear velocity tracking raised 2.0 -> 2.6 (2026-09-13).  Measured: with
    # the head-world and tail tracking terms at full strength the policy prefers
    # HOLDING the pose over walking -- the tail-up phase ran at 0.04 m/s against
    # a 0.35 command (tail angle tracked 84-96%, no falls), and plain forward
    # fell to 0.230 m/s.  This is the roadmap's step 2 (walk/run) pulled forward
    # because the posture terms now outbid locomotion.
    if "track_linear_velocity" in cfg.rewards:
        cfg.rewards["track_linear_velocity"].weight = 2.6
    return cfg
if "nodr" in MICRODINO_PROBE:
    MICRODINOSAUR_ACTION_SCALE_DR = None
if "ang2" in MICRODINO_PROBE:
    MICRODINOSAUR_ANG_TRACK = 2.0

# Feedback buffers advance once per policy observation (20 ms), so these
# joint-feedback lags mean 20-40 ms, NOT 5-10 ms. Verified in mjlab 1.3.0's
# ManagerBasedRlEnv.step / ObservationManager.compute_group. Still provisional
# pending timestamps; actuator command lags advance at the 5 ms physics step.
MICRODINOSAUR_OBS_DELAY = (1, 2)

# Rough terrain for MicroDinosaur: same steps/bumps as the microduck rough
# config but steeper slopes (0.05-0.15 rise/run = 2.9°-8.5° — the tail gives it
# more pitch authority, and the user asked for slope capability explicitly).
from dataclasses import replace as _replace
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MICRODUCK_ROUGH_TERRAINS_CFG as _MD_ROUGH

DUCKREX_TAIL_ROUGH_TERRAINS = _replace(
    _MD_ROUGH,
    sub_terrains={**_MD_ROUGH.sub_terrains,
                  "pyramid_slope": _replace(
                      _MD_ROUGH.sub_terrains["pyramid_slope"],
                      slope_range=(0.05, 0.15),
                  )},
)


def make_microdinosaur_rough_env_cfg(play: bool = False):
    cfg = make_microduck_velocity_env_cfg(
        play=play, rough=True,
        robot_cfg=MICRODINOSAUR_ROBOT_CFG,
        reset_z=(0.115, 0.125),
        pose_exclude_extra="|.*tail.*|.*arm.*|.*jaw.*",
        rough_terrains=DUCKREX_TAIL_ROUGH_TERRAINS,
        arm_pose_cmd=True,
        tail_pose_cmd=True,
        jaw_pose_cmd=True,
        lin_vel_max=0.55,
        body_pose_weight=1.0,
        body_nominal_height=0.117,
        servo_dr=MICRODINOSAUR_SERVO_DR,
        obs_delay=MICRODINOSAUR_OBS_DELAY,
        stand_still_weight=MICRODINOSAUR_STAND_STILL,
        stand_slip_weight=MICRODINOSAUR_STAND_SLIP,
        head_gaze_weight=MICRODINOSAUR_HEAD_GAZE,
        head_world_weight=MICRODINOSAUR_HEAD_WORLD,
        action_filter=MICRODINOSAUR_ACTION_FILTER,
        lateral_drift_weight=MICRODINOSAUR_LATERAL_DRIFT,
        stand_jitter_weight=MICRODINOSAUR_STAND_JITTER,
        foot_condim6=True,
        action_scale_dr=MICRODINOSAUR_ACTION_SCALE_DR,
        ang_vel_max=0.9,
        ang_track_weight=MICRODINOSAUR_ANG_TRACK,
    )
    if not play:
        # Post-training focus: pull capacity out of turning into linear
        # tracking (the flat-model eval showed turning is over-learned and
        # forward/backward/strafe lag).
        cmd = cfg.commands["twist"]
        cmd.ranges.ang_vel_z = (-0.8, 0.8)
        cmd.rel_turn_in_place_envs = 0.10
        cfg.rewards["track_linear_velocity"].weight = 2.5
    return _microdinosaur_post(cfg)


def make_microdinosaur_velstand_env_cfg(play: bool = False):
    return _microdinosaur_post(make_microduck_velstand_env_cfg(
        play=play,
        robot_cfg=MICRODINOSAUR_GC_ROBOT_CFG,
        reset_z=(0.115, 0.125),
        pose_exclude_extra="|.*tail.*|.*arm.*|.*jaw.*",
        arm_pose_cmd=True,
        tail_pose_cmd=True,
        jaw_pose_cmd=True,
        lin_vel_max=0.55,
        body_pose_weight=1.0,
        body_nominal_height=0.117,
    ))


import dataclasses as _dc

MicroDinosaurRoughRlCfg = _microdinosaur_rl("velocity_microdinosaur_rough")
MicroDinosaurVelStandRlCfg = _microdinosaur_rl("velstand_microdinosaur")

# Phase 1 (simplified domain) twin of the flat task: one ideal PD on all 19
# joints, solver iterations 10 -> 4 — the fast kinematics-learning stage of the
# two-stage method. Phase 2 = the main task post-trained from this run.
from mjlab_microduck.robot.microdinosaur_constants import MICRODINOSAUR_PHASE1_ROBOT_CFG

import dataclasses as _dc2

MicroDinosaurP1RlCfg = _dc2.replace(
    MicroduckRlCfg,
    experiment_name="velocity_microdinosaur_p1",
    run_name="velocity_microdinosaur_p1",
)  # P1 保持无对称(简化域基线)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDinosaur-P1",
    env_cfg=make_microduck_velocity_env_cfg(robot_cfg=MICRODINOSAUR_PHASE1_ROBOT_CFG,
                                            reset_z=(0.115, 0.125),
                                            pose_exclude_extra="|.*tail.*|.*arm.*|.*jaw.*",
                                            arm_pose_cmd=True, tail_pose_cmd=True,
                                            jaw_pose_cmd=True, lin_vel_max=0.55,
                                            body_pose_weight=1.0, body_nominal_height=0.117,
                                            sol_iter=4),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True,
                                                 robot_cfg=MICRODINOSAUR_PHASE1_ROBOT_CFG,
                                                 reset_z=(0.115, 0.125),
                                                 pose_exclude_extra="|.*tail.*|.*arm.*|.*jaw.*",
                                                 arm_pose_cmd=True, tail_pose_cmd=True,
                                                 jaw_pose_cmd=True, lin_vel_max=0.55,
                                                 body_pose_weight=1.0, body_nominal_height=0.117,
                                                 sol_iter=4),
    rl_cfg=MicroDinosaurP1RlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# 诊断任务: 关掉手臂/尾巴/嘴的手势跟踪奖励(权重量 0), 验证"手势跟踪挤压走路
# 学习"的假设 —— 观测量与主任务一致(指令槽保留, 置零), 只改奖励权重。
MicroDinosaurNoLimbRlCfg = _dc2.replace(
    MicroduckRlCfg,
    experiment_name="velocity_microdinosaur_nolimb",
    run_name="velocity_microdinosaur_nolimb",
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDinosaur-Nolimb",
    env_cfg=make_microduck_velocity_env_cfg(robot_cfg=MICRODINOSAUR_PHASE1_ROBOT_CFG,
                                            reset_z=(0.115, 0.125),
                                            pose_exclude_extra="|.*tail.*|.*arm.*|.*jaw.*",
                                            arm_pose_cmd=True, tail_pose_cmd=True,
                                            jaw_pose_cmd=True, lin_vel_max=0.55,
                                            body_pose_weight=1.0, body_nominal_height=0.117,
                                            sol_iter=4, limb_track_scale=0.0),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True,
                                                 robot_cfg=MICRODINOSAUR_PHASE1_ROBOT_CFG,
                                                 reset_z=(0.115, 0.125),
                                                 pose_exclude_extra="|.*tail.*|.*arm.*|.*jaw.*",
                                                 arm_pose_cmd=True, tail_pose_cmd=True,
                                                 jaw_pose_cmd=True, lin_vel_max=0.55,
                                                 body_pose_weight=1.0, body_nominal_height=0.117,
                                                 sol_iter=4, limb_track_scale=0.0),
    rl_cfg=MicroDinosaurNoLimbRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-MicroDinosaur",
    env_cfg=make_microdinosaur_rough_env_cfg(),
    play_env_cfg=make_microdinosaur_rough_env_cfg(play=True),
    rl_cfg=MicroDinosaurRoughRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-VelStand-Flat-MicroDinosaur",
    env_cfg=make_microdinosaur_velstand_env_cfg(),
    play_env_cfg=make_microdinosaur_velstand_env_cfg(play=True),
    rl_cfg=MicroDinosaurVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDinosaur",
    env_cfg=_microdinosaur_post(make_microduck_velocity_env_cfg(robot_cfg=MICRODINOSAUR_ROBOT_CFG,
                                            reset_z=(0.115, 0.125),
                                            pose_exclude_extra="|.*tail.*|.*arm.*|.*jaw.*",
                                            arm_pose_cmd=True,
                                            tail_pose_cmd=True,
                                            jaw_pose_cmd=True,
                                            lin_vel_max=0.55,
                                            body_pose_weight=1.0,
                                            body_nominal_height=0.117,
                                            sol_iter=20,
                                            servo_dr=MICRODINOSAUR_SERVO_DR,
                                            obs_delay=MICRODINOSAUR_OBS_DELAY,
                                            stand_still_weight=MICRODINOSAUR_STAND_STILL,
                                            stand_slip_weight=MICRODINOSAUR_STAND_SLIP,
                                            head_gaze_weight=MICRODINOSAUR_HEAD_GAZE,
                                            head_world_weight=MICRODINOSAUR_HEAD_WORLD,
                                            action_filter=MICRODINOSAUR_ACTION_FILTER,
                                            lateral_drift_weight=MICRODINOSAUR_LATERAL_DRIFT,
                                            stand_jitter_weight=MICRODINOSAUR_STAND_JITTER,
                                            foot_condim6=True,
                                            action_scale_dr=MICRODINOSAUR_ACTION_SCALE_DR,
                                            ang_vel_max=0.9,
                                            ang_track_weight=MICRODINOSAUR_ANG_TRACK)),
    play_env_cfg=_microdinosaur_post(make_microduck_velocity_env_cfg(play=True,
                                                 robot_cfg=MICRODINOSAUR_ROBOT_CFG,
                                                 reset_z=(0.115, 0.125),
                                                 pose_exclude_extra="|.*tail.*|.*arm.*|.*jaw.*",
                                                 arm_pose_cmd=True,
                                                 tail_pose_cmd=True,
                                                 jaw_pose_cmd=True,
                                                 lin_vel_max=0.55,
                                                 body_pose_weight=1.0,
                                                 body_nominal_height=0.117,
                                                 sol_iter=20,
                                                 servo_dr=MICRODINOSAUR_SERVO_DR,
                                                 obs_delay=MICRODINOSAUR_OBS_DELAY,
                                                 stand_still_weight=MICRODINOSAUR_STAND_STILL,
                                                 stand_slip_weight=MICRODINOSAUR_STAND_SLIP,
                                                 head_gaze_weight=MICRODINOSAUR_HEAD_GAZE,
                                                 head_world_weight=MICRODINOSAUR_HEAD_WORLD,
                                                 action_filter=MICRODINOSAUR_ACTION_FILTER,
                                                 lateral_drift_weight=MICRODINOSAUR_LATERAL_DRIFT,
                                                 stand_jitter_weight=MICRODINOSAUR_STAND_JITTER,
                                                 foot_condim6=True,
                                                 action_scale_dr=MICRODINOSAUR_ACTION_SCALE_DR,
                                                 ang_vel_max=0.9,
                                                 ang_track_weight=MICRODINOSAUR_ANG_TRACK)),
    rl_cfg=MicroDinosaurRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Backlash variants — ±1° serial gear play per servo + encoder-through-backlash
# actuator feedback and joint obs (see tasks/backlash.py). Each family keeps its
# base task's collision model: Velocity → robot_walk_backlash.xml,
# VelStand/StandUp → robot_groundcontact_backlash.xml. Obs/action dims are
# unchanged vs the base tasks.
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_BACKLASH_ROBOT_CFG,
    MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG,
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
)

# (task_id, make_fn, make_kwargs, rl_cfg, backlash robot cfg). Task ids mirror
# the base ids with "-Backlash" inserted. Walk-model tasks get the walk
# backlash robot, roller tasks the wheels+backlash robot, the rest the
# groundcontact backlash robot — same model as their base task in each case.
_BL_GROUNDCONTACT = MICRODUCK_BACKLASH_ROBOT_CFG
_BL_WALK = MICRODUCK_WALK_BACKLASH_ROBOT_CFG
_BL_ROLLERS = MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG
_BACKLASH_TASKS = (
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-Velocity-Rough-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {"rough": True}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-VelStand-Flat-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {}, MicroduckVelStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-VelStand-Rough-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {"rough": True}, MicroduckVelStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-StandUp-Flat-Backlash-MicroDuck", make_microduck_standup_env_cfg, {}, MicroduckStandUpRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-StandUp-Rough-Backlash-MicroDuck", make_microduck_standup_env_cfg, {"rough": True}, MicroduckStandUpRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-SitStand-Flat-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {}, MicroduckSitStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-SitStand-Rough-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {"rough": True}, MicroduckSitStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-GroundPick-Flat-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {}, MicroduckGroundPickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-GroundPick-Rough-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {"rough": True}, MicroduckGroundPickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-BallKick-Flat-Backlash-MicroDuck", make_microduck_ball_kick_env_cfg, {}, MicroduckBallKickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers", make_microduck_velocity_rollers_env_cfg, {}, MicroduckRollersRlCfg, _BL_ROLLERS),
    ("Mjlab-Velocity-Swizzle-Backlash-MicroDuck", make_microduck_velocity_swizzle_env_cfg, {}, MicroduckSwizzleRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerCrouch-Flat-Backlash-MicroDuck", make_microduck_roller_crouch_env_cfg, {}, MicroduckRollerCrouchRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerSlope-Flat-Backlash-MicroDuck", make_microduck_roller_slope_env_cfg, {}, MicroduckRollerSlopeRlCfg, _BL_ROLLERS),
)
for _task_id, _make_cfg, _kw, _rl_cfg, _robot_cfg in _BACKLASH_TASKS:
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_backlash_variant(_make_cfg(**_kw), _robot_cfg),
        play_env_cfg=make_backlash_variant(_make_cfg(play=True, **_kw), _robot_cfg),
        rl_cfg=_rl_cfg,
        runner_cls=MicroduckOnPolicyRunner,
    )
