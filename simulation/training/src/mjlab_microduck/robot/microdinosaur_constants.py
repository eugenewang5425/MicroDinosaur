"""MicroDinosaur-Tail robot model constants.

MicroDinosaur (v1, 2026-09, S288 servos) with its 2-DOF balance tail **actuated**: the CAD's
DCTL_Tail_Yaw / DCTL_Tail_Pitch frames become hinge joints driven by the two
SC09 serial servos, so the policy can use the tail for attitude control.

Differences vs the plain MicroDinosaur walk model:
  * 16 actuated joints: the 14 XL330 joints (BAM) + tail_yaw/tail_pitch (SC09).
  * The SC09 joints are driven by mjlab's BuiltinPositionActuatorCfg with the
    datasheet torque limit; the XL330 BAM group excludes them.
"""

import os
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab_microduck.robot.microduck_constants import (
    FULL_COLLISION,
    HOME_FRAME,
    make_bam_actuator,
)

_ROBOT_DIR: Path = Path(os.path.dirname(__file__)) / "microdinosaur"

MICRODINOSAUR_XML: Path = _ROBOT_DIR / "robot_microdinosaur.xml"

assert MICRODINOSAUR_XML.exists(), f"XML not found: {MICRODINOSAUR_XML}"

# SC09 (Waveshare serial bus servo, metal gears) — datasheet values:
#   stall torque 2.3 kg.cm @6V = 0.226 N.m, no-load speed 0.1 s/60 deg.
# The position loop is a plain PD: kp sized so the tail's own weight is held
# with a sub-degree error, kv set just above critical damping for the tail's
# inertia about each axis.
TAIL_SC09_ACTUATOR = BuiltinPositionActuatorCfg(
    # ALL 19 joints are Unitree S288 on the 2026-09 hardware — the jaw as well
    # (it used to fall into the BAM/XL330 group by accident, since only tail_*
    # and arm_* were excluded).
    target_names_expr=(r"^(tail_|arm_|jaw_).*",),
    stiffness=2.0,
    damping=0.06,
    effort_limit=0.226,
    armature=0.002,
    frictionloss=0.01,
)

# BAM drives the 14 XL330 joints only (tail + arm joints excluded by the regex).
TAIL_BAM_ACTUATOR = make_bam_actuator(r"^(?!passive_|tail_|arm_|jaw_).*")


def get_microdinosaur_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODINOSAUR_XML))

# Phase 1 (simplified domain, per the two-stage method in microdinosaur/README):
# ALL 19 joints on one ideal PD with the S288 stall-torque ceiling — the
# kinematics-learning stage before the fine-model calibration stage.  Uses its
# own XML without baked actuators so the cfg below owns every actuator.
MICRODINOSAUR_P1_XML: Path = Path(os.path.dirname(__file__)) / "microdinosaur_p1" / "robot_microdinosaur_p1.xml"
assert MICRODINOSAUR_P1_XML.exists(), f"XML not found: {MICRODINOSAUR_P1_XML}"


def get_microdinosaur_p1_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODINOSAUR_P1_XML))


# Phase 2 (fine / realistic domain): EVERY joint is a Unitree S288 on the
# 2026-09 hardware, so the whole robot is driven by one S288 position-servo
# model.  The BAM/XL330 model below is legacy hardware and must NOT be used for
# this robot (wrong motor: brushless FOC at 12.6 V vs cored at 5 V).
S288_PD_ALL = BuiltinPositionActuatorCfg(
    target_names_expr=(r"^(?!passive_).*",),
    # ---------------------------------------------------------------- nominal
    # Source of these numbers is the S288 protocol conversion + a physical
    # requirement, NOT a measured robot (2026-09-12 decision, option 3:
    # "official-protocol nominal model + later bench identification").
    #   * The S288 control law is impedance control:
    #         tau = tau_ff + kp*(p_des - p) + kd*(omega_des - omega)
    #     and the protocol encodes kp/kd rotor-side with gear 288.35, so the
    #     official tables give a CONVERSION, not the Kp/Kd we must run.
    #   * Nominal chosen so a static leg load (~0.25 N.m) holds with ~2 deg
    #     error: kp = 0.25 / 0.035 rad ~= 7 N.m/rad.
    #   * Torque ceiling = 0.6 N.m, the official STALL torque.  It is NOT the
    #     protocol-encodable ~36.9 N.m, and it is NOT sustainable: real output
    #     falls with speed, battery sag, current limit and temperature.
    #   >>> BENCH IDENTIFICATION REQUIRED: fix kp, kd and the sustainable
    #   torque curve once measured; then update this block and narrow the
    #   servo_dr ranges in the task configs.
    stiffness=7.0,        # N.m/rad      (nominal -- to be identified)
    # kd 0.4 -> 0.8 on 2026-09-13.  MEASURED with a FIXED policy (only the servo
    # changes, fine model, standing 5 s): kd=0.4 gives 13.7 deg yaw drift,
    # 47 mm/s creep, 0.073 action jerk; kd=0.8 gives 7.9 deg, 15 mm/s, 0.042;
    # kd=1.6 over-damps (19.3 deg, 12 deg spread) because the joints lag the
    # commands.  The joint loops are already OVERDAMPED at 0.4 (zeta 1.2-1.8,
    # 6.6-9.9 Hz), so this is not fixing a servo ring -- it is damping the
    # policy's own high-frequency output.  The S288's kd is commandable, so this
    # is directly requestable on hardware; confirm the real value on the bench.
    damping=0.8,          # N.m/(rad/s)  (nominal -- to be identified)
    effort_limit=0.6,     # N.m  stall ceiling (not sustainable; see above)
    armature=0.002,
    frictionloss=0.01,
    # ---------------------------------------------------------------- delay
    # Command path: policy output at an ASSUMED 50 Hz (not verified on the Pi)
    # dominates; the 500 Hz bus update (2 ms) is below the 5 ms physics step.
    # PLACEHOLDER range pending timestamp measurement -- do NOT read the
    # protocol lower bound (0.767 ms per full A-bus poll) as the actuation
    # delay of the whole robot.
    delay_min_lag=1,      # TBD (physics steps, 5 ms each)
    delay_max_lag=3,      # TBD
    delay_hold_prob=0.5,
)

PHASE1_PD = BuiltinPositionActuatorCfg(
    target_names_expr=(r"^(?!passive_).*",),
    stiffness=0.6,
    damping=0.05,
    effort_limit=0.6,
    armature=0.002,
    frictionloss=0.01,
)

# Foot contact model.  The microduck-inherited FULL_COLLISION gave the feet
# condim=3 -- normal + 2 tangential only -- so the contact CANNOT resist torsion
# at all (MuJoCo needs condim>=4 for torsional friction), and rolling friction
# stayed at the default 1e-4 (free).  Contact audit 2026-09-12: the sole's
# bottom is only a 36 x 8 mm strip, so with no torsional and no rolling
# resistance the foot can twist about the normal and rock on that strip almost
# for free -- which is where the measured ~20 Hz standing chatter (124 foot
# peaks / 5 s) and the "turn by grinding the feet" strategy come from.  condim=6
# adds torsion + rolling; mjlab's own Go1 config uses condim-6 feet with
# per-axis friction randomization ("Replace the base foot_friction with per-axis
# friction events for condim 6"), and that is the pattern followed here.
#   friction = (slide, torsion [m], roll [m]); torsion 0.01 m corresponds to the
#   uniform-pressure estimate (2/3)*mu*r with mu=1.0 and a ~15 mm patch radius.
# MEASURED 2026-09-12 (same policy, no retraining, standing drift / 5 s):
#   condim=3: +71.2 deg | condim=4 (torsion only): +64.8 deg | condim=6: -3.2 deg
# -- the fix comes from condim=6 itself, not from the rolling-friction value.
# COST: condim=6 at mu_roll=1e-3 makes MJX take 194 s/iter (vs 2.9 s), while
# mu_roll=1e-4 already costs 1.5x and >=1e-3 costs 67x.  A flat pad cannot roll
# at all, so the physically right value is ~0 and the roll entry is kept at a
# numerical floor (1e-6) and NOT randomized -- randomizing it only buys solver
# time.  Slide and torsion are the physically meaningful randomized axes.
MICRODINOSAUR_FOOT_FRICTION = (1.0, 0.01, 1e-6)

FULL_COLLISION_CONDIM6 = CollisionCfg(
    geom_names_expr=[".*_collision"],
    condim={r"^(left|right)_foot_collision$": 6, ".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": MICRODINOSAUR_FOOT_FRICTION},
)

MICRODINOSAUR_PHASE1_ROBOT_CFG = EntityCfg(
    spec_fn=get_microdinosaur_p1_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION_CONDIM6,),
    articulation=EntityArticulationInfoCfg(
        actuators=(PHASE1_PD,),
        soft_joint_pos_limit_factor=0.9,
    ),
)



MICRODINOSAUR_ROBOT_CFG = EntityCfg(
    spec_fn=get_microdinosaur_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION_CONDIM6,),
    articulation=EntityArticulationInfoCfg(
        actuators=(S288_PD_ALL,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Ground-contact variant for fall-recovery (VelStand): trunk spine, torso
# panels, battery, head shells, hips, legs, tail cradle and rigid tail all
# carry collision geoms so the robot can physically lie down and push off.
MICRODINOSAUR_GC_XML: Path = Path(os.path.dirname(__file__)) / "microdinosaur_gc" / "robot_microdinosaur_gc.xml"
assert MICRODINOSAUR_GC_XML.exists(), f"XML not found: {MICRODINOSAUR_GC_XML}"


def get_microdinosaur_gc_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODINOSAUR_GC_XML))


MICRODINOSAUR_GC_ROBOT_CFG = EntityCfg(
    spec_fn=get_microdinosaur_gc_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION_CONDIM6,),
    articulation=EntityArticulationInfoCfg(
        actuators=(S288_PD_ALL,),
        soft_joint_pos_limit_factor=0.9,
    ),
)
