"""Convert a pollen-robotics-style Blender CAD assembly into a MuJoCo MJCF model.

The .blend must carry the onshape-to-robot helper-empties convention used by the
microduck CAD (FRAME_<part> / AXIS_<joint> / CTRL_<joint> / BODY_<part>), with
each part mesh parented under its kinematic body.

IMPORTANT concurrency (validated against microduck robot_walk.xml):
  * The CAD is stored in the HOME stance: BODY_* empties (part frames) equal the
    joint's AXIS_* empty pre-rotated by the joint's home angle about its own Z.
  * The AXIS_* empty carries the joint axis direction AT HOME — i.e. including
    the upstream home rotations.  MuJoCo body frames must be at ZERO joint
    angle, so every frame and mesh pose is unwound level by level:
        Fz[b]   = Fz[parent] @ H[parent]^-1 @ A[b]
        mesh_0  = Fz[b] @ H[b]^-1 @ mesh_home
  * Joint origins are the AXIS_* origins; hinge axes are body-local +Z.

Run inside Blender (headless):

  blender --background --python blend2mjcf.py -- \
      --blend <file.blend> --mode inspect|validate|gen \
      --out <repo robot dir> [--name duckrex]

Modes:
  inspect   dump joint/body world frames and per-body part counts
  validate  unwind the frames and compare each joint's zero-angle world axis
            against the axis set measured from microduck robot_walk.xml
  gen       write robot_<name>.xml + assets/*.stl (body-local merged meshes)

Density calibration: shared-robot material classes (printed parts, XL330 servos,
bearings, battery) are fitted with non-negative least squares against the
microduck reference masses; DuckRex-only materials use DENSITY_FALLBACK.
"""

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET

import bpy  # noqa: E402  (Blender-provided)
import numpy as np  # noqa: E402  (Blender-provided)
from mathutils import Matrix, Quaternion, Vector  # noqa: E402  (Blender-provided)

# ------------------------------------------------------------------ topology
# (body_name, joint_name, parent_body_name) — same serial-chain order as the
# microduck XML so observation/action joint order stays 61/14-compatible.
BASE_BODY_TABLE = [
    ("yaw2roll", "left_hip_yaw", "trunk_base"),
    ("hip_l", "left_hip_roll", "yaw2roll"),
    ("upper_leg_left", "left_hip_pitch", "hip_l"),
    ("leg", "left_knee", "upper_leg_left"),
    ("ankle_left", "left_ankle", "leg"),
    ("neck", "neck_pitch", "trunk_base"),
    ("neck_pitch", "head_pitch", "neck"),
    ("yaw_roll_motion", "head_yaw", "neck_pitch"),
    ("jaw_soft", "head_roll", "yaw_roll_motion"),
    ("bearing_roll", "right_hip_yaw", "trunk_base"),
    ("hip_l_2", "right_hip_roll", "bearing_roll"),
    ("upper_leg_right", "right_hip_pitch", "hip_l_2"),
    ("leg_2", "right_knee", "upper_leg_right"),
    ("ankle_right", "right_ankle", "leg_2"),
]

# Tail: 2 DOF balance tail (SC09 yaw + pitch).  The CAD parents the tail parts
# under DCTL_Tail_* (dummy control frames): everything under DCTL_Tail_Yaw
# rotates with the yaw, everything under DCTL_Tail_Pitch (the rigid tail +
# pitch horn/axle) rotates with the pitch, and the yaw servo's case stays on the
# trunk.  Only models built with --with-tail articulate it; otherwise the tail
# parts stay merged into the trunk as a static mass.
TAIL_BODY_TABLE = [
    ("tail_yaw", "tail_yaw", "trunk_base"),
    ("tail_pitch", "tail_pitch", "tail_yaw"),
]

# Arm "wings": two small SC09-driven forelimbs on the trunk front
# (Rex_Arm_L/R). Same construction as the tail bodies — the servo case stays
# on the trunk, the horn/output gear and the arm block ride the arm body, and
# the swing axis is the horizontal lateral axis (world +Y).
ARM_BODY_TABLE = [
    ("arm_l", "arm_l", "trunk_base"),
    ("arm_r", "arm_r", "trunk_base"),
]

# Jaw: the XL330-driven lower mandible (Rex_Jaw_Shell + rotor disc + hinge
# hardware under DCTL_Jaw_Hinge). The servo's fixed case stays inside the head
# (jaw_soft body); the mandible swings about the lateral axis (world +Y).
JAW_BODY_TABLE = [
    ("jaw_hinge", "jaw_hinge", "jaw_soft"),
]

# Set from --with-tail / --with-arms / --with-jaw in main().
WITH_TAIL = False
WITH_ARMS = False
WITH_JAW = False
NO_XML_ACTUATORS = False
COLLISION_PARTS = None

# Ground-contact collision set for fall-recovery / body-on-ground tasks.
# Kept INSIDE the generator (used with `--collision-parts auto`) so a long CLI
# string can't silently lose parts when regenerating -- that happened once and
# the tail ended up with no collision (2026-09-12, user: "尾巴好像没有碰撞体积").
DEFAULT_COLLISION_PARTS = (
    "REX_SKULL_SHELL",        # head shell (top of head, lands in a fall)
    "REX_JAW_SHELL",          # mandible
    "MOUTH",                  # mouth hinge bearings
    "HIP_L",                  # both hip modules (hip_l, hip_l_2)
    "LEG",                    # shank + leg guards
    "REX_CHASSIS_SPINE",      # trunk spine
    "REX_TORSO_PANEL",        # torso side panels
    "REX_ARM",                # forelimbs
    "REX_YAW_PITCH_CRADLE",   # tail mount cradle
    "REX_RIGID_BALANCE_TAIL", # the balance tail itself
    "JMP_3S4000",             # battery pack (back of the robot)
    "V10_BATTERY_6V_ENVELOPE",
)

# Same joint ranges as microduck robot_walk.xml (rad, hinge, frame local Z).
JOINT_RANGES = {
    "left_hip_yaw": (-0.4363323129985824, 0.5235987755982988),
    "left_hip_roll": (-0.3839724354386992, 0.38397243543880577),
    "left_hip_pitch": (-1.570796326794949, 1.5707963267948442),
    "left_knee": (-1.5707963267948983, 1.5707963267948948),
    "left_ankle": (-1.5707963267949063, 1.5707963267948868),
    "neck_pitch": (-1.5707963267948974, 1.0471975511965967),
    "head_pitch": (-1.5707963267948966, 1.5707963267948966),
    "head_yaw": (-2.9670597283903613, 2.9670597283903595),
    "head_roll": (-0.43633231299858327, 0.4363323129985815),
    "right_hip_yaw": (-0.523598775598297, 0.43633231299858416),
    "right_hip_roll": (-0.3839724354387507, 0.38397243543875426),
    "right_hip_pitch": (-1.57079632679494, 1.570796326794853),
    "right_knee": (-1.5707963267949339, 1.5707963267948593),
    "right_ankle": (-1.5707963267949028, 1.5707963267948903),
    # Tail joint limits are a first estimate from the mechanism envelope; the
    # yaw axis is vertical at the SC09 yaw servo, the pitch axis is lateral at
    # the SC09 pitch servo / trunnion.  Adjust to the real linkage if needed.
    "tail_yaw": (-0.7, 0.7),
    # tail_pitch widened 2026-09-13 (user: "让尾巴完全竖起来"): the tail is
    # nearly level at HOME and "vertical" needs about +90 deg, so the old
    # +-0.9 rad cap made a raised tail impossible.  +1.75 rad = +100 deg leaves
    # margin.  STILL A SIMULATION LIMIT: the real SC09/S288 trunnion linkage
    # range must be confirmed on the bench before commanding this on hardware.
    "tail_pitch": (-0.9, 1.75),
    # Forelimbs: generous first estimate (small arms, mostly counterweight
    # sweeps); the CAD doesn't encode mechanical stops.
    "arm_l": (-1.2, 1.2),
    "arm_r": (-1.2, 1.2),
    "jaw_hinge": (-0.6, 0.6),
}

# microduck robot_walk.xml composite inertials (kg, m, m^2) — NNLS calibration
# targets for the shared-part material densities.
MICRODUCK_REF_INERTIAL = {
    "trunk_base": (0.199224, (-0.0226332, -2.70828e-05, 0.00280064),
                   (0.000123975, 0.000145931, 0.000115351, 7.1651e-08, -2.06915e-05, 1.32429e-07)),
    "yaw2roll": (0.0230406, (-8.8e-11, 0.00099414, 0.0168917),
                 (4.1964e-06, 3.41752e-06, 2.3265e-06, -0, -0, -5.0881e-08)),
    "hip_l": (0.00618934, (0.014815, 1.6179e-08, -0.00829992),
              (8.60626e-07, 1.15753e-06, 6.51588e-07, 1e-12, 3.89882e-07, -1e-12)),
    "upper_leg_left": (0.0482067, (0.006766, 0.0228838, 0.0134531),
                       (1.68753e-05, 1.03691e-05, 1.98269e-05, -4.41852e-06, 2.27982e-07, 3.89298e-07)),
    "leg": (0.0215844, (-1.52e-10, 0.0318937, -0.00953505),
            (5.25056e-06, 2.15507e-06, 4.33925e-06, -1e-12, 0, 7.65005e-07)),
    "ankle_left": (0.0300246, (-0.00615742, -0.0137483, -0.0156585),
                   (5.91525e-06, 1.11534e-05, 7.74016e-06, -2.05354e-07, 8.218e-08, 5.6722e-08)),
    "neck": (0.0368414, (-0, -0.025, 0.0141665),
             (1.72925e-05, 3.12694e-06, 1.63707e-05, -0, 0, -0)),
    "neck_pitch": (0.00572, (-6.533e-09, 0.0116641, -0.0145),
                   (1.09186e-06, 9.66855e-07, 5.12111e-07, 0, 0, 0)),
    "yaw_roll_motion": (0.0486, (-0.000166657, -0.00500964, 0.0178065),
                        (6.97903e-06, 8.10293e-06, 9.57884e-06, 3.21657e-07, -3.4797e-08, 3.4295e-07)),
    "jaw_soft": (0.188766, (0.00460523, 0.00120047, -0.0324761),
                 (0.000320811, 0.0002541, 0.000149882, -6.32293e-07, 6.57153e-06, 4.18477e-06)),
    # mirrored right-side bodies (robot_walk.xml values, in the right frames)
    "bearing_roll": (0.0230406, (-8.8e-11, 0.00099414, 0.0168917),
                     (4.1964e-06, 3.41752e-06, 2.3265e-06, -0, -0, -5.0881e-08)),
    "hip_l_2": (0.00618934, (0.014815, 1.6212e-08, -0.00829992),
                (8.60626e-07, 1.15753e-06, 6.51588e-07, 1e-12, 3.89882e-07, -1e-12)),
    "upper_leg_right": (0.0482067, (-0.006766, 0.0228838, 0.0134531),
                        (1.68753e-05, 1.03691e-05, 1.98269e-05, 4.41852e-06, -2.27982e-07, 3.89298e-07)),
    "leg_2": (0.0215844, (-0.0318937, -1.52e-10, -0.00953505),
              (2.15507e-06, 5.25056e-06, 4.33925e-06, 1e-12, -7.65005e-07, 0)),
    "ankle_right": (0.0300251, (0.00615539, -0.0137482, -0.0156566),
                    (5.91534e-06, 1.11534e-05, 7.74015e-06, 2.04877e-07, -8.2467e-08, 5.7017e-08)),
}

# ------------------------------------------------------------------ materials
MATERIAL_CLASS = [
    (("fastener steel", "steel"), "steel"),
    (("unitree s288 graphite",), "servo"),
    (("xl330", "sc09"), "servo"),
    (("seeed_bearing", "f683zz", "bearing"), "bearing"),
    (("np_f970", "6v pack", "battery"), "battery"),
    (("lipo battery", "jmp design material lipo"), "battery"),
    (("pa12",), "printed"),
    (("tpu95a", "nylon hook"), "misc"),
    (("selected pcb", "pcb green", "schematic"), "pcb"),
    (("selected core component",), "misc"),
    (("service mounts",), "printed"),
    (("elec_rpi", "pcb", "banana"), "pcb"),
    (("lens", "optical"), "lens"),
    (("speaker",), "misc"),
    (("graphite", "warm edge", "shell", "face", "jaw", "neck", "hip", "leg",
      "ankle", "foot", "sole", "yaw", "roll", "mouth", "trunk", "power_support",
      "motor_support", "locker", "spacer", "shim", "pad", "clamp", "horn",
      "washer", "insert", "bearing"), "printed"),
]

# Fallback densities (g/cm^3) for classes not derivable from the microduck fit.
DENSITY_FALLBACK = {
    "printed": 1.25,   # filament-ish
    "servo": 2.60,     # XL330 housing + winding
    "bearing": 7.60,
    "battery": 4.50,   # LiPo pack
    "pcb": 1.90,
    "lens": 2.20,
    "misc": 1.30,
    "steel": 7.90,
}

# Collections that are NOT part of the physical build.  The CAD keeps design
# helpers, studio props, clearance envelopes, switched-off v10 fallback parts
# and microduck reference geometry in separate collections — counting those as
# structure inflated the trunk mass by ~99 g (12%).
EXCLUDE_COLLECTIONS = (
    "09_STUDIO",                                   # studio floor + lights + camera
    "30_COMPONENT_ENVELOPES_NOT_DETAILED_CAD",     # envelope placeholders (non-solid)
    "90_DESIGN_RESERVATIONS_NOT_PARTS",            # design reservations, not parts
)
CURRENT_LEDGER_NAMES = set()


def in_excluded_collection(obj) -> bool:
    """True when a part lives in a non-build collection (see EXCLUDE_COLLECTIONS)."""
    if obj is None:
        return False
    if obj.name in CURRENT_LEDGER_NAMES:
        return False
    return any(c.name in EXCLUDE_COLLECTIONS for c in obj.users_collection)


# Per-part world-position offsets (m) applied before any mesh transform.
# v2: the battery moves from the front cradle to the (extended) back, per the
# 2026-09-12 hardware revision.
MESH_POS_OVERRIDE = {}


def eff_matrix(obj):
    """World matrix with the MESH_POS_OVERRIDE shift applied."""
    mw = obj.matrix_world.copy()
    if obj.name in MESH_POS_OVERRIDE:
        dx, dy, dz = MESH_POS_OVERRIDE[obj.name]
        mw.translation.x += dx
        mw.translation.y += dy
        mw.translation.z += dz
    return mw


# Part-name markers that are pure CAD design helpers or parts not installed on
# the robot (kept out of the mass model entirely).
SKIP_NAMES = ("CLEAR_WIRE_ROUTE_", "COM_GRAVITY", "KEEPOUT", "STUDIO_FLOOR",
              "REFERENCE_COM_FROM_MJCF_MASSES",
              # microduck's rear NP-F970 battery model: DuckRex carries a single
              # 6V pack in the FRONT cradle (V10_BATTERY_6V_ENVELOPE).
              "NP_F970",
              # Rear battery bay, deprecated in v12 (the robot now carries the
              # single 6V pack in the FRONT cradle): the cassette, its service
              # hatch, the back service cover that shielded it, and the
              # cassette's own fasteners.
              "V10_BATTERY_CASSETTE", "V10_BATTERY_SERVICE_HATCH",
              "V10_BACK_SERVICE_COVER", "V10_CASSETTE_",
              # v10 torso shell wrap (superseded) and the screws that held it.
              "V10_SHELL_WRAP", "V10_SHELL_M3X14")
SKIP_MATERIALS = ("clearance cyan", "wire route", "com_gravity", "allocation")

# body_name -> its joint-axis empty.  The CAD names axes after the JOINT
# (AXIS_left_hip_roll) while bodies are named after the part (hip_l).
AXIS_EMPTY = {
    "yaw2roll": "AXIS_left_hip_yaw",
    "hip_l": "AXIS_left_hip_roll",
    "upper_leg_left": "AXIS_left_hip_pitch",
    "leg": "AXIS_left_knee",
    "ankle_left": "AXIS_left_ankle",
    "neck": "AXIS_neck_pitch",
    "neck_pitch": "AXIS_head_pitch",
    "yaw_roll_motion": "AXIS_head_yaw",
    "jaw_soft": "AXIS_head_roll",
    "bearing_roll": "AXIS_right_hip_yaw",
    "hip_l_2": "AXIS_right_hip_roll",
    "upper_leg_right": "AXIS_right_hip_pitch",
    "leg_2": "AXIS_right_knee",
    "ankle_right": "AXIS_right_ankle",
    # Tail: the CAD has no AXIS_* empties for the tail; DCTL_Tail_Yaw /
    # DCTL_Tail_Pitch are the servo control frames (both identity-oriented).
    "tail_yaw": "DCTL_Tail_Yaw",
    "tail_pitch": "DCTL_Tail_Pitch",
    # Arms: DCTL_Arm_* are the SC09 servo frames (identity-oriented).
    "arm_l": "DCTL_Arm_L",
    "arm_r": "DCTL_Arm_R",
    "jaw_hinge": "DCTL_Jaw_Hinge",
}

# Extra rotation applied to a body frame so its local +Z becomes the physical
# joint axis.  The tail pitch axis is lateral (world Y) while the CAD's
# DCTL_Tail_Pitch frame is identity-oriented, so rotate -90 deg about X
# (local Z -> world +Y).
FRAME_ROT_OVERRIDE = {
    "tail_pitch": Quaternion((0.7071067811865476, -0.7071067811865476, 0.0, 0.0)),
    # Arms swing about the horizontal lateral axis (world +Y): rotate the
    # identity servo frame -90 deg about X so local +Z points along world +Y.
    "arm_l": Quaternion((0.7071067811865476, -0.7071067811865476, 0.0, 0.0)),
    "arm_r": Quaternion((0.7071067811865476, -0.7071067811865476, 0.0, 0.0)),
    # Jaw: DCTL_Jaw_Hinge's local Z points along world -X (the fore-aft axis),
    # so a rotation ABOUT Z would leave the hinge axis unchanged — the jaw would
    # roll left/right instead of opening (2026-09-12 user report).  The mandible
    # hinge is lateral (the CAD trunnion/bearing axis is world +/-Y), so rotate
    # -90deg about the LOCAL X axis, which maps local Z -> local Y = world +Y.
    "jaw_hinge": Quaternion((0.7071067811865476, -0.7071067811865476, 0.0, 0.0)),
}

# Joint world axes as measured inside MuJoCo for microduck robot_walk.xml.
EXPECT_AXIS = {
    "left_hip_yaw": (0, 0, -1), "left_hip_roll": (1, 0, 0),
    "left_hip_pitch": (0, 1, 0), "left_knee": (0, -1, 0),
    "left_ankle": (0, 1, 0), "neck_pitch": (0, -1, 0),
    "head_pitch": (0, 1, 0), "head_yaw": (0, 0, 1),
    "head_roll": (-1, 0, 0), "right_hip_yaw": (0, 0, -1),
    "right_hip_roll": (1, 0, 0), "right_hip_pitch": (0, -1, 0),
    "right_knee": (0, 1, 0), "right_ankle": (0, -1, 0),
    # Tail (from the CAD mechanism): yaw about the vertical axis at the SC09
    # yaw servo, pitch about the lateral axis at the SC09 pitch servo.
    "tail_yaw": (0, 0, 1), "tail_pitch": (0, 1, 0),
}


# ------------------------------------------------------------------ helpers
class BlendData:
    def __init__(self, empties, meshes):
        self.empties = empties
        self.meshes = meshes

    def pose(self, name):
        o = self.empties[name]
        loc, q = o.matrix_world.translation, o.matrix_world.to_quaternion()
        return np.array(loc), Quaternion((q.w, q.x, q.y, q.z))


def body_table():
    """Active body table (tail bodies with --with-tail, arms with --with-arms)."""
    table = list(BASE_BODY_TABLE)
    if WITH_TAIL:
        table += TAIL_BODY_TABLE
    if WITH_ARMS:
        table += ARM_BODY_TABLE
    if WITH_JAW:
        table += JAW_BODY_TABLE
    return table


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--mode", default="inspect", choices=("inspect", "validate", "gen", "fit"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default="duckrex")
    ap.add_argument("--with-tail", action="store_true",
                    help="articulate the 2-DOF balance tail (16 joints)")
    ap.add_argument("--with-arms", action="store_true",
                    help="articulate the two SC09 forelimbs (18 joints with tail)")
    ap.add_argument("--with-jaw", action="store_true",
                    help="articulate the XL330 lower jaw (19 joints with tail+arms)")
    ap.add_argument("--no-xml-actuators", action="store_true",
                    help="omit XML position actuators (all joints are added by "
                         "mjlab actuator cfgs at build time — Phase-1 models)")
    ap.add_argument("--collision-parts", default=None,
                    help="comma-separated part-name substrings to give collision "
                         "geoms (ground-contact model for fall-recovery tasks)")
    ap.add_argument("--densities", default=None,
                      help="JSON of calibrated g/cm3 densities (fit mode output)")
    ap.add_argument("--mass-ledger", default=None,
                    help="Current CAD per-part mass_estimate.json; replaces legacy inertials")
    ap.add_argument("--imu-mounts", default=None,
                    help="CAD-hash-bound imu_mounts.json for nominal sensor frames")
    ap.add_argument(
        "--reference-xml",
        default=str(
            Path(__file__).resolve().parents[1]
            / "training/src/mjlab_microduck/robot/microduck/robot_walk.xml"
        ),
    )
    return ap.parse_args(argv)


def load():
    bpy.ops.wm.open_mainfile(filepath=args.blend)
    empties = {o.name: o for o in bpy.data.objects if o.type == "EMPTY"}
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    return BlendData(empties, meshes)


def M4(t, q):
    mm = np.eye(4)
    mm[:3, :3] = np.array(q.to_matrix())
    mm[:3, 3] = np.array(t)
    return mm


def q2np(q):
    return np.array([q.w, q.x, q.y, q.z])


def classify_skip(mats, name):
    if name in CURRENT_LEDGER_NAMES:
        return False
    joined = " ".join((m.name if m else "").lower() for m in mats)
    if not joined or any(s in joined for s in SKIP_MATERIALS):
        return True
    upper = name.upper()
    return any(s in upper for s in SKIP_NAMES)


# Explicit per-part material overrides (the CAD gives some parts several
# material slots; the first-slot rule handles most of them, but the design
# intent for selected parts is pinned here by part name).
NAME_CLASS_OVERRIDE = {
    # Rex_Chassis_Spine is a 3D-printed structure (graphite-structure shell,
    # armor panels, steel fasteners assembled separately) — never treat it as
    # solid steel even if a "fastener steel" slot shows up first.
    "REX_CHASSIS_SPINE": "printed",
}


def material_class(mats, name=""):
    for n, cls in NAME_CLASS_OVERRIDE.items():
        if n in name.upper():
            return cls
    # Composite meshes list several materials; the first slot is the primary
    # structure (e.g. spine = graphite+armor+steel -> print, not solid steel).
    mats = mats[:1]
    joined = " ".join((m.name if m else "").lower() for m in mats)
    for pats, cls in MATERIAL_CLASS:
        if any(p in joined for p in pats):
            return cls
    return "printed"


def anchor_body(o, empties):
    cur = o
    while cur is not None:
        n = cur.name
        # Tail: DCTL_Tail_Pitch parts ride the pitch body, DCTL_Tail_Yaw parts
        # (pitch-servo case, cradle, axle clamps) ride the yaw body.  Without
        # --with-tail the tail stays part of the trunk (static mass).
        if WITH_TAIL:
            if n == "DCTL_Tail_Pitch":
                return "tail_pitch"
            if n == "DCTL_Tail_Yaw":
                return "tail_yaw"
        if WITH_ARMS:
            if n == "DCTL_Arm_L":
                return "arm_l"
            if n == "DCTL_Arm_R":
                return "arm_r"
        if WITH_JAW:
            if n == "DCTL_Jaw_Hinge":
                return "jaw_hinge"
        if n.startswith("BODY_"):
            return n
        if n.startswith("CTRL_"):
            # Parts parented directly to CTRL rotate with its output body even
            # though BODY is a sibling in the CAD hierarchy. AXIS/FRAME parts
            # remain upstream. Resolve both cases without a FRAME-name guess.
            joint = n[len("CTRL_"):]
            for body, joint_name, _ in body_table():
                if joint_name == joint:
                    return "BODY_" + body
        if n in ("FRAME_trunk_base", "V12_Hip_Module"):
            return "FRAME_trunk_base"
        # FRAME_/AXIS_ sit BEFORE the joint's CTRL node. A part attached
        # there belongs to the upstream rigid body, not the similarly named
        # BODY_ downstream of CTRL. In particular, several S288 cases rotate
        # while their output disks/fasteners remain fixed to the upstream part.
        # Keep traversing actual ancestry instead of guessing from frame names.
        if n.startswith("DCTL_"):
            p = cur.parent
            while p is not None:
                if p.name.startswith("BODY_") or p.name in ("FRAME_trunk_base",):
                    return p.name
                p = p.parent
            return "FRAME_trunk_base"
        cur = cur.parent
    return "FRAME_trunk_base"


def body_parts(empties, meshes):
    parts = {b: [] for b, _, _ in body_table()}
    parts["trunk_base"] = []
    for o in meshes:
        if in_excluded_collection(o):
            continue
        a = anchor_body(o, empties)
        if a in parts:
            key = a
        elif a.startswith("BODY_"):
            key = a[len("BODY_"):]
        else:
            key = "trunk_base"
        if key in parts:
            parts[key].append(o)
    return parts


# ---- frame unwinding -----------------------------------------------------
def unwind(blend):
    """Return (zero_pose, home_pose, home_theta, rel, mz).

    zero_pose[b] : (loc, quat) of body b's frame at zero joint angle in CAD world
    home_pose[b] : (loc, quat) of the BODY_* empty (home stance)
    home_theta[b]: signed home angle of body b's joint (rad, about local +Z)
    rel[b]       : (loc, quat) of body b's zero frame relative to parent's zero frame
    mz[b]        : 4x4 world transform home-pose -> zero-angle pose (for meshes)
    """
    home = {}
    zero = {}
    theta = {}
    home["trunk_base"] = blend.pose("FRAME_trunk_base")
    zero["trunk_base"] = blend.pose("FRAME_trunk_base")
    for b, j, parent in body_table():
        a_t, a_q = blend.pose(AXIS_EMPTY[b])
        if b in FRAME_ROT_OVERRIDE:
            a_q = a_q @ FRAME_ROT_OVERRIDE[b]
        try:
            h_t, h_q = blend.pose("BODY_" + b)
        except KeyError:
            # Bodies without a BODY_* empty (the tail) are stored at zero angle:
            # home pose == the joint frame itself.
            h_t, h_q = a_t, a_q
        home[b] = (h_t, h_q)
        # signed home angle: A^-1 @ H is (ideally) a pure rotation about A's Z
        rel_q = a_q.inverted() @ h_q
        x0 = rel_q @ Vector((1.0, 0.0, 0.0))
        y0 = rel_q @ Vector((0.0, 1.0, 0.0))
        phi = math.atan2(x0.y, x0.x)
        zl = rel_q @ Vector((0.0, 0.0, 1.0))
        if abs(zl.z) < 0.99:
            print(f"[warn] {j}: home offset is not a pure Z rotation")
        theta[b] = phi
        # zero pose: Fz[b] = Fz[p] @ H[p]^-1 @ A[b]
        p_t, p_q = zero[parent]
        M = M4(p_t, p_q) @ np.linalg.inv(M4(home[parent][0], home[parent][1])) @ M4(a_t, a_q)
        zero[b] = (M[:3, 3], Matrix((tuple(M[0, :3]), tuple(M[1, :3]), tuple(M[2, :3]))).to_quaternion())
    rel = {}
    for b, j, parent in body_table():
        p_t, p_q = zero[parent]
        c_t, c_q = zero[b]
        q_rel = p_q.inverted() @ c_q
        loc_rel = np.array(p_q.to_matrix()).T @ (c_t - p_t)
        rel[b] = (loc_rel, q2np(q_rel))
    mz = {}
    for b in list(zero):
        mz[b] = M4(zero[b][0], zero[b][1]) @ np.linalg.inv(M4(home[b][0], home[b][1]))
    return zero, home, theta, rel, mz


# ---- geometry / inertia --------------------------------------------------
def mesh_tris_world(obj):
    """(M,3,3) triangle soup of a mesh object in world coords (fan-triangulated)."""
    me = obj.data
    verts = me.vertices
    tris = []
    for poly in me.polygons:
        idx = list(poly.vertices)
        for k in range(1, len(idx) - 1):
            tris.append((idx[0], idx[k], idx[k + 1]))
    out = np.zeros((len(tris), 3, 3))
    for t, tri in enumerate(tris):
        for i in range(3):
            out[t, i] = eff_matrix(obj) @ verts[tri[i]].co
    return out


def moments_from_tris(tri):
    M0 = 0.0
    M1 = np.zeros(3)
    M2 = np.zeros((3, 3))
    for a, b, c in tri:
        vol = np.linalg.det(np.stack([a, b, c])) / 6.0
        if abs(vol) < 1e-16:
            continue
        M0 += vol
        M1 += vol * (a + b + c) / 4.0
        for i in range(3):
            for j in range(3):
                M2[i, j] += (vol / 10.0) * (a[i] * a[j] + b[i] * b[j] + c[i] * c[j])
                M2[i, j] += (vol / 20.0) * (a[i] * b[j] + a[j] * b[i]
                                            + a[i] * c[j] + a[j] * c[i]
                                            + b[i] * c[j] + b[j] * c[i])
    return M0, M1, M2


def part_tris_body(obj, body_loc, body_quat):
    """Triangles of obj in the body frame (world -> R^T (v - p))."""
    Rm = np.array(body_quat.to_matrix())
    tris = mesh_tris_world(obj)
    return (Rm.T @ (tris.reshape(-1, 3) - body_loc).T).T.reshape(-1, 3, 3)


def part_tris_body_zero(obj, body_loc, body_quat, mz_body):
    """Triangles of obj in the part's ZERO-angle body frame (h->0 pulled back).

    mz_body maps home-world coords -> zero-world coords as a full affine:
    y = R @ x + t (composed 4x4, so apply the matrix, not R(x - t)).
    """
    Rm = np.array(body_quat.to_matrix())
    R = mz_body[:3, :3]
    t = mz_body[:3, 3]
    tris = mesh_tris_world(obj)
    w = (R @ tris.reshape(-1, 3).T).T + t
    return (Rm.T @ (w - body_loc).T).T.reshape(-1, 3, 3)


def compute_body_inertia(parts, zero_pose, mz, body, densities,
                         handled=frozenset(), servo_extra=()):
    """Volume-integrated inertia of one body.

    handled     : part names already accounted for by a servo group (skipped)
    servo_extra : [(mass_kg, com_body, I_cm_body)] from servo_groups()
    """
    body_loc, body_quat = zero_pose[body]
    mass = 0.0
    com = np.zeros(3)
    I_origin = np.zeros((3, 3))
    for o in parts.get(body, []):
        if o.name in handled:
            continue
        mats = list(o.data.materials) if o.data.materials else []
        if classify_skip(mats, o.name):
            continue
        cls = material_class(mats, o.name)
        rho = densities.get(cls, 1.25)
        tris_b = part_tris_body_zero(o, body_loc, body_quat, mz[body])
        v0, v1, v2 = moments_from_tris(tris_b)
        if v0 <= 0 or not (v0 == v0):
            continue
        m = rho * v0 * 1000.0
        if not m > 0:
            continue
        mass += m
        com += m * (v1 / v0)
        tr = v2[0, 0] + v2[1, 1] + v2[2, 2]
        I_origin += rho * 1000.0 * (np.eye(3) * tr - v2)
    for m, com_s, I_s in servo_extra:
        mass += m
        com += m * com_s
        # parallel axis: I about body origin from I about the servo's own CoM
        I_origin += I_s + m * (np.eye(3) * (com_s @ com_s) - np.outer(com_s, com_s))
    if mass <= 0:
        return None
    com = com / mass
    I_cm = I_origin - mass * (np.eye(3) * (com @ com) - np.outer(com, com))
    I_cm = regularize_inertia(I_cm)
    return mass, com, I_cm


def ledger_inertials(parts, zero, mz, ledger_path):
    """Use each current CAD ledger mass once; integrate its shape for inertia.

    These remain nominal estimates, not identified physical parameters. No
    old Microduck body inertial or additional servo mass is mixed into them.
    """
    with open(ledger_path, encoding="utf-8-sig") as stream:
        ledger = json.load(stream)
    rows = {row["name"]: row for row in ledger["rows"]}
    if len(rows) != len(ledger["rows"]):
        raise ValueError("Duplicate names in mass ledger")
    used, result, body_parts_audit = set(), {}, {}
    for body, objects in parts.items():
        mass, first, inertia_origin = 0.0, np.zeros(3), np.zeros((3, 3))
        names = []
        for obj in objects:
            if obj.name not in rows:
                raise ValueError(f"Physical mesh absent from current mass ledger: {obj.name}")
            if obj.name in used:
                raise ValueError(f"Part assigned twice: {obj.name}")
            used.add(obj.name)
            names.append(obj.name)
            row = rows[obj.name]
            m = float(row["mass_estimate_g"]) / 1000.0
            if m == 0:
                continue
            if not np.isfinite(m) or m < 0:
                raise ValueError(f"Invalid ledger mass: {obj.name}")
            tri = part_tris_body_zero(obj, *zero[body], mz[body])
            # Signed tetrahedron volume moments, vectorized. Orientation sign
            # cancels in the normalized moments (including mirrored meshes).
            a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
            volume = np.einsum('ij,ij->i', a, np.cross(b, c)) / 6.0
            v0 = volume.sum()
            if abs(v0) < 1e-15:
                raise ValueError(f"No closed volume for ledger part: {obj.name}")
            sums = a + b + c
            center = np.einsum('i,ij->j', volume, sums) / (4 * v0)
            second = (np.einsum('i,ij,ik->jk', volume, sums, sums)
                      + np.einsum('i,ij,ik->jk', volume, a, a)
                      + np.einsum('i,ij,ik->jk', volume, b, b)
                      + np.einsum('i,ij,ik->jk', volume, c, c)) / (20 * v0)
            mass += m
            first += m * center
            inertia_origin += m * (np.eye(3) * np.trace(second) - second)
        if mass <= 0:
            raise ValueError(f"No positive ledger mass in body {body}")
        center = first / mass
        inertia = inertia_origin - mass * (np.eye(3) * (center @ center) - np.outer(center, center))
        result[body] = (mass, center, regularize_inertia(inertia))
        body_parts_audit[body] = names
    missing = sorted(set(rows) - used)
    if missing:
        raise ValueError(f"Ledger parts omitted from conversion: {missing}")
    total = sum(value[0] for value in result.values())
    if abs(total * 1000 - ledger["mass_estimate_g"]) > 1e-6:
        raise ValueError("Converted total differs from ledger")
    with open(os.path.join(args.out, "mass_provenance.json"), "w", encoding="utf-8") as stream:
        json.dump({"cad_sha256": hashlib.sha256(open(args.blend, "rb").read()).hexdigest(),
                   "ledger_sha256": hashlib.sha256(open(ledger_path, "rb").read()).hexdigest(),
                   "mass_kg": total, "parts_count": len(used),
                   "physical_identification": False, "body_parts": body_parts_audit}, stream, indent=2)
    return result


def regularize_inertia(I):
    """Make the inertia tensor physical (positive-definite, triangle inequality).

    Tiny/thin parts (the 4 g forelimbs, the tail block) integrate to
    near-singular tensors; MuJoCo refuses non-positive eigenvalues.
    """
    eig = np.linalg.eigvalsh(I)
    if eig[0] < 1e-6:
        # floor the smallest eigenvalue at 1e-6 kg.m^2 — also keeps fmt() from
        # rounding tiny values to "0" in the XML
        I = I + np.eye(3) * (1e-6 - eig[0])
    # triangle inequality: A + B >= C for each principal diagonal pair
    for _ in range(3):
        a, b, c = I[0, 0], I[1, 1], I[2, 2]
        if a + b < c:
            I[2, 2] = a + b - 1e-9
        if a + c < b:
            I[1, 1] = a + c - 1e-9
        if b + c < a:
            I[0, 0] = b + c - 1e-9
    return I


# Real servo masses (kg).  The CAD represents some servos with several meshes
# (neck-pitch and the mouth servo each have a body plus case/disc parts), and
# the density fit cannot reproduce a real servo — so every physical servo is
# replaced by its datasheet mass, distributed with the CAD shape's inertia.
#   XL330-M288-T (ROBOTIS): 18 g
#   SC09 (Waveshare serial bus servo, metal gears): 20.5 g
SERVO_REAL_MASS = {"xl330": 0.018, "sc09": 0.0205, "s288": 0.0195}
SC09_PART_RE = re.compile(r"^SC09_(ARM_[LR]|PITCH|YAW)_(CASE|HORN|OUTPUT_GEAR)")


def part_tris_in_frame(obj, body_of_obj, owner, zero_pose, mz):
    """Mesh triangles in the ZERO-angle frame of body `owner`."""
    tris = mesh_tris_world(obj)
    R = mz[body_of_obj][:3, :3]
    t = mz[body_of_obj][:3, 3]
    w = (R @ tris.reshape(-1, 3).T).T + t
    bl, bq = zero_pose[owner]
    Rm = np.array(bq.to_matrix())
    return (Rm.T @ (w - bl).T).T.reshape(-1, 3, 3)


def servo_groups(parts, zero_pose, mz):
    """Group CAD servo meshes into physical servos.

    Returns (handled_names, {body: [(mass, com, I_cm), ...]}) where each group's
    mass is its datasheet value and the owner body is the one holding most of
    the group's volume.
    """
    groups = {}
    for b, objs in parts.items():
        for o in objs:
            mats = " ".join(m.name for m in (o.data.materials or []) if m).lower()
            key = None
            if "xl330" in mats:
                w = eff_matrix(o).translation
                key = ("xl330", round(w.x, 3), round(w.y, 3), round(w.z, 3))
            elif o.name.upper().startswith("S288_"):
                # S288 servo meshes are named S288_<joint>_<part>: one physical
                # servo per joint key. GUARD_* are armor screws, not servos —
                # leave them to the density path.
                m = re.match(r"^S288_(.+?)_(?:CASE|DISC|OUT|DIRECT|FIXED)", o.name)
                if m and not m.group(1).startswith("GUARD"):
                    key = ("s288", m.group(1))
            else:
                m = SC09_PART_RE.match(o.name)
                if m:
                    key = ("sc09", m.group(1))
            if key is not None:
                groups.setdefault(key, []).append((b, o))

    handled = set()
    per_body = {}
    for key, members in groups.items():
        kind = key[0]
        vols = []
        for b, o in members:
            tris = part_tris_body_zero(o, zero_pose[b][0], zero_pose[b][1], mz[b])
            v0, v1, v2 = moments_from_tris(tris)
            vols.append((v0, b, o))
            handled.add(o.name)
        vols.sort(key=lambda t: -t[0])
        if vols[0][0] <= 0:
            continue
        owner = vols[0][1]
        M0 = 0.0
        M1 = np.zeros(3)
        M2 = np.zeros((3, 3))
        for v0, b, o in vols:
            tris = part_tris_in_frame(o, b, owner, zero_pose, mz)
            a0, a1, a2 = moments_from_tris(tris)
            if a0 <= 0:
                continue
            M0 += a0
            M1 += a1
            M2 += a2
        if M0 <= 0:
            continue
        target = SERVO_REAL_MASS[kind]
        rho = target / M0  # kg per m^3
        com = M1 / M0
        tr = M2[0, 0] + M2[1, 1] + M2[2, 2]
        I_origin = rho * (np.eye(3) * tr - M2)
        I_cm = I_origin - target * (np.eye(3) * (com @ com) - np.outer(com, com))
        per_body.setdefault(owner, []).append((target, com, I_cm))
        print(f"[servo] {key}: {len(members)} mesh(es), vol={M0*1e6:.1f} cm3 -> "
              f"{target*1000:.1f} g on body '{owner}'")
    return handled, per_body


def fit_nnls(A, b, iters=6000):
    x = np.zeros(A.shape[1])
    gamma = 0.2 / (np.linalg.norm(A.T @ A, 2) + 1e-9)
    for _ in range(iters):
        nxt = np.maximum(0.0, x - gamma * (A.T @ (A @ x - b)))
        if np.linalg.norm(nxt - x) < 1e-12 * (np.linalg.norm(x) + 1.0):
            return nxt
        x = nxt
    return x


def fit_class_densities(blend, parts, zero_pose):
    classes = ["printed", "servo", "bearing", "battery"]
    rows = list(MICRODUCK_REF_INERTIAL)
    A = np.zeros((len(rows), len(classes)))
    for bi, body in enumerate(rows):
        for o in parts.get(body, []):
            mats = list(o.data.materials) if o.data.materials else []
            if classify_skip(mats, o.name):
                continue
            cls = material_class(mats, o.name)
            if cls not in classes:
                continue
            bl = zero_pose[body][0]
            bq = zero_pose[body][1]
            tris_b = part_tris_body(o, bl, bq)
            v0, _, _ = moments_from_tris(tris_b)
            if v0 > 0:
                A[bi, classes.index(cls)] += v0 * 1000.0
    b = np.array([MICRODUCK_REF_INERTIAL[k][0] for k in rows])
    x = fit_nnls(A, b)
    return dict(zip(classes, x))


# ---- STL / XML -----------------------------------------------------------
# Mesh budget is derived from the ENGINE's hard limit and from what the source
# CAD actually contains — never a hand-picked constant (2026-09-12 lesson).
#
# Why it matters: an earlier version capped every part at 700 triangles, which
# turned the CAD's big shells (head pan 50k tris, ankle 261k, arms 91k) into
# crude polyhedra that bulged through their neighbours — the model looked like
# everything was interpenetrating while the Blender source was clean.
#
# Policy: a body's merged visual mesh must fit MUJOCO_MESH_FACE_LIMIT.  If the
# CAD as-is fits, keep every triangle (no decimation at all).  Only when it does
# not, decimate — and then touch as FEW parts as possible, largest first, each
# down to its fair share of the budget.  Small parts are never degraded to make
# room for a big one.
MUJOCO_MESH_FACE_LIMIT = 200_000        # engine constant (stl_decoder: 1..200000)
BODY_BUDGET_SAFETY = 0.98               # headroom under the engine limit
COLLISION_TRIS_PER_PART = 2_500         # collision hulls only need shape, not detail


def tri_count(obj) -> int:
    """Triangle count after fan-triangulation (what the STL will hold)."""
    return sum(max(len(p.vertices) - 2, 0) for p in obj.data.polygons)


def allocate_tris(objs, limit=MUJOCO_MESH_FACE_LIMIT, safety=BODY_BUDGET_SAFETY):
    """Per-part TRIANGLE allowance so the body's merged mesh fits `limit`.

    Data-driven and conservative: if the CAD as-is already fits, every part keeps
    all its triangles.  Only when the body is over budget do we trim — starting
    from the LARGEST part and taking exactly the excess — so small parts (screws,
    bearings, tabs) are never degraded to make room for a big shell.
    """
    counts = [tri_count(o) for o in objs]
    budget = int(limit * safety)
    total = sum(counts)
    if total <= budget:
        return counts
    alloc = list(counts)
    need = total - budget
    for i in sorted(range(len(counts)), key=lambda k: -counts[k]):
        if need <= 0:
            break
        keep_floor = min(counts[i], 1_000)   # never obliterate a part
        give = min(need, max(counts[i] - keep_floor, 0))
        alloc[i] -= give
        need -= give
    return alloc


def tri_arrays(obj, max_tris):
    """(N,3,3) world-space triangle verts of obj, decimated to ~max_tris TRIANGLES.

    The budget is compared and applied in triangle terms (the STL's unit), not
    raw polygon counts — mixing the two made the allowance meaningless for
    quad-heavy meshes.
    """
    me = obj.data
    count = sum(max(len(p.vertices) - 2, 0) for p in me.polygons)
    owned = False
    if count > max_tris and not obj.modifiers.get("zdec"):
        mod = obj.modifiers.new("zdec", "DECIMATE")
        mod.ratio = max(max_tris / max(count, 1), 0.01)
        deps = bpy.context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(deps)
        me = bpy.data.meshes.new_from_object(ev)
        owned = True
    verts = me.vertices
    tris = []
    for p in me.polygons:
        idx = list(p.vertices)
        for k in range(1, len(idx) - 1):
            tris.append((idx[0], idx[k], idx[k + 1]))
    out = np.zeros((len(tris), 3, 3))
    for t, tri in enumerate(tris):
        for i in range(3):
            wv = eff_matrix(obj) @ verts[tri[i]].co
            out[t, i] = wv
    if owned:
        bpy.data.meshes.remove(me)
    return out.reshape(-1, 3)


def write_stl(path, tri_arrays):
    total = sum(t.shape[0] for t in tri_arrays)
    with open(path, "wb") as f:
        f.write(b"ZCode blend2mjcf" + b"\x00" * (80 - 16))
        f.write(struct.pack("<I", total))
        for t in tri_arrays:
            for tri in t:
                n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                nl = np.linalg.norm(n)
                n = n / nl if nl > 1e-12 else np.zeros(3)
                for vec in (n, tri[0], tri[1], tri[2]):
                    f.write(struct.pack("<3f", *(float(vec[i]) for i in range(3))))
                f.write(struct.pack("<H", 0))


def fmt(v, d=6):
    s = f"{float(v):.{d}f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def emit_xml(name, rel, inertials, outdir, sites, camera, trunk_z, collision_bodies=()):
    lines = []
    A = lines.append
    A('<?xml version="1.0" ?>')
    A(f'<!-- Generated by blend2mjcf.py from CAD assembly. Model: {name} -->')
    A('<mujoco model="%s">' % name)
    A('  <compiler angle="radian" meshdir="assets" autolimits="true" balanceinertia="true"/>')
    A('  <default>')
    A('    <default class="microduck">')
    A('      <joint frictionloss="0.1" armature="0.005"/>')
    A('      <position kp="50" dampratio="1"/>')
    A('      <default class="visual">')
    A('        <geom type="mesh" contype="0" conaffinity="0" group="2"/>')
    A('      </default>')
    A('      <default class="collision">')
    A('        <geom group="3"/>')
    A('      </default>')
    A('    </default>')
    A('    <default class="chosen_actuator">')
    A('      <geom contype="0" conaffinity="0"/>')
    A('      <joint damping="0.053" frictionloss="0.0048" armature="0.0018"/>')
    A('      <position kp="0.55" kv="0.0" forcerange="-0.96 0.96" ctrlrange="-10.0 10.0"/>')
    A('    </default>')
    A('    <default class="tail_actuator">')
    A('      <geom contype="0" conaffinity="0"/>')
    A('      <joint damping="0.01" frictionloss="0.01" armature="0.002"/>')
    A('      <position kp="2.0" kv="0.06" forcerange="-0.226 0.226" ctrlrange="-3.15 3.15"/>')
    A('    </default>')
    A('    <default class="self_collision_only">')
    A('      <geom group="3" contype="2" conaffinity="2"/>')
    A('    </default>')
    A('  </default>')
    A('  <sensor>')
    A('    <framequat name="orientation" objtype="site" noise="0.001" objname="imu"/>')
    A('    <gyro name="angular-velocity" site="imu" noise="0.005"/>')
    A('    <gyro name="imu_ang_vel" site="imu"/>')
    A('    <velocimeter name="imu_lin_vel" site="imu"/>')
    A('    <accelerometer name="imu_accel" site="imu"/>')
    if any(s[0] == "head_imu" for entries in sites.values() for s in entries):
        A('    <gyro name="head_imu_ang_vel" site="head_imu"/>')
        A('    <accelerometer name="head_imu_accel" site="head_imu"/>')
        A('    <framequat name="head_imu_orientation" objtype="site" objname="head_imu"/>')
    A('    <subtreeangmom name="root_angmom" body="trunk_base"/>')
    A('  </sensor>')
    A('  <worldbody>')
    m, com, icm = inertials["trunk_base"]
    A('    <body name="trunk_base" pos="0 0 %s" quat="1 0 0 0" childclass="microduck">' % fmt(trunk_z))
    A('      <freejoint name="trunk_base_freejoint"/>')
    A('      <inertial pos="%s" mass="%.6g" fullinertia="%s"/>' % (
        " ".join(fmt(x) for x in com), m,
        " ".join(fmt(x) for x in (icm[0, 0], icm[1, 1], icm[2, 2], icm[0, 1], icm[0, 2], icm[1, 2]))))
    A('      <geom type="mesh" class="visual" mesh="%s_trunk_base" material="%s_material"/>' % (name, name))
    if "trunk_base" in collision_bodies:
        A('      <geom type="mesh" name="trunk_base_collision" class="collision" mesh="%s_trunk_base_collision" material="%s_material"/>' % (name, name))
    for sname, spos, squat in sites.get("trunk_base", []):
        A('      <site group="3" name="%s" pos="%s" quat="%s"/>' % (
            sname, " ".join(fmt(x) for x in spos), " ".join(fmt(x) for x in squat)))
    if camera is not None:
        A('      <site group="3" name="head_camera" pos="%s" quat="%s"/>' % (
            " ".join(fmt(x) for x in camera[0]), " ".join(fmt(x) for x in camera[1])))
        A('      <camera name="head_camera" pos="%s" quat="%s"/>' % (
            " ".join(fmt(x) for x in camera[0]), " ".join(fmt(x) for x in camera[1])))
    children = {}
    for b, j, parent in body_table():
        children.setdefault(parent, []).append((b, j))

    def rec(parent, depth):
        indent = "    " * (depth + 2)
        for b, joint in children.get(parent, []):
            loc_rel, quat_rel = rel[b]
            m, com, icm = inertials[b]
            A('%s<body name="%s" pos="%s" quat="%s">' % (
                indent, b, " ".join(fmt(x) for x in loc_rel), " ".join(fmt(x) for x in quat_rel)))
            rng = JOINT_RANGES[joint]
            jclass = "tail_actuator" if (joint.startswith("tail_") or joint.startswith("arm_")) else "chosen_actuator"
            A('%s  <joint axis="0 0 1" name="%s" type="hinge" range="%s %s" class="%s"/>' % (
                indent, joint, fmt(rng[0]), fmt(rng[1]), jclass))
            A('%s  <inertial pos="%s" mass="%.6g" fullinertia="%s"/>' % (
                indent, " ".join(fmt(x) for x in com), m,
                " ".join(fmt(x) for x in (icm[0, 0], icm[1, 1], icm[2, 2], icm[0, 1], icm[0, 2], icm[1, 2]))))
            A('%s  <geom type="mesh" class="visual" mesh="%s_%s" material="%s_material"/>' % (indent, name, b, name))
            if b in collision_bodies:
                A('%s  <geom type="mesh" name="%s_collision" class="collision" mesh="%s_%s_collision" material="%s_material"/>' % (
                    indent, b, name, b, name))
            if b in ("ankle_left", "ankle_right"):
                foot = "left" if b.endswith("_left") else "right"
                A('%s  <geom type="mesh" name="%s_foot_collision" class="collision" mesh="%s_%s_foot" material="%s_material"/>' % (
                    indent, foot, name, b, name))
            if b in ("leg", "leg_2") and b not in collision_bodies:
                A('%s  <geom type="mesh" class="self_collision_only" mesh="%s_%s" material="%s_material"/>' % (
                    indent, name, b, name))
            for sname, spos, squat in sites.get(b, []):
                A('%s  <site group="3" name="%s" pos="%s" quat="%s"/>' % (
                    indent, sname, " ".join(fmt(x) for x in spos), " ".join(fmt(x) for x in squat)))
            rec(b, depth + 1)
            A('%s</body>' % indent)

    rec("trunk_base", 0)
    A('    </body>')
    A('  </worldbody>')
    A('  <asset>')
    A('    <mesh file="%s_trunk_base.stl"/>' % name)
    if "trunk_base" in collision_bodies:
        A('    <mesh file="%s_trunk_base_collision.stl"/>' % name)
    for b, _, _ in body_table():
        A('    <mesh file="%s_%s.stl"/>' % (name, b))
        if b in collision_bodies:
            A('    <mesh file="%s_%s_collision.stl"/>' % (name, b))
        if b in ("ankle_left", "ankle_right"):
            A('    <mesh file="%s_%s_foot.stl"/>' % (name, b))
    A('    <material name="%s_material" rgba="0.72 0.55 0.42 1"/>' % name)
    A('  </asset>')
    A('  <actuator>')
    if not NO_XML_ACTUATORS:
        for b, joint, parent in body_table():
            # The tail/arm/jaw joints get their actuators from mjlab's
            # BuiltinPositionActuatorCfg (S288 gains); emitting them here too
            # would duplicate the actuator (they are all S288 servos, so they
            # belong to the mjlab-owned group, not to the XML class).
            if joint.startswith(("tail_", "arm_", "jaw_")):
                continue
            A('    <position class="chosen_actuator" name="%s" joint="%s"/>' % (joint, joint))
    A('  </actuator>')
    A('  <equality/>')
    A('</mujoco>')
    path = os.path.join(outdir, "robot_%s.xml" % name)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)


# ------------------------------------------------------------------ main
args = None
blend = None


def main():
    global args, blend, CURRENT_LEDGER_NAMES
    args = parse_args()
    CURRENT_LEDGER_NAMES = set()
    if args.mass_ledger:
        with open(args.mass_ledger, encoding="utf-8-sig") as stream:
            CURRENT_LEDGER_NAMES = {row["name"] for row in json.load(stream)["rows"]}
    global WITH_TAIL, WITH_ARMS, WITH_JAW, COLLISION_PARTS, NO_XML_ACTUATORS
    WITH_TAIL = bool(args.with_tail)
    WITH_ARMS = bool(args.with_arms)
    WITH_JAW = bool(args.with_jaw)
    NO_XML_ACTUATORS = bool(args.no_xml_actuators)
    if args.collision_parts in (None, "", "none"):
        COLLISION_PARTS = None
    elif args.collision_parts == "auto":
        COLLISION_PARTS = DEFAULT_COLLISION_PARTS
    else:
        COLLISION_PARTS = tuple(p.strip().upper() for p in args.collision_parts.split(","))
    blend = load()

    if args.mode == "inspect":
        for b, j, p in body_table():
            try:
                t, q = blend.pose(AXIS_EMPTY[b])
                print(f"{b:24s} {j:16s} loc={np.round(t, 4)} quat={np.round(q2np(q), 4)}")
            except KeyError:
                print(f"{b:24s} MISSING")
        parts = body_parts(blend.empties, blend.meshes)
        for b in sorted(parts):
            print(f"{b:24s} parts={len(parts[b])}")
        sys.exit(0)

    if args.mode == "fit":
        # Calibrate material densities on the microduck reference CAD, verify the
        # inertia pipeline against robot_walk.xml, and dump the per-joint frame
        # twist (XML import frame vs CAD AXIS frame) for the generator to rebase
        # the reference inertials into any CAD assembly of the same convention.
        import json as _json
        zero, home, theta, rel, mz = unwind(blend)
        parts = body_parts(blend.empties, blend.meshes)
        dens = fit_class_densities(blend, parts, zero)
        merged = dict(DENSITY_FALLBACK)
        merged.update({k: v for k, v in dens.items() if v > 1e-6})
        print("[fit] densities (g/cm3):", {k: round(v, 3) for k, v in merged.items()})

        # accumulate the reference XML body frames (world) and the twist vs the
        # CAD AXIS frames
        tree = ET.parse(args.reference_xml)
        root = tree.getroot()
        xmlb = {c.get("name"): c for c in root.iter("body")}
        parent_of = {}

        def scan(el, parent):
            for c in el:
                if c.tag in ("worldbody",):
                    scan(c, parent)
                elif c.tag == "body":
                    parent_of[c.get("name")] = parent
                    scan(c, c.get("name"))
        scan(root, None)

        def xml_abs(name):
            chain = []
            n = name
            while n is not None:
                chain.append(n)
                n = parent_of.get(n)
            p = np.zeros(3)
            q = Quaternion((1.0, 0.0, 0.0, 0.0))
            for nm in reversed(chain):
                b = xmlb[nm]
                pp = np.array([float(x) for x in b.get("pos").split()])
                pq = [float(x) for x in b.get("quat", "1 0 0 0").split()]
                qq = Quaternion((pq[0], pq[1], pq[2], pq[3]))
                p = q @ Vector(pp) + Vector(p)
                q = q @ qq
            return q, p

        refs = {}
        ok = True
        for body, (mref, cref, iref) in MICRODUCK_REF_INERTIAL.items():
            r = compute_body_inertia(parts, zero, mz, body, merged)
            if r is not None:
                m, c, icm = r
                err = abs(m - mref) / mref
                tag = "OK " if err < 0.15 else "BAD"
                if err >= 0.15:
                    ok = False
                print(f"{tag} {body:22s} mass={m:.4f} ref={mref:.4f} err={err*100:.1f}%")
            # twist: Q_xml_world = Q_cad @ Rz(psi)  ->  in CAD coords, Rz(psi)
            xq, xp = xml_abs(body)
            cadq = zero[body][1] if body in zero else blend.pose("BODY_" + body)[1]
            rel_tw = cadq.inverted() @ xq
            # verify pure Z in the CAD frame
            zls = rel_tw @ Vector((0.0, 0.0, 1.0))
            if abs(zls.z) < 0.99:
                print(f"[warn] {body}: non-Z twist")
            x0 = rel_tw @ Vector((1.0, 0.0, 0.0))
            y0 = rel_tw @ Vector((0.0, 1.0, 0.0))
            psi = math.atan2(x0.y, x0.x)
            refs[body] = {"psi": psi, "mass": mref,
                          "com": list(cref),
                          "I": [iref[0], iref[1], iref[2], iref[3], iref[4], iref[5]]}
            print(f"    {body:22s} twist_psi={np.degrees(psi):+8.2f} deg")
        if args.densities:
            with open(args.densities, "w") as f:
                _json.dump({"densities": merged, "refs": refs}, f, indent=1)
            print("wrote", args.densities)
        print("FIT", "PASSED" if ok else "FAILED (mass errors remain)")
        sys.exit(0)

    if args.mode == "validate":
        zero, home, theta, rel, mz = unwind(blend)
        ok = True
        for b, j, parent in body_table():
            tr, qr = zero[b]
            axis = qr @ Vector((0.0, 0.0, 1.0))
            if j not in EXPECT_AXIS:
                print(f"SKIP {b:16s} {j:16s} (no reference axis)")
                continue
            exp = np.array(EXPECT_AXIS[j])
            err = min(np.linalg.norm(np.array(axis) - exp),
                      np.linalg.norm(np.array(axis) + exp))
            tag = "OK " if err < 0.07 else "MISMATCH"
            if tag != "OK ":
                ok = False
            print(f"{tag} {b:16s} {j:16s} zero-axis=({axis.x:+.3f},{axis.y:+.3f},{axis.z:+.3f}) "
                  f"expected=({exp[0]:+.3f},{exp[1]:+.3f},{exp[2]:+.3f}) com_home_theta={theta[b]:+.4f}")
        print("VALIDATION", "PASSED" if ok else "FAILED")
        # home angles vs microduck HOME_FRAME reference
        HOME_REF = {"left_hip_yaw": 0.0, "left_hip_roll": -0.0873, "left_hip_pitch": -0.4579,
                    "left_knee": -0.0049, "left_ankle": 0.4530, "neck_pitch": 0.3491,
                    "head_pitch": 0.3491, "head_yaw": 0.0, "head_roll": 0.0,
                    "right_hip_yaw": 0.0, "right_hip_roll": 0.0873, "right_hip_pitch": 0.4579,
                    "right_knee": 0.0049, "right_ankle": -0.4530}
        for b, j, parent in body_table():
            ref = HOME_REF.get(j)
            print(f"  HOME {j:16s} cad={theta[b]:+.4f}  microduck_ref=" + (f"{ref:+.4f}" if ref is not None else "n/a"))
        sys.exit(0 if ok else 1)

    # ---- gen
    assert args.out, "--out required for gen"
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "assets"), exist_ok=True)

    zero, home, theta, rel, mz = unwind(blend)
    print("=== HOME joint angles extracted from CAD (rad) ===")
    for b, j, parent in body_table():
        print(f"  {j:16s} {theta[b]:+.4f}")

    parts = body_parts(blend.empties, blend.meshes)

    import json as _json
    ref_inertials = {}
    if args.densities:
        with open(args.densities) as f:
            calib = _json.load(f)
        merged = calib["densities"]
        for b, r in calib.get("refs", {}).items():
            ref_inertials[b] = (r["mass"], np.array(r["com"]),
                                np.array([[r["I"][0], r["I"][3], r["I"][4]],
                                          [r["I"][3], r["I"][1], r["I"][5]],
                                          [r["I"][4], r["I"][5], r["I"][2]]]))
        print("[gen] densities from", args.densities)
    else:
        merged = dict(DENSITY_FALLBACK)

    # Bodies whose parts are shared with microduck reuse the real-robot inertials;
    # the changed ones (new trunk with tail-balance module, new feet) are measured.
    # Servo meshes are pulled out of the volume integration and replaced by their
    # datasheet masses (one entry per physical servo, not per CAD mesh).
    GEOM_BODIES = ("trunk_base", "ankle_left", "ankle_right")
    handled_servos, servo_by_body = (set(), {}) if args.mass_ledger else servo_groups(parts, zero, mz)
    inertials = {}
    for b in ([] if args.mass_ledger else list(parts)):
        if b not in GEOM_BODIES:
            ref = ref_inertials.get(b)
            if ref is not None:
                inertials[b] = (ref[0], ref[1], ref[2])
                continue
            ref = MICRODUCK_REF_INERTIAL.get(b)
            if ref:
                mass, com, I = ref[0], np.array(ref[1]), np.array([[ref[2][0], ref[2][3], ref[2][4]],
                                                                    [ref[2][3], ref[2][1], ref[2][5]],
                                                                    [ref[2][4], ref[2][5], ref[2][2]]])
                inertials[b] = (mass, com, I)
                continue
        r = compute_body_inertia(parts, zero, mz, b, merged,
                                 handled=handled_servos,
                                 servo_extra=servo_by_body.get(b, ()))
        if r is None:
            inertials[b] = (0.01, np.zeros(3), np.eye(3) * 1e-6)
        else:
            inertials[b] = r

    if args.mass_ledger:
        inertials = ledger_inertials(parts, zero, mz, args.mass_ledger)

    print("=== BODY MASSES (kg) ===")
    total = 0.0
    for b in sorted(inertials):
        m, c, icm = inertials[b]
        total += m
        print(f"  {b:24s} m={m:.4f}  com={np.round(c, 4)}")
    print(f"  TOTAL: {total:.4f} kg")

    # ---- sites (measured at home, then zeroed like the meshes)
    def site_rel(body, loc_world):
        bl, bq = zero[body]
        Rm = np.array(bq.to_matrix())
        w = mz[body] @ np.append(np.array(loc_world), 1.0)
        return Rm.T @ (w[:3] - bl)

    sites = {}
    if "IMU_REFERENCE_imu" in blend.empties:
        sites["trunk_base"] = [("imu", site_rel("trunk_base",
                                                blend.empties["IMU_REFERENCE_imu"].matrix_world.translation),
                               (1.0, 0.0, 0.0, 0.0))]
    else:
        # New CAD (MicroDinosaur v1) has no IMU reference empty — the JY61P
        # board is modelled as a part. Put the IMU site at the trunk centre.
        sites["trunk_base"] = [("imu", (0.0, 0.0, 0.03), (1.0, 0.0, 0.0, 0.0))]
    for foot, body in (("left", "ankle_left"), ("right", "ankle_right")):
        for o in parts.get(body, []):
            if o.name.startswith("foot_" + foot):
                bb = [eff_matrix(o) @ Vector(c) for c in o.bound_box]
                cx = sum(v.x for v in bb) / 8
                cy = sum(v.y for v in bb) / 8
                cz = min(v.z for v in bb)
                sites[body] = [(foot + "_foot", site_rel(body, (cx, cy, cz + 0.008)), (1.0, 0.0, 0.0, 0.0))]
    if "IMU_REFERENCE_head_imu" in blend.empties:
        sites.setdefault("jaw_soft", []).append(
            ("head_imu", site_rel("jaw_soft",
                                  blend.empties["IMU_REFERENCE_head_imu"].matrix_world.translation),
             (1.0, 0.0, 0.0, 0.0)))

    if args.imu_mounts:
        with open(args.imu_mounts, encoding="utf-8-sig") as stream:
            mounts = json.load(stream)
        cad_hash = hashlib.sha256(open(args.blend, "rb").read()).hexdigest()
        if mounts["model_sha256"] != cad_hash:
            raise ValueError("IMU metadata belongs to a different CAD hash")
        transform = np.array(mounts["head"]["T_world_sensor_reference_m"])
        body = mounts["head"]["parent"].removeprefix("BODY_")
        rotation = np.array(zero[body][1].to_matrix()).T @ mz[body][:3, :3] @ transform[:3, :3]
        quat = Matrix(rotation.tolist()).to_quaternion().normalized()
        sites[body] = [s for s in sites.get(body, []) if s[0] != "head_imu"]
        sites[body].append(("head_imu", site_rel(body, transform[:3, 3]), tuple(quat)))
        # Keep body-aligned axes for the existing 81D policy; its physical
        # sensor-to-body rotation is not calibrated. Place it at the actual
        # nominal PCB center rather than the old guessed trunk +30 mm.
        body_center = np.array(mounts["body"]["reference_board_center_mm"]) / 1000.0
        sites["trunk_base"] = [s for s in sites["trunk_base"] if s[0] != "imu"]
        sites["trunk_base"].append(("imu", site_rel("trunk_base", body_center), (1., 0., 0., 0.)))
        print("[imu] imported CAD-hash-bound nominal mounts; hardware axis calibration remains pending")

    # Camera: the real camera is the M12 lens + lens holder + Pi carrier, all
    # parented to BODY_jaw_soft in the CAD (bbox z~0.237-0.253 = the head, well
    # above the trunk at 0.117).  The old code put the site into emit_xml's
    # `camera` argument, which is written INTO THE TRUNK body block, so the
    # emitted "head_camera" sat on trunk_base at a wrong (sign-flipped) offset.
    # Emit it through `sites` instead -- same path as head_imu and the foot
    # sites -- so it lands on the body it belongs to.
    camera = None
    for o in parts.get("jaw_soft", []):
        if "lens" in o.name.lower():
            bb = [eff_matrix(o) @ Vector(c) for c in o.bound_box]
            center = Vector((sum(v.x for v in bb) / 8, sum(v.y for v in bb) / 8, sum(v.z for v in bb) / 8))
            q_track = Vector((1.0, 0.0, 0.0)).to_track_quat("-Z", "Y")
            bq = zero["jaw_soft"][1]
            q_rel = (bq.inverted() @ q_track)
            sites.setdefault("jaw_soft", []).append(
                ("head_camera", site_rel("jaw_soft", center), (q_rel.w, q_rel.x, q_rel.y, q_rel.z)))
            print(f"[camera] lens={o.name} 中心(CAD世界)={tuple(round(v, 4) for v in center)} -> 挂到 jaw_soft")
            break

    # Collision parts: parts matching the requested substrings get their own
    # merged collision STL per body (ground-contact model for fall-recovery).
    collision_bodies = []
    if COLLISION_PARTS:
        for b in list(parts):
            hits = [o for o in parts[b]
                    if not classify_skip(list(o.data.materials or []), o.name)
                    and any(k in o.name.upper() for k in COLLISION_PARTS)]
            if hits:
                collision_bodies.append(b)
                print(f"[collision] {b}: {len(hits)} part(s) -> {','.join(o.name for o in hits[:5])}")
    emit_xml(args.name, rel, inertials, args.out, sites, camera,
             zero["trunk_base"][0][2], collision_bodies=collision_bodies)

    # ---- STLs (body-local zero-frame merged; foot collision mesh separate)
    for b in list(parts):
        body_loc, body_quat = zero[b]
        Rm = np.array(body_quat.to_matrix())
        Rw = mz[b][:3, :3]
        tv = mz[b][:3, 3]
        arrays = []
        coll_arrays = []
        foot_arrays = None
        body_objs = [o for o in parts[b]
                     if not classify_skip(list(o.data.materials) if o.data.materials else [], o.name)]
        alloc = allocate_tris(body_objs)
        total_cad = sum(tri_count(o) for o in body_objs)
        if total_cad > int(MUJOCO_MESH_FACE_LIMIT * BODY_BUDGET_SAFETY):
            print(f"[mesh] {b}: CAD {total_cad} tris -> 合并后 {sum(alloc)} tris "
                  f"(引擎上限 {MUJOCO_MESH_FACE_LIMIT})")
        for o, allow in zip(body_objs, alloc):
            tris_world = tri_arrays(o, max(allow, 1))
            if len(tris_world) == 0:
                continue
            w = (Rw @ tris_world.T).T + tv
            tris_body = (Rm.T @ (w - body_loc).T).T.reshape(-1, 3, 3)
            if o.name.startswith(("foot_left", "foot_right")):
                foot_arrays = [tris_body]
            else:
                arrays.append(tris_body)
            if COLLISION_PARTS and any(k in o.name.upper() for k in COLLISION_PARTS):
                coll_world = tri_arrays(o, COLLISION_TRIS_PER_PART)
                wc = (Rw @ coll_world.T).T + tv
                coll_arrays.append((Rm.T @ (wc - body_loc).T).T.reshape(-1, 3, 3))
        write_stl(os.path.join(args.out, "assets", "%s_%s.stl" % (args.name, b)),
                  arrays if arrays else [np.zeros((1, 3, 3))])
        if coll_arrays:
            write_stl(os.path.join(args.out, "assets", "%s_%s_collision.stl" % (args.name, b)),
                      coll_arrays)
        if foot_arrays is not None:
            write_stl(os.path.join(args.out, "assets", "%s_%s_foot.stl" % (args.name, b)), foot_arrays)
        print("wrote STL", b)

    print("DONE")


if __name__ == "__main__":
    main()
