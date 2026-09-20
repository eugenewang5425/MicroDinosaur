"""CPU-only invariants for the two new roller task families:

  - Mjlab-SitStand-Flat-MicroDuck-Rollers (posture-conditioned sit↔stand)
  - Mjlab-Figure8-Flat-MicroDuck-Rollers  (figure-8 path, moving heading target)

Same contract as the other cfg tests: joint indices resolve on the real model,
heights are roller-specific, penalty weights carry the intended sign, and the
task registry sees the tasks.
"""

import math

import pytest


def _sitstand_cfg():
    from mjlab_microduck.tasks.microduck_roller_sitstand_env_cfg import (
        make_microduck_roller_sitstand_env_cfg,
    )

    return make_microduck_roller_sitstand_env_cfg()


def _figure8_cfg():
    from mjlab_microduck.tasks.microduck_figure8_rollers_env_cfg import (
        make_microduck_figure8_rollers_env_cfg,
    )

    return make_microduck_figure8_rollers_env_cfg()


# ── registration ─────────────────────────────────────────────────────────────

def test_both_tasks_registered():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401

    tasks = list_tasks()
    assert "Mjlab-SitStand-Flat-MicroDuck-Rollers" in tasks
    assert "Mjlab-Figure8-Flat-MicroDuck-Rollers" in tasks


# ── sit-stand: keyframe, heights, indices ────────────────────────────────────

def test_sitstand_uses_roller_heights():
    from mjlab_microduck.tasks.microduck_roller_sitstand_env_cfg import (
        ROLLER_SIT_Z,
        ROLLER_STAND_Z,
    )

    cfg = _sitstand_cfg()
    command = cfg.commands["twist"]
    assert command.sit_z == ROLLER_SIT_Z == 0.063
    assert command.stand_z == ROLLER_STAND_Z == 0.138
    # The posture stack must read the same numbers as the command, or the
    # policy is rewarded for a height the command never asks for.
    for name in ("posture_height", "posture_height_sharp", "posture_height_l1",
                 "posture_stillness", "posture_composite"):
        params = cfg.rewards[name].params
        assert params["sit_z"] == ROLLER_SIT_Z
        assert params["stand_z"] == ROLLER_STAND_Z


def test_sitstand_keyframe_is_the_verified_roller_pose():
    from mjlab_microduck.tasks.microduck_roller_sitstand_env_cfg import (
        ROLLER_SITTING_TARGET_OVERRIDES as O,
    )

    # The pose verified by scripts/search_roller_sit.py (20/20 stable, 2.9°).
    assert O[2] == pytest.approx(-0.80)   # left hip_pitch
    assert O[3] == pytest.approx(1.20)    # left knee
    assert O[1] == pytest.approx(0.20)    # left hip_roll (splay)
    assert O[4] == pytest.approx(0.00)    # left ankle flat
    # Right leg mirrors the left.
    assert O[11] == pytest.approx(-O[2])
    assert O[12] == pytest.approx(-O[3])
    assert O[10] == pytest.approx(-O[1])
    assert O[13] == pytest.approx(-O[4])


def test_sitstand_joint_indices_are_servo_space():
    """joint_indices are SERVO-space: on the roller model the 14 servos sit at
    [0-4, 7-10, 11-15] of the 18-joint model order, so a model-space index >= 14
    would blow up the 14-column joint_pos view (the bug fixed in roller_standup).
    """
    import mujoco

    from mjlab_microduck.robot.microduck_constants import get_walk_rollers_spec
    from mjlab_microduck.tasks.microduck_roller_sitstand_env_cfg import _LEG_JOINTS

    model = get_walk_rollers_spec().compile()
    articulated = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
    ]
    servo = [n for n in articulated if not n.startswith("passive_")]
    assert len(servo) == 14
    assert [servo[i] for i in _LEG_JOINTS] == [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ]
    # Keyframe overrides must land inside the servo view too.
    from mjlab_microduck.tasks.microduck_roller_sitstand_env_cfg import (
        ROLLER_SITTING_TARGET_OVERRIDES,
    )
    assert max(ROLLER_SITTING_TARGET_OVERRIDES) < len(servo)


def test_sitstand_skating_rewards_removed():
    cfg = _sitstand_cfg()
    for name in ("wheel_speed", "braking", "glide", "single_support",
                 "skating_air_time", "gait_symmetry", "forward_lean",
                 "heading_hold", "feet_flat", "hip_roll_neutral",
                 "pose", "upright", "com_height_target"):
        assert name not in cfg.rewards, f"{name} should be replaced/removed"


def test_sitstand_penalties_use_positive_weights():
    """The three gentleness functions already return <= 0 → positive weights.

    A negative weight double-negates into a reward for the violation (the
    walker env's crash-sit bug). Same rule as roller_standup's gentle_rise.
    """
    cfg = _sitstand_cfg()
    assert cfg.rewards["descent_speed"].weight > 0
    assert cfg.rewards["gentle_motion"].weight > 0
    # rise_speed starts disabled and is phased in by curriculum.
    assert cfg.rewards["rise_speed"].weight == 0.0
    assert "rise_speed_weight" in cfg.curriculum


def test_sitstand_episode_and_command_dwell():
    cfg = _sitstand_cfg()
    assert cfg.episode_length_s == 12.0
    command = cfg.commands["twist"]
    assert command.resampling_time_range == (3.5, 6.5)
    assert command.sit_prob == 0.5
    assert command.ramp_s == 2.0


# ── figure-8: command geometry ───────────────────────────────────────────────

def test_figure8_command_replaces_heading_hold():
    cfg = _figure8_cfg()
    assert "heading_hold" not in cfg.rewards
    assert "heading_tracking" in cfg.rewards
    command = cfg.commands["twist"]
    assert type(command).__name__ == "Figure8VelocityCommandCfg"
    assert command.period_s == 8.0
    assert command.radius == 0.8
    assert command.throttle_min == 0.0 and command.throttle_max == 0.5


def test_gerono_lemniscate_is_actually_a_figure_eight():
    """The parameterisation must trace an 8: two x-lobes, y antisymmetric.

    Uses the same formulas as Figure8VelocityCommand._update_command.
    """
    R = 0.8
    pts = []
    for i in range(721):
        theta = 2.0 * math.pi * i / 720.0
        pts.append((R * math.cos(theta), R * math.sin(theta) * math.cos(theta)))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    # Full x swing from -R to +R (the two lobes).
    assert min(xs) == pytest.approx(-R, abs=1e-6)
    assert max(xs) == pytest.approx(R, abs=1e-6)
    # y peaks at ±R/2 (the lobe height).
    assert max(ys) == pytest.approx(R / 2.0, abs=1e-6)
    assert min(ys) == pytest.approx(-R / 2.0, abs=1e-6)
    # Closed curve: returns to the start after one period.
    assert pts[0] == pytest.approx(pts[-1], abs=1e-9)
    # The defining property of an 8 (as opposed to a circle or an ellipse):
    # the two lobes touch at a single point, the origin — x ≈ 0 happens only
    # at y ≈ 0, nowhere else on the curve.
    for x, y in pts:
        if abs(x) < 1e-3:
            assert abs(y) < 1e-3
    # Axis symmetry of the Gerono lemniscate (NOT point symmetry):
    # y(θ) = −y(−θ) about the x axis, x(θ) = −x(π−θ) about the y axis.
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(len(pts) - 1 - i) % (len(pts) - 1)]  # θ → −θ
        assert x2 == pytest.approx(x1, abs=1e-6)
        assert y2 == pytest.approx(-y1, abs=1e-6)


def test_figure8_throttle_sweeps_push_and_coast():
    """cmd[0] must actually sweep the configured range over one period."""
    v_min, v_max = 0.0, 0.5
    vals = []
    for i in range(721):
        theta = 2.0 * math.pi * i / 720.0
        ramp = 0.5 + 0.5 * math.cos(2.0 * theta)
        vals.append(v_min + (v_max - v_min) * ramp)
    assert min(vals) == pytest.approx(v_min, abs=1e-6)
    assert max(vals) == pytest.approx(v_max, abs=1e-6)


# ── stride: the skating push ─────────────────────────────────────────────────

def _stride_cfg():
    from mjlab_microduck.tasks.microduck_stride_rollers_env_cfg import (
        make_microduck_stride_rollers_env_cfg,
    )

    return make_microduck_stride_rollers_env_cfg()


def test_stride_task_registered():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401

    assert "Mjlab-Stride-Flat-MicroDuck-Rollers" in list_tasks()


def test_stride_relaxes_hip_roll_and_strengthens_symmetry():
    """The two base weights that produced the walk-like lopsided gait."""
    from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
        make_microduck_velocity_rollers_env_cfg,
    )

    base = make_microduck_velocity_rollers_env_cfg()
    cfg = _stride_cfg()

    # Centring pull must be weaker than the base (it fights abduction) but not
    # zero (the stance must still not park splayed on the joint stops).
    assert base.rewards["hip_roll_neutral"].weight == -2.0
    assert cfg.rewards["hip_roll_neutral"].weight == -0.3
    # Symmetry penalty must be stronger (measured L/R travel 0.94 vs 0.52 rad).
    assert base.rewards["gait_symmetry"].weight == -1.0
    assert cfg.rewards["gait_symmetry"].weight == -3.0


def test_stride_new_rewards_and_sign_convention():
    """lateral_push returns >= 0 → positive weight; pitch_swing self-negates →
    positive weight (a negative weight would reward the walk-like swing)."""
    cfg = _stride_cfg()
    assert cfg.rewards["lateral_push"].weight == 2.0
    assert cfg.rewards["pitch_swing"].weight == 0.5
    # Both are gated on the same contact sensor / command as the other stride terms.
    for name in ("lateral_push", "pitch_swing"):
        params = cfg.rewards[name].params
        assert params["sensor_name"] == "feet_ground_contact"
        assert params["command_name"] == "twist"
        assert params["vel_gate_ref"] == 0.2
    # The pitch free band must be generous enough not to forbid hip extension.
    assert cfg.rewards["pitch_swing"].params["free_band"] == 0.35


def test_stride_reward_helpers_exist_and_resolve_joints():
    """Both helpers must resolve their joints BY NAME on the roller model."""
    import mujoco

    from mjlab_microduck.robot.microduck_constants import get_walk_rollers_spec
    from mjlab_microduck.tasks import mdp as m

    assert callable(m.lateral_push_reward)
    assert callable(m.pitch_swing_penalty)

    model = get_walk_rollers_spec().compile()
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
    ]
    # The regexes the helpers use must match exactly one left + one right joint.
    import re

    for pattern in (r"^(left|right)_hip_roll$", r"^(left|right)_hip_pitch$"):
        hits = [n for n in names if n and re.match(pattern, n)]
        assert len(hits) == 2, f"{pattern} matched {hits}"
