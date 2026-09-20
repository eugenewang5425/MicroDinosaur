"""Microduck roller SITSTAND — commanded sit ↔ stand ON ROLLERS.

Port of the walker sitstand recipe (``microduck_sitstand_env_cfg``) to the
roller model, following the roller_standup pattern: derive from the roller
velocity env (robot, sensors, DR, 61D obs all inherited → runtime-swappable via
``--new-cmd-obs``) and swap only the reward recipe and the command.

The roller-specific work is the SIT KEYFRAME. The walker keyframe (flat feet,
knee ±1.35) is NOT statically stable on rollers — measured 100° tilt, 5%
survival under a firmware-strength PD hold: the blades+wheels cannot brace the
way a flat sole does. A symmetric leg-pose sweep
(``scripts/search_roller_sit.py``) found the stable seated family; the chosen
pose survives 20/20 noisy resets at median 2.9° tilt:

    hip_pitch ∓0.80, knee ±1.20, ankle 0, hip_roll ±0.20, trunk z 0.063
    (splayed blades = lateral support; ankle flat, no sole edge to pivot on)

Re-verify with ``scripts/measure_roller_sit.py`` / ``search_roller_sit.py``
before changing the pose or the robot model.

Posture command and rewards are the walker's, with roller heights:
    STAND = HOME + ROLLER_STAND_Z (0.138, measured 0.140 under PD hold)
    SIT   = keyframe above + ROLLER_SIT_Z (0.063)
The head/body command slots stay zero-padded (roller family convention); the
neck is held by the inherited neck_joint_pos_l2, resolved by NAME.

Joint indices are SERVO-space (14 layout) — the roller model interleaves the
four passive wheels, so never index the entity joint array directly
(AGENTS.md; locked by tests/test_roller_sitstand_cfg.py).
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# ── Trunk heights (m) — measured on scene_rollers.xml, never carried over ────
ROLLER_STAND_Z = 0.138   # standing on blades (0.1402 measured under PD hold)
ROLLER_SIT_Z = 0.063     # seated keyframe below (0.0629 measured, 20/20 stable)

# Episode length: room for 2-3 posture segments (dwell 3.5-6.5 s each).
EPISODE_LENGTH_S = 12.0
POSTURE_DWELL_S = (3.5, 6.5)
POSTURE_RAMP_S = 2.0     # STAND↔SIT blend duration (the anti-crash mechanism)
SIT_PROB = 0.5
NUM_STEPS_PER_ENV = 24

# ── Seated keyframe (SERVO joint index → angle, rad) ─────────────────────────
# Verified stable on the roller model (see module docstring). hip_roll is
# splayed ±0.20 for lateral support; ankle flat so no sole edge can pivot.
ROLLER_SITTING_TARGET_OVERRIDES = {
    1:   0.20,    # left  hip_roll  (splay)
    2:  -0.80,    # left  hip_pitch
    3:   1.20,    # left  knee
    4:   0.00,    # left  ankle
    10: -0.20,    # right hip_roll
    11:  0.80,    # right hip_pitch
    12: -1.20,    # right knee
    13:  0.00,    # right ankle
}

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# Upright gating window (roller heights): full upright incentive above
# STAND_UPRIGHT_Z, fades to 0 at SIT_UPRIGHT_Z.
STAND_UPRIGHT_Z = 0.120
SIT_UPRIGHT_Z = 0.080

# Motion caps (walker values — the gentleness calibration is robot-agnostic;
# roller friction makes fast drops EASIER, so keep them at least as tight).
MAX_DESCENT_SPEED = 0.05
MAX_RISE_SPEED = 0.08

# Skating rewards: meaningless (and counterproductive) for a sit-stand task.
_SKATING_REWARDS = (
    "wheel_speed", "braking", "skating_air_time", "glide", "single_support",
    "gait_symmetry", "forward_lean", "heading_hold", "feet_flat",
    "hip_roll_neutral",
    # Walk/stand-pose terms replaced by the posture-conditioned stack.
    "pose", "upright", "com_height_target",
)


def make_microduck_roller_sitstand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Roller sit-stand env: roller velocity base minus skating terms, plus the
    walker posture-conditioned sit-stand reward stack at roller heights."""

    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.viewer.body_name = "trunk_base"

    for name in _SKATING_REWARDS:
        cfg.rewards.pop(name, None)

    # ── Command: posture flag in the twist slot (walker recipe) ──────────────
    # The roller base installs a RelativeHeadingVelocityCommandCfg; replace it
    # with the sit-stand posture command. head_pose / body_pose slots stay
    # zero-padded (roller convention) → 61D obs parity preserved.
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = POSTURE_DWELL_S
    command.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.SitStandCommandCfg(
        **{
            **vars(command),
            "sit_prob": SIT_PROB,
            "ramp_s": POSTURE_RAMP_S,
            "sit_z": ROLLER_SIT_Z,
            "stand_z": ROLLER_STAND_Z,
        }
    )

    # ── Posture-conditioned task rewards (walker weights) ───────────────────
    cfg.rewards["posture_pose_legs"] = RewardTermCfg(
        func=microduck_mdp.posture_pose_match,
        weight=4.0,
        params={
            "command_name": "twist",
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "sit_overrides": ROLLER_SITTING_TARGET_OVERRIDES,
        },
    )
    cfg.rewards["posture_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.posture_pose_l1,
        weight=1.0,
        params={
            "command_name": "twist",
            "joint_indices": _LEG_JOINTS,
            "sit_overrides": ROLLER_SITTING_TARGET_OVERRIDES,
        },
    )
    cfg.rewards["posture_height"] = RewardTermCfg(
        func=microduck_mdp.posture_height_gaussian,
        weight=1.0,
        params={
            "command_name": "twist",
            "sit_z": ROLLER_SIT_Z,
            "stand_z": ROLLER_STAND_Z,
            "std": 0.04,
        },
    )
    cfg.rewards["posture_height_sharp"] = RewardTermCfg(
        func=microduck_mdp.posture_height_gaussian,
        weight=1.0,
        params={
            "command_name": "twist",
            "sit_z": ROLLER_SIT_Z,
            "stand_z": ROLLER_STAND_Z,
            "std": 0.015,
        },
    )
    cfg.rewards["posture_height_l1"] = RewardTermCfg(
        func=microduck_mdp.posture_height_l1,
        weight=6.0,
        params={
            "command_name": "twist",
            "sit_z": ROLLER_SIT_Z,
            "stand_z": ROLLER_STAND_Z,
        },
    )
    # Rise bootstrap: pays for upward motion itself while STAND is commanded
    # and the trunk is below the cutoff (just ABOVE the target so the final cm
    # still pays). Zero under a SIT command.
    cfg.rewards["rise_bootstrap"] = RewardTermCfg(
        func=microduck_mdp.posture_rise_bootstrap,
        weight=0.75,
        params={
            "command_name": "twist",
            "max_height": ROLLER_STAND_Z + 0.012,
            "max_vz": MAX_RISE_SPEED,
        },
    )
    # Gentleness: positive weights on already-negative functions (the double
    # negative bug trained crash-sitting policies in the walker env).
    cfg.rewards["descent_speed"] = RewardTermCfg(
        func=microduck_mdp.trunk_downward_velocity_penalty,
        weight=10.0,
        params={
            "max_down_vel": MAX_DESCENT_SPEED,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["rise_speed"] = RewardTermCfg(
        func=microduck_mdp.trunk_upward_velocity_penalty,
        weight=0.0,
        params={
            "max_up_vel": MAX_RISE_SPEED,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["gentle_motion"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=2.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["upright_while_tall"] = RewardTermCfg(
        func=microduck_mdp.upright_while_tall,
        weight=1.5,
        params={
            "height_low": SIT_UPRIGHT_Z,
            "height_high": STAND_UPRIGHT_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["posture_stillness"] = RewardTermCfg(
        func=microduck_mdp.posture_stillness,
        weight=2.0,
        params={
            "command_name": "twist",
            "sit_z": ROLLER_SIT_Z,
            "stand_z": ROLLER_STAND_Z,
            "band_full": 0.012,
            "band_zero": 0.03,
            "vel_std": 0.05,
            "tilt_full_deg": 25.0,
            "tilt_zero_deg": 60.0,
        },
    )
    # head_std is OMITTED: the roller family has no head_pose command (head/body
    # slots stay zero-padded), and the walker's dangling-head exploit is already
    # covered here by the inherited neck_joint_pos_l2 (-0.5) pulling the neck to
    # HOME. Passing head_std would raise KeyError on the missing command.
    cfg.rewards["posture_composite"] = RewardTermCfg(
        func=microduck_mdp.posture_composite,
        weight=3.0,
        params={
            "command_name": "twist",
            "sit_overrides": ROLLER_SITTING_TARGET_OVERRIDES,
            "joint_indices": _LEG_JOINTS,
            "sit_z": ROLLER_SIT_Z,
            "stand_z": ROLLER_STAND_Z,
            "height_std": 0.03,
            "upright_std": 0.40,
            "pose_std": 0.40,
        },
    )

    # ── Sim2real regularisers — velocity parity ──────────────────────────────
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    # ── Curricula ────────────────────────────────────────────────────────────
    # action_rate ramp (walker schedule), rise-speed cap introduced late (the
    # standup attempt-tax lesson: a motion tax during skill discovery makes
    # exploratory attempts net-negative and the skill is never found).
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.2},
                {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.4},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.6},
                {"step": 1250 * NUM_STEPS_PER_ENV, "weight": -0.8},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -1.0},
            ],
        },
    )
    cfg.curriculum["rise_speed_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "rise_speed",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": 5.0},
                {"step": 2500 * NUM_STEPS_PER_ENV, "weight": 10.0},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.1},
            ],
        },
    )

    return cfg


MicroduckRollerSitStandRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="roller_sitstand",
    run_name="roller_sitstand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=15_000,
)
