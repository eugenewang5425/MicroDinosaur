"""Microduck roller STRIDE — the real skating push (alternating lateral drive).

Why a separate task from ``Mjlab-Velocity-Flat-MicroDuck-Rollers``: the base
roller recipe rewards the TIMING of a stride (single_support / glide /
skating_air_time) but is direction-agnostic, and it actively pulls hip_roll
toward neutral (-2.0). Measured on the base task's trained checkpoint
(gait_skate.csv, 600 steps at cmd_x=0.5):

    - L/R hip_pitch correlated +0.51  → the legs swing in PHASE, not alternating
    - hip_roll travel 0.94 rad (L) vs 0.52 rad (R) → lopsided
    - the left leg does the stepping while the right stays extended

That is a walk-like / one-legged-hobble gait, not skating. Real skating drives
the blade OUT to the side (abduction) and alternates left/right.

Changes vs the roller base:
  - ``hip_roll_neutral`` -2.0 → -0.3: the centring pull fights the stroke.
    Kept non-zero so the stance still cannot park splayed on the joint stops.
  - ``gait_symmetry`` -1.0 → -3.0: the measured lopsidedness says -1.0 loses
    to the stride rewards.
  - ``lateral_push`` (NEW, +2.0): pays the swing leg's abduction.
  - ``pitch_swing`` (NEW, +0.5 on a self-negating penalty): lightly discourages
    the big front-to-back pitch arc of a step (free band 0.35 rad, so a normal
    hip extension in the stroke is untouched).

Everything else (robot, DR, 61D obs, command, export path) is the roller base
unchanged, so the policy deploys with the same ``--roller --new-cmd-obs`` path.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# Abduction below this is natural splay, not a stroke (rad).
MIN_ABDUCTION = 0.08
# Hip-pitch travel that is free while swinging (a stroke DOES extend the hip).
PITCH_FREE_BAND = 0.35


def make_microduck_stride_rollers_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    # Let the stroke happen: a weak centring pull still closes a parked splay,
    # but no longer out-muscles the abduction the stroke needs.
    cfg.rewards["hip_roll_neutral"].weight = -0.3
    # The measured L/R asymmetry means the base weight was losing to the stride
    # rewards; alternating is the whole point of this task.
    cfg.rewards["gait_symmetry"].weight = -3.0

    cfg.rewards["lateral_push"] = RewardTermCfg(
        func=microduck_mdp.lateral_push_reward,
        weight=2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "vel_gate_ref": 0.2,
            "min_abduction": MIN_ABDUCTION,
        },
    )
    # Self-negating penalty → POSITIVE weight (AGENTS.md sign convention).
    cfg.rewards["pitch_swing"] = RewardTermCfg(
        func=microduck_mdp.pitch_swing_penalty,
        weight=0.5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "vel_gate_ref": 0.2,
            "free_band": PITCH_FREE_BAND,
        },
    )
    return cfg


MicroduckStrideRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="stride_rollers",
    run_name="stride_rollers",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=15_000,
)
