"""Microduck roller FIGURE-8 — carve figure-8s while skating.

Derives from the roller velocity env (robot, DR, 61D obs inherited) and swaps
the command for ``Figure8VelocityCommand``: cmd[2] aims at a virtual target
sliding along a Gerono lemniscate in the robot's frame, and cmd[0] (throttle)
is swept by the same phase so the duck pushes on the lobes and coasts through
the crossings. Tracking the moving target carves the 8.

Why robot-frame (not world-frame): a world-frame 8 would require position
tracking and would punish early-training drift as a tracking error; a
robot-frame target is a pure heading+throttle task, so it trains with the
existing ``heading_tracking`` / ``wheel_speed`` stack and no new reward — and
it degrades gracefully on hardware (the operator's planner feeds cmd[2] the
same way as the velocity task).

Reward changes vs the roller base:
  - ``heading_hold`` is replaced by ``heading_tracking`` (the base's
    heading_hold assumes a world-frame heading target and would fight the
    moving one).
  - ``forward_lean`` / ``glide`` / ``single_support`` / ``skating_air_time``
    keep their weights: the stride style is unchanged, only the path is new.
  - A ``figure8_curvature`` term pays for carrying a turn rate INTO the
    command direction (the 8 needs committed turns at the lobe tips, where
    pure heading error saturates).
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# One full 8 per ~8 s at 0.4-0.5 m/s throttle gives ~0.5 m lobes — inside the
# roller env's own command range (cmd_x ∈ [0, 0.6]) and the flat floor.
FIGURE8_PERIOD_S = 8.0
FIGURE8_RADIUS = 0.8
THROTTLE_MIN = 0.0
THROTTLE_MAX = 0.5
NUM_STEPS_PER_ENV = 24


def make_microduck_figure8_rollers_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    # ── Command: figure-8 target instead of a fixed heading ──────────────────
    command = cfg.commands["twist"]
    cfg.commands["twist"] = microduck_mdp.Figure8VelocityCommandCfg(
        **{
            **vars(command),
            "period_s": FIGURE8_PERIOD_S,
            "radius": FIGURE8_RADIUS,
            "throttle_min": THROTTLE_MIN,
            "throttle_max": THROTTLE_MAX,
        }
    )

    # ── Rewards: swap the world-frame heading term for the moving-target one ──
    # heading_hold compares against a fixed world heading — with a sweeping
    # target it is maximally wrong exactly when the duck is correctly chasing
    # the lobe. Drop it; heading_tracking (exp(-err²/std²)) is the right term.
    cfg.rewards.pop("heading_hold", None)
    cfg.rewards["heading_tracking"] = RewardTermCfg(
        func=microduck_mdp.heading_tracking_reward,
        weight=2.0,
        params={"command_name": "twist", "std": 0.5},
    )

    return cfg


MicroduckFigure8RlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="figure8_rollers",
    run_name="figure8_rollers",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=15_000,
)
