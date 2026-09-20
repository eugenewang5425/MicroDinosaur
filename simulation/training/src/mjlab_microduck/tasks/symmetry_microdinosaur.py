"""Bilateral (left-right) symmetry augmentation for the MicroDinosaur 81-D / 19-joint envs.

Rewritten for MicroDinosaur (2026-09-12) because the microduck version in symmetry.py is
hard-coded to the 61-D / 14-joint layout.  Enabling this gives PPO a mirror loss,
which is the principled fix for left-right gait asymmetry at large stride
(the observed failure: right foot lifts 5.6 mm vs left 3.5 mm at action scale 1.6).

Actor observation layout (81-dim flat tensor, term insertion order):
    [0:3]   base_ang_vel      (roll, pitch, yaw            -- body frame)
    [3:6]   projected_gravity (gx, gy, gz                  -- body frame)
    [6:25]  joint_pos_rel     (19 joints, relative to default)
    [25:44] joint_vel_rel     (19 joints)
    [44:63] last_action       (19 joints)
    [63:66] twist command     (lin_vel_x, lin_vel_y, ang_vel_z)
    [66:70] head command      (neck_pitch, head_pitch, head_yaw, head_roll)
    [70:76] body command      (x, y, z, roll, pitch, yaw)
    [76:78] arm command       (arm_l, arm_r)
    [78:80] tail command      (tail_yaw, tail_pitch)
    [80]    jaw command       (jaw_hinge)

Joint order (mjlab actuator order == XML body traversal order):
    0  left_hip_yaw     5  neck_pitch      10 right_hip_yaw    15 tail_yaw
    1  left_hip_roll    6  head_pitch      11 right_hip_roll   16 tail_pitch
    2  left_hip_pitch   7  head_yaw        12 right_hip_pitch  17 arm_l
    3  left_knee        8  head_roll       13 right_knee       18 arm_r
    4  left_ankle       9  jaw_hinge       14 right_ankle

Mirroring rules (reflection about the sagittal plane, y -> -y):
  * left leg (0-4) <-> right leg (10-14); midline joints (5-9) keep their slot
  * arms swap (17 <-> 18), tail_yaw/tail_pitch keep their slots
  * sign flips: every left/right pair joint (the home frames carry mirrored
    sign conventions), head_yaw, head_roll, tail_yaw, both arm joints
    (a rotation about +Y becomes -Y under the mirror)
  * base_ang_vel: flip roll[0], yaw[2];  projected_gravity: flip gy[1 of block]
  * twist: flip lin_vel_y, ang_vel_z;  head cmd: flip yaw, roll
  * body cmd: flip y, roll, yaw; arm cmd: swap left/right and flip both;
    tail cmd: flip tail_yaw
"""

from dataclasses import dataclass

import torch
from tensordict import TensorDict
from mjlab.rl import RslRlPpoAlgorithmCfg


@dataclass
class PpoWithSymmetryCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm config extended with an optional symmetry_cfg field."""

    symmetry_cfg: dict | None = None


# Within a 19-joint block: left leg (0-4) <-> right leg (10-14), midline (5-9)
# unchanged, tail (15,16) unchanged, arms (17,18) swapped.
_JOINT_PERM = [10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 15, 16, 18, 17]
# Signs applied AFTER permutation.
_JOINT_SIGN = [-1, -1, -1, -1, -1, 1, 1, -1, -1, 1,
               -1, -1, -1, -1, -1, -1, 1, -1, -1]

# Obs index map: (index, sign) pairs are built from the block layout above.
_OBS_PERM: list[int] = (
    [0, 1, 2]                                   # base_ang_vel
    + [3, 4, 5]                                 # projected_gravity
    + [6 + j for j in _JOINT_PERM]              # joint_pos
    + [25 + j for j in _JOINT_PERM]             # joint_vel
    + [44 + j for j in _JOINT_PERM]             # last_action
    + [63, 64, 65]                              # twist command
    + [66, 67, 68, 69]                          # head command
    + [70, 71, 72, 73, 74, 75]                  # body command
    + [77, 76]                                  # exchange left/right arm targets
    + [78, 79]                                  # tail command
    + [80]                                      # jaw command
)
_OBS_SIGN: list[float] = (
    [-1, 1, -1]                                 # ang vel: flip roll, yaw
    + [1, -1, 1]                                # gravity: flip gy
    + _JOINT_SIGN + _JOINT_SIGN + _JOINT_SIGN
    + [1, -1, -1]                               # twist: flip vy, wz
    + [1, 1, -1, -1]                            # head: flip yaw, roll
    + [1, -1, 1, -1, 1, -1]                     # body: flip y, roll, yaw
    + [-1, -1]                                  # arm: flip both
    + [-1, 1]                                   # tail: flip tail_yaw only
    + [1]                                       # jaw
)


def _get_tensors(device: torch.device):
    perm = torch.tensor(_OBS_PERM, dtype=torch.long, device=device)
    sign = torch.tensor(_OBS_SIGN, dtype=torch.float32, device=device)
    act_perm = torch.tensor(_JOINT_PERM, dtype=torch.long, device=device)
    act_sign = torch.tensor(_JOINT_SIGN, dtype=torch.float32, device=device)
    return perm, sign, act_perm, act_sign


def microdinosaur_vel_symmetry(
    env,
    obs: TensorDict | None,
    actions: torch.Tensor | None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Bilateral symmetry augmentation for the MicroDinosaur 81-D / 19-joint env.

    Contract (rsl_rl PPO ``symmetry_cfg`` interface): either argument may be None
    -- the mirror loss path calls this with observations only, so the device is
    taken from whichever input is present, and a non-None input comes back
    doubled along the batch axis as ``[original; mirrored]``.

    Reflect positions by y -> -y and quaternions by (w,x,y,z) -> (w,-x,y,-z).
    Limb commands must undergo the same left/right exchange as their joint
    states. The asymmetric-arm regression checks tracking-reward invariance;
    zero or equal arm commands alone cannot expose a missing exchange.
    """
    aug_obs: TensorDict | None = None
    aug_actions: torch.Tensor | None = None

    if obs is not None:
        actor_orig: torch.Tensor = obs["actor"]
        perm, sign, _, _ = _get_tensors(actor_orig.device)
        actor_sym = actor_orig[:, perm] * sign
        critic_orig: torch.Tensor = obs["critic"]
        # Critic mirroring is not implemented (not needed for the mirror loss);
        # repeating the unmirrored critic obs is harmless because the critic sees
        # privileged information the actor does not.
        critic_repeated = torch.cat([critic_orig, critic_orig], dim=0)
        aug_obs = TensorDict(
            {
                "actor": torch.cat([actor_orig, actor_sym], dim=0),
                "critic": critic_repeated,
            },
            batch_size=[actor_orig.shape[0] * 2],
            device=actor_orig.device,
        )

    if actions is not None:
        _, _, act_perm, act_sign = _get_tensors(actions.device)
        actions_sym = actions[:, act_perm] * act_sign
        aug_actions = torch.cat([actions, actions_sym], dim=0)

    return aug_obs, aug_actions


# loss coefficient 0.3 -> 0.01 on 2026-09-12: measured with the loss probe
# (MICRODINO_PROBE=lossprint) the mirror MSE on the warm-start policy is 0.064
# while the PPO surrogate loss is only 0.0028, so 0.3 * 0.064 = 0.019 was 6.7x
# the policy-gradient term -- the update was optimizing symmetry instead of
# reward (reward fell 129 -> 62 while mean_action_acc rose 0.40 -> 0.58).
# 0.01 keeps the mirror term ~0.2x the surrogate: a regulariser, not a driver.
# Re-measure the balance whenever the observation layout or the reward changes.
SYMMETRY_CFG = {
    "use_data_augmentation": False,
    "use_mirror_loss": True,
    "mirror_loss_coeff": 0.01,
    "data_augmentation_func": "mjlab_microduck.tasks.symmetry_microdinosaur.microdinosaur_vel_symmetry",
}
