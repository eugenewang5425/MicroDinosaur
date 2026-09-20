"""Action low-pass + slew-rate limiting, applied INSIDE the training loop.

Why this exists (2026-09-13, user finding + our own measurements):

The user reported that jitter and in-place rotation grow with the action
amplitude, grow with servo stiffness kp, and grow when damping kd falls -- while
solver iterations make no difference ("求解器迭代没有影响").

Two quantitative checks pinned the cause down:

1. The joint servo loops are NOT ringing.  With the P2 nominal (kp=7,
   kd=0.4) and the leg's reflected inertia (J_eff ~ 1.8-2.2e-3 kg.m^2 from the
   mass-matrix diagonal), every one of the 19 joints is OVERDAMPED:
   zeta = kd / (2*sqrt(kp*J)) = 1.19 .. 1.78, natural frequency 6.6 .. 9.9 Hz.
   A ringing servo would sit far below zeta = 1.
2. Sweeping kd with a FIXED policy (only the servo changes): 0.4 -> 0.8 cut the
   standing yaw drift 42% (13.7 -> 7.9 deg / 5 s), the horizontal creep 68%
   (47 -> 15 mm/s) and the action jerk 43% (0.073 -> 0.042); 1.6 over-damped it
   again (drift 19.3 deg with a 12 deg spread) because the joints lag the
   commands.

So the oscillation is the POLICY's own high-frequency output: stiffness turns it
into torque, low kd fails to damp its velocity component, and the solver is
irrelevant.  Our training logs agree -- the runs whose `mean_action_acc` drifts
up to 0.58 are the runs that jitter and rotate, against 0.39 for the calm ones.

The literature-backed fix is to smooth the action stream, not the solver:

* low-pass after the policy output -- a Butterworth LPF with a 4 Hz cutoff in a
  biped locomotion stack, whose ablation shows jittering motion and worse
  convergence without it;
* a per-step change limit (slew rate), as in deployed legged stacks (EMA
  alpha ~ 0.25 plus a per-joint max delta);
* both sources warn the filter only removes high-frequency POLICY output -- the
  closed loop can still oscillate on large joint velocities, so the action-rate
  reward must stay as well (we already penalise it heavily at -3.5).

Two implementation notes:
* The filter lives in the action term, so it applies during TRAINING (a filter
  added only at deployment would invalidate the policy's learned authority and
  break the MDP-to-plant match).
* `last_action` in the observation stays the policy's own raw output, matching
  what deployed stacks feed back; only the applied target is filtered.
"""

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass(kw_only=True)
class FilteredJointPositionActionCfg(JointPositionActionCfg):
    """Joint position action with an EMA low-pass and/or a slew-rate limit.

    lp_alpha: NEW-target coefficient in (0, 1], applied once per control step:
        y[n] = (1-alpha)*y[n-1] + alpha*x[n].  0 disables the filter.
        At 50 Hz, alpha=0.9 attenuates even Nyquist by only 1.74 dB, so it
        has no -3 dB cutoff below 25 Hz. Do not use the small-alpha cutoff
        approximation for this setting. Alpha=0.5 has a 5.75 Hz cutoff.
    max_delta: maximum change of the applied target per control step, in HARD
        units (rad for these joints).  None disables.  0.08 rad/step = 4 rad/s
        at 50 Hz.
    """

    lp_alpha: float = 0.0
    max_delta: float | None = None

    def build(self, env: ManagerBasedRlEnv) -> "FilteredJointPositionAction":
        return FilteredJointPositionAction(self, env)


class FilteredJointPositionAction(JointPositionAction):
    def __init__(self, cfg: FilteredJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._lp_alpha = float(cfg.lp_alpha)
        self._max_delta = cfg.max_delta
        self._applied = self._processed_actions.clone()  # starts at the offset (default pose)

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        target = self._processed_actions
        smoothed = self._applied
        if self._lp_alpha > 0.0:
            smoothed = smoothed + self._lp_alpha * (target - smoothed)
        else:
            smoothed = target
        if self._max_delta is not None:
            delta = (smoothed - self._applied).clamp(-self._max_delta, self._max_delta)
            smoothed = self._applied + delta
        self._applied = smoothed
        self._processed_actions = smoothed

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        # Drop the filter state to the default pose for the reset envs, so a
        # fresh episode does not inherit the previous episode's filter memory.
        # NOTE: _offset is a per-env (num_envs, action_dim) tensor when
        # use_default_offset is set, so it must be indexed by env_ids -- a first
        # version assigned the whole batch to the selected rows and blew up with
        # "shape mismatch: [4096, 19] cannot be broadcast to [2, 19]".
        if torch.is_tensor(self._offset):
            if env_ids is None:
                self._applied = self._offset.clone()
            else:
                self._applied[env_ids] = self._offset[env_ids]
        else:
            if env_ids is None:
                self._applied.fill_(float(self._offset))
            else:
                self._applied[env_ids] = float(self._offset)
