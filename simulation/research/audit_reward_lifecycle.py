"""Exercise real mjlab RewardManager reset hooks on the current v07 model."""
import argparse
import json
from pathlib import Path

import mjlab
import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks import mdp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    cfg = load_env_cfg('Mjlab-Velocity-Flat-MicroDinosaur', play=False)
    cfg.scene.num_envs = 8
    cfg.scene.entities['robot'].spec_fn = lambda: mujoco.MjSpec.from_file(args.xml)
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    env.reset()
    terms = {n: env.reward_manager.get_term_cfg(n).func
             for n in ('lean_drift', 'contact_timing', 'head_world_gaze')}
    for _ in range(12):
        env.step(torch.randn(8, 19, device=env.device)*.05)
    # Distinct histories expose accidentally clearing the whole vector batch.
    terms['lean_drift'].ema.copy_(torch.arange(8, device=env.device)*.01)
    terms['contact_timing'].ema[:, 0] = torch.arange(8, device=env.device)*.02
    terms['contact_timing'].ema[:, 1] = .75
    kept = torch.tensor([0, 2, 4, 6, 7], device=env.device)
    reset = torch.tensor([1, 3, 5], device=env.device)
    saved = {n: terms[n].ema.clone() for n in ('lean_drift', 'contact_timing')}
    home_before = terms['head_world_gaze'].home_quat.clone()
    env.reset(env_ids=reset)
    for name, neutral in [('lean_drift', 0.), ('contact_timing', .5)]:
        assert torch.all(terms[name].ema[reset] == neutral)
        torch.testing.assert_close(terms[name].ema[kept], saved[name][kept])
    torch.testing.assert_close(home_before, terms['head_world_gaze'].home_quat)
    reward_cfg = env.reward_manager.get_term_cfg('head_world_gaze')
    qpos_before = env.sim.data.qpos.clone()
    rebuilt = mdp.head_world_gaze_tracking(reward_cfg, env)
    torch.testing.assert_close(rebuilt.home_quat, home_before)
    torch.testing.assert_close(env.sim.data.qpos.clone(), qpos_before)
    report = {'status': 'PASS', 'num_envs': 8, 'reset_envs': reset.tolist(),
              'non_reset_histories_preserved': True, 'reset_matches_neutral_state': True,
              'head_home_independent_of_live_rollout': True, 'no_live_state_mutation': True,
              'home_quat_wxyz': home_before.tolist(), 'weights': {
                  n: env.reward_manager.get_term_cfg(n).weight for n in terms}}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == '__main__':
    main()
