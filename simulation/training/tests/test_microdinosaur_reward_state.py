"""Reward lifecycle regressions: partial resets and a deterministic head target."""
from types import SimpleNamespace

import mjlab  # register plugins before importing task MDP helpers
import mujoco
import numpy as np
import unittest
import torch

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_microduck.tasks import mdp


def fake_env():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="trunk"><freejoint/><geom size=".1" mass="1"/>
        <body name="neck" pos="0 0 .2"><joint axis="0 1 0"/><geom size=".03"/>
          <body name="head" pos=".1 0 0" quat=".70710678 0 -.70710678 0">
            <joint axis="0 1 0"/><geom size=".03"/>
          </body>
        </body>
      </body></worldbody></mujoco>''')
    state = SimpleNamespace(default_root_state=torch.tensor([[0., 0., 0., 1., 0., 0., 0.]]).repeat(3, 1),
                            default_joint_pos=torch.tensor([[.35, .35]]).repeat(3, 1))
    asset = SimpleNamespace(data=state, is_fixed_base=False, indexing=SimpleNamespace(
        free_joint_q_adr=torch.arange(7), joint_q_adr=torch.tensor([7, 8]),
        body_ids=torch.tensor([1, 2, 3]), root_body_id=1))
    contact = SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[1., 0.], [0., 1.], [1., 0.]])))
    commands = {'twist': torch.ones(3, 3), 'head_pose': torch.zeros(3, 4)}
    env = SimpleNamespace(num_envs=3, device='cpu', sim=SimpleNamespace(mj_model=model),
                          scene={'robot': asset, 'feet': contact},
                          command_manager=SimpleNamespace(get_command=lambda name: commands[name]))
    set_pose(env)
    return env


def set_pose(env, yaw=0., neck_offset=0.):
    model = env.sim.mj_model
    d = mujoco.MjData(model)
    d.qpos[:7] = env.scene['robot'].data.default_root_state[0].numpy()
    d.qpos[3:7] = [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
    d.qpos[7:] = [.35 + neck_offset, .35]
    mujoco.mj_kinematics(model, d)
    state = env.scene['robot'].data
    state.root_link_quat_w = torch.tensor(d.xquat[1].copy(), dtype=torch.float)[None].repeat(3, 1)
    state.body_link_quat_w = torch.tensor(d.xquat[1:].copy(), dtype=torch.float)[None].repeat(3, 1, 1)


def check_partial_reset_matches_fresh_episode_and_preserves_other_environments(name):
    env = fake_env()
    if name == 'lean':
        env.scene['robot'].data.root_link_quat_w[:] = torch.tensor([.98, .19899749, 0., 0.])
        cfg = RewardTermCfg(func=mdp.lean_drift_penalty, weight=-4.)
        kwargs = {}
        neutral = 0.
    else:
        cfg = RewardTermCfg(func=mdp.contact_timing_asymmetry, weight=-4., params={'sensor_name': 'feet'})
        kwargs = cfg.params
        neutral = .5
    term = cfg.func(cfg, env)
    fresh = cfg.func(cfg, env)
    for _ in range(100):
        term(env, **kwargs)
    history = term.ema.clone()
    term.reset(torch.tensor([1]))
    torch.testing.assert_close(term.ema[[0, 2]], history[[0, 2]])
    assert torch.all(term.ema[1] == neutral)
    result = term(env, **kwargs)
    expected = fresh(env, **kwargs)
    torch.testing.assert_close(result[1], expected[1])
    term.reset(None)
    assert torch.all(term.ema == neutral)


def check_head_home_is_independent_of_first_live_pose_and_spawn_heading():
    env = fake_env()
    selector = SceneEntityCfg('robot', body_ids=[2])
    cfg = RewardTermCfg(func=mdp.head_world_gaze_tracking, weight=1., params={'asset_cfg': selector})
    reference = cfg.func(cfg, env).home_quat.clone()
    set_pose(env, yaw=1.2, neck_offset=.8)
    live_before = env.scene['robot'].data.body_link_quat_w.clone()
    term = cfg.func(cfg, env)
    torch.testing.assert_close(term.home_quat, reference)
    torch.testing.assert_close(env.scene['robot'].data.body_link_quat_w, live_before)
    assert torch.all(term(env, asset_cfg=selector) < .1)
    for yaw in (0., 1.2, -2.4):
        set_pose(env, yaw=yaw)
        torch.testing.assert_close(term(env, asset_cfg=selector), torch.ones(3), atol=1e-5, rtol=0.)
    term.reset(torch.tensor([1]))
    torch.testing.assert_close(term.home_quat, reference)


class RewardStateTests(unittest.TestCase):
    def test_lean_partial_reset(self):
        check_partial_reset_matches_fresh_episode_and_preserves_other_environments('lean')

    def test_contact_partial_reset(self):
        check_partial_reset_matches_fresh_episode_and_preserves_other_environments('contact')

    def test_head_home(self):
        check_head_home_is_independent_of_first_live_pose_and_spawn_heading()
