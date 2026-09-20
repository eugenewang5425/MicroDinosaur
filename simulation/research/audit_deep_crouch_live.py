"""Live GPU action-boundary check: delayed packets, raw history, partial reset."""
import json
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from deep_crouch_cfg import build_config


def main():
    _, cfg = build_config(8)
    env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0')
    env.reset()
    term = env.action_manager.get_term('joint_pos')
    seen = {}
    original = term.controller.update
    def update(nominal, angles, orientation, gyro, desired, omega, dt, age):
        seen.update(gyro=gyro.clone(), angles=angles.clone(), age=age.clone())
        return original(nominal, angles, orientation, gyro, desired, omega, dt, age)
    term.controller.update = update
    max_packet_error = 0.
    for i in range(120):
        packet = term.packet.clone()
        measured = (env.obs_buf['actor'][:, 6:25]+term._offset)[:, term.chain].clone()
        actions = torch.zeros(8, 19, device=env.device)
        actions[:, 6] = .05*torch.sin(torch.tensor(i*.1, device=env.device))
        _, reward, _, _, _ = env.step(actions)
        torch.testing.assert_close(seen['gyro'], packet[:, :3], atol=0, rtol=0)
        torch.testing.assert_close(seen['angles'], measured, atol=0, rtol=0)
        torch.testing.assert_close(term._raw_actions, actions, atol=0, rtol=0)
        assert torch.isfinite(term._processed_actions).all() and torch.isfinite(reward).all()
        max_packet_error = max(max_packet_error, float((seen['gyro']-packet[:, :3]).abs().max()))
    assert term.ready.all(), 'Accelerometer initialization did not finish'
    assert torch.all(seen['age'] == .02)
    before = {n: getattr(term, n).clone() for n in ('packet', '_applied', 'yaw_reference', 'ready')}
    correction = term.controller.correction.clone()
    imu = term.imu.r.clone()
    env.reset(env_ids=torch.tensor([1, 4], device=env.device))
    keep = torch.tensor([0, 2, 3, 5, 6, 7], device=env.device)
    for name, old in before.items(): torch.testing.assert_close(getattr(term, name)[keep], old[keep], atol=0, rtol=0)
    torch.testing.assert_close(term.controller.correction[keep], correction[keep], atol=0, rtol=0)
    torch.testing.assert_close(term.imu.r[keep], imu[keep], atol=0, rtol=0)
    assert not term.ready[[1, 4]].any()
    assert not term.controller.correction[[1, 4]].any()
    env.close()
    record = dict(status='PASS', envs=8, control_steps=120, packet_delay_ms=20,
        maximum_packet_error=max_packet_error, encoder_source='same delayed actor observation',
        raw_actor_history_unchanged=True, selected_reset_preserves_other_histories=True,
        caveat='Reset tilt uses first valid accelerometer packet; no stationary warmup in GPU training')
    (Path(__file__).parent/'20260914_deep_crouch/live_audit.json').write_text(json.dumps(record, indent=2))
    print(record)


if __name__ == '__main__': main()
