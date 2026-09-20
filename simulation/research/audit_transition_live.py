"""Check calibrated sensor/physics correspondence and reset isolation on GPU."""
import json
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks import mdp
from transition_refine_cfg import build_config,OUT


def main():
    _,cfg=build_config(8)
    env=ManagerBasedRlEnv(cfg=cfg.env,device='cuda:0');env.reset()
    a=env.action_manager.get_term('joint_pos'); robot=env.scene['robot']
    assert a.ready.all() and a.packet_valid.all()
    torch.testing.assert_close(robot.data.joint_pos[:,a._target_ids],a.bank['joint_pos'][a.bank_index])
    torch.testing.assert_close(a.imu.r,a.bank['head_rotation'][a.bank_index])
    torch.testing.assert_close(a.imu.accel,a.bank['head_accel'][a.bank_index])
    torch.testing.assert_close(a.controller.last_target,a.bank['applied'][a.bank_index][:,a.head])
    torch.testing.assert_close(env.action_manager.action,a.bank['raw_action'][a.bank_index])
    initial_reward=mdp.navigation_head_attitude(env)
    assert initial_reward.min()>.97, initial_reward
    twist=env.command_manager.get_term('twist');twist.mode[:]=1;twist._command[:]=0
    r=a.imu.r.clone();acc=a.imu.accel.clone()
    env.step(a._raw_actions.clone())
    torch.testing.assert_close(a.imu.r,r,atol=0,rtol=0)
    torch.testing.assert_close(a.imu.accel,acc,atol=0,rtol=0)
    for _ in range(40):
        _,reward,_,_,_=env.step(torch.zeros(8,19,device=env.device))
        assert torch.isfinite(reward).all() and torch.isfinite(a._processed_actions).all()
    keep=torch.tensor([0,2,3,5,6,7],device=env.device)
    snapshot={name:getattr(a,name).clone() for name in ('packet','_applied','yaw_reference','reward_heading_zero','bank_index')}
    imu=a.imu.r.clone(); correction=a.controller.correction.clone(); clock=twist.elapsed.clone()
    env.reset(env_ids=torch.tensor([1,4],device=env.device))
    for name,old in snapshot.items(): torch.testing.assert_close(getattr(a,name)[keep],old[keep],atol=0,rtol=0)
    torch.testing.assert_close(a.imu.r[keep],imu[keep],atol=0,rtol=0)
    torch.testing.assert_close(a.controller.correction[keep],correction[keep],atol=0,rtol=0)
    torch.testing.assert_close(twist.elapsed[keep],clock[keep],atol=0,rtol=0)
    assert (a.motor_buffer._buffer.current_length[[1,4]]==13).all()
    env.close()
    record=dict(status='PASS',envs=8,steps=41,initial_head_reference_reward_min=float(initial_reward.min()),
        physical_joint_pose_matches_calibration=True,calibration_packet_not_integrated_twice=True,
        raw_action_and_head_filter_seeded=True,partial_reset_preserves_other_sensor_reference_and_phase_states=True,
        motor_history_backfilled_before_first_action=True)
    (OUT/'live_audit.json').write_text(json.dumps(record,indent=2));print(record)


if __name__=='__main__':main()
