"""Live CUDA integration check: torque limits, COM, reset isolation and finite steps."""
import json
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from jump_refine_cfg import build_config,OUT,whole_com,torque_bounds_torch


def main():
    _,cfg=build_config(8,61)
    env=ManagerBasedRlEnv(cfg.env,device='cuda:0');env.reset(seed=61)
    action=env.action_manager.get_term('joint_pos')
    assert action.cap.shape==(8,19)
    assert torch.all((action.cap>=.48-1e-6)&(action.cap<=.6+1e-6))
    env.sim.data.qvel.zero_();env.sim.data.qvel[:,:3]=torch.tensor([.1,-.2,.3],device=env.device)
    env.sim.forward();_,v=whole_com(env)
    torch.testing.assert_close(v,torch.tensor([.1,-.2,.3],device=env.device).expand(8,3),atol=2e-5,rtol=0)
    env.sim.data.qvel[:,action._entity.indexing.joint_v_adr]=torch.linspace(-25,25,19,device=env.device)
    env.sim.forward()
    velocity=action._entity.data.joint_vel[:,action._target_ids].clone()
    lo,hi=torque_bounds_torch(velocity,action.voltage,action.cap)
    action.apply_actions()
    actual=env.sim.model.actuator_forcerange[:,action.ctrl_ids]
    torch.testing.assert_close(actual,torch.stack([lo,hi],-1))
    env.scene.write_data_to_sim();env.sim.step()
    tau=env.sim.data.actuator_force[:,action.ctrl_ids]
    violation=torch.maximum(lo-tau,tau-hi).clamp(min=0).max().item()
    assert violation<1e-5,violation
    env.reset(seed=62)
    state=env._jump_refine_state;state.flight_max[:]=.10
    cap=action.cap.clone();voltage=action.voltage.clone()
    env.reset(env_ids=torch.tensor([0,2],device=env.device))
    torch.testing.assert_close(action.cap[1],cap[1]);torch.testing.assert_close(action.voltage[1],voltage[1])
    assert state.flight_max[0]==0 and state.flight_max[2]==0 and abs(state.flight_max[1].item()-.1)<1e-6
    assert torch.all((action.cap>=.48-1e-6)&(action.cap<=.6+1e-6))
    for _ in range(110):
        obs,reward,terminated,truncated,extras=env.step(torch.zeros((8,19),device=env.device))
        assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
    report=dict(passed=True,live_gpu_envs=8,finite_steps=110,actor_dimensions=list(obs['actor'].shape),
        actual_motor_torque_max_envelope_violation_nm=violation,whole_body_translation_test=True,
        caps_after_reset_min_nm=float(action.cap.min()),caps_after_reset_max_nm=float(action.cap.max()),
        voltage_range_v=[float(action.voltage.min()),float(action.voltage.max())],
        partial_reset_isolated=True,physics_dt=env.physics_dt,policy_dt=env.step_dt)
    (OUT/'gpu_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));env.close()


if __name__=='__main__':main()
