"""GPU reset/ray/command audit, plus deterministic-policy contact exposure."""
import json
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks import mdp
from terrain_lane_cfg import build_config,OUT
from probe_head_ownership import CHECKPOINT


def main():
    _,cfg=build_config(16);env=ManagerBasedRlEnv(cfg=cfg.env,device='cuda:0');env.reset()
    a=env.action_manager.get_term('joint_pos');robot=env.scene['robot'];terrain=env.scene.terrain
    clearance=env.scene['body_clearance'].data.heights[:,0].clone()
    assert (clearance>.10).all() and (clearance<.125).all(),clearance
    assert set(terrain.terrain_types.cpu().tolist())=={0,1,2}
    assert mdp.navigation_head_attitude(env).min()>.97
    term=env.command_manager.get_term('twist')
    assert (term.mode[terrain.terrain_types>0]==0).all()
    original={name:getattr(a,name).clone() for name in ('packet','_applied','yaw_reference','reward_heading_zero','bank_index')}
    old_clock=term.elapsed.clone();ids=torch.tensor([1,9],device=env.device)
    env.reset(env_ids=ids);keep=torch.ones(16,dtype=torch.bool,device=env.device);keep[ids]=False
    for name,old in original.items():torch.testing.assert_close(getattr(a,name)[keep],old[keep],atol=0,rtol=0)
    torch.testing.assert_close(term.elapsed[keep],old_clock[keep],atol=0,rtol=0)
    # CPU loaded normalized actor implemented explicitly for this deterministic
    # diagnostic only; formal deployment exports remain the official path.
    state=torch.load(CHECKPOINT,map_location=env.device,weights_only=False)['actor_state_dict']
    seen=torch.zeros(16,dtype=torch.bool,device=env.device);resets=0
    term.mode[:]=0;term.request[:]=0;term.request[:,0]=.35
    for step in range(450):
        x=(env.obs_buf['actor']-state['obs_normalizer._mean'])/(state['obs_normalizer._std']+.01)
        for i in (0,2,4,6):
            x=torch.nn.functional.linear(x,state[f'mlp.{i}.weight'],state[f'mlp.{i}.bias'])
            if i<6:x=torch.nn.functional.elu(x)
        _,r,terminated,truncated,_=env.step(x)
        assert torch.isfinite(r).all() and torch.isfinite(a._processed_actions).all()
        relative=robot.data.root_link_pos_w-env.scene.env_origins
        seen |= (relative[:,0]>.4)&(terrain.terrain_types>0)
        resets+=int(terminated.sum())
    result=dict(status='PASS',environments=16,steps=450,terrain_counts=torch.bincount(terrain.terrain_types).tolist(),
        terrain_envs_crossed_first_step=int(seen.sum()),initial_ground_clearance_range_m=[float(clearance.min()),float(clearance.max())],
        partial_reset_isolated=True,diagnostic_terminations=resets)
    assert seen.any(),'No real step exposure in live probe'
    env.close();(OUT/'live_audit.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':main()
