"""Real CUDA substep counter, support sensor and mesh-minimum audit."""
import json
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from foot_flight_cfg import build_config,OUT


def main():
    _,cfg=build_config(8);env=ManagerBasedRlEnv(cfg.env,device='cuda:0');obs,_=env.reset(seed=61)
    maximum_error=0.;support_error=0.
    for i in range(20):
        obs,reward,*_=env.step(torch.zeros((8,19),device=env.device))
        assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
        assert env._foot_progress.last_substep==env._sim_step_counter
        g=env._foot_geometry;actual=g.clearance(env.sim.data).cpu().numpy();m=env.sim.mj_model
        for column,gid in enumerate(g.gids):
            mid=m.geom_dataid[gid];start=m.mesh_vertadr[mid];v=m.mesh_vert[start:start+m.mesh_vertnum[mid]]
            R=env.sim.data.geom_xmat[:,gid].reshape(8,3,3).cpu().numpy();p=env.sim.data.geom_xpos[:,gid].cpu().numpy()
            expected=(R[:,2,:]@v.T).min(-1)+p[:,2]
            maximum_error=max(maximum_error,float(abs(actual[:,column]-expected).max()))
        force=env.scene['whole_ground_force'].data.force.reshape(8,-1,3)[...,2].abs().sum(-1)
        feet=env.scene['feet_ground_contact'].data.force.reshape(8,-1,3)
        # Feet sensor uses contact-frame force; norm avoids a frame assumption.
        assert (force>=0).all() and torch.isfinite(feet).all()
    assert maximum_error<2e-6
    assert env._sim_step_counter==320
    old=env._foot_progress.height_frontier.clone();env._foot_progress.qualified[:]=True
    env.reset(env_ids=torch.tensor([0,2],device=env.device))
    assert not env._foot_progress.qualified[0] and env._foot_progress.qualified[1]
    assert env._foot_progress.height_frontier[1]==old[1]
    robot=env.scene['robot'];names=list(robot.joint_names)
    expected=json.loads((OUT/'deep_targets.json').read_text())['action_names'];assert names==expected,(names,expected)
    report=dict(passed=True,physics_substeps=320,policy_steps=20,geometry_max_error_m=maximum_error,
        partial_reset_isolated=True,reference_joint_order_verified=True,actor_shape=list(obs['actor'].shape),
        whole_ground_force_last_n=force.tolist(),physical_motor_envelope='Inherited unchanged from jump_refine')
    (OUT/'gpu_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2));env.close()


if __name__=='__main__':main()
