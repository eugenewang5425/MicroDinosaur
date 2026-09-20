"""Diagnostic, not CPU acceptance: deterministic policy in randomized GPU training worlds."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from jump_refine_cfg import build_config,OUT,whole_com
from run_jump_cfg import contacts


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--label',required=True)
    args=p.parse_args();_,cfg=build_config(8,71)
    env=ManagerBasedRlEnv(cfg.env,device='cuda:0');obs,_=env.reset(seed=71)
    state=torch.load(args.checkpoint,map_location='cuda:0',weights_only=False)['actor_state_dict']
    m=env.sim.mj_model; gids=[m.geom('robot/'+side+'_foot_collision').id for side in ['left','right']]
    vertices=[]
    for gid in gids:
        mesh=m.geom_dataid[gid];start=m.mesh_vertadr[mesh];count=m.mesh_vertnum[mesh]
        vertices.append(torch.tensor(m.mesh_vert[start:start+count].copy(),device=env.device))
    rows=[]
    with torch.inference_mode():
        for k in range(300):
            x=(obs['actor']-state['obs_normalizer._mean'])/(state['obs_normalizer._std']+.01)
            for i in [0,2,4,6]:
                x=torch.nn.functional.linear(x,state[f'mlp.{i}.weight'],state[f'mlp.{i}.bias'])
                if i<6:x=torch.nn.functional.elu(x)
            obs,reward,terminated,truncated,_=env.step(x)
            pos,vel=whole_com(env)
            heights=[]
            for gid,v in zip(gids,vertices):
                rotation=env.sim.data.geom_xmat[:,gid].reshape(8,3,3)
                heights.append((rotation[:,2,:]@v.T).amin(-1)+env.sim.data.geom_xpos[:,gid,2])
            gap=torch.stack(heights,-1).amin(-1)
            rows.append(torch.stack([env.command_manager.get_term('body_pose').elapsed,pos[:,2],vel[:,2],
                ~contacts(env,'robot_ground').any(-1),gap,env._jump_refine_state.flight_max,
                env._jump_refine_state.velocity_max,terminated|truncated,reward],-1).cpu().numpy())
    a=np.stack(rows);assert np.isfinite(a).all();env.close()
    longest=[]
    for j in range(8):
        counts=[]
        for mask in [a[:,j,3]>.5,(a[:,j,3]>.5)&(a[:,j,4]>.002)]:
            run=peak=0
            for k,value in enumerate(mask):
                if a[k,j,7]>.5:run=0
                run=run+1 if value else 0;peak=max(peak,run)
            counts.append(peak*.02)
        longest.append(dict(env=j,contact_free_s=counts[0],contact_free_and_mesh_clear_s=counts[1],
            maximum_gap_mm=float(a[:,j,4].max()*1000),reward_flight_frontier_s=float(a[:,j,5].max()),
            reward_com_velocity_frontier_m_s=float(a[:,j,6].max()),episode_resets=int(a[:,j,7].sum())))
    destination=OUT/f'gpu_probe_{args.label}'
    np.savez_compressed(str(destination)+'.npz',trace=a)
    report=dict(checkpoint=str(args.checkpoint),seed=71,deterministic_actor=True,observation_noise=True,
        randomized_training_domain=True,sampled_at_policy_50hz=True,not_CPU_acceptance=True,worlds=longest)
    Path(str(destination)+'.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
