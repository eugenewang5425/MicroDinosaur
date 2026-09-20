"""Read actual GPU observations to audit the historical teacher gate."""
import json
import torch
import numpy as np
from mjlab.envs import ManagerBasedRlEnv
from demonstration_cfg import build_config
from demonstration_expert import OUT,SOURCE


def main():
    _,cfg=build_config('legacy',16,43);env=ManagerBasedRlEnv(cfg=cfg.env,device='cuda:0');env.reset()
    state=torch.load(SOURCE,map_location=env.device,weights_only=False)['actor_state_dict']
    counts={name:dict(observations=0,eligible=0) for name in ('flat','up','down')}
    for step in range(200):
        obs=env.obs_buf['actor'];eligible=obs[:,72].abs()<1e-6
        for index,name in enumerate(counts):
            ids=env.scene.terrain.terrain_types==index
            counts[name]['observations']+=int(ids.sum());counts[name]['eligible']+=int((ids&eligible).sum())
        x=(obs-state['obs_normalizer._mean'])/(state['obs_normalizer._std']+.01)
        for i in (0,2,4,6):
            x=torch.nn.functional.linear(x,state[f'mlp.{i}.weight'],state[f'mlp.{i}.bias'])
            if i<6:x=torch.nn.functional.elu(x)
        env.step(x)
        assert torch.isfinite(env.obs_buf['actor']).all()
    for r in counts.values():r['eligible_fraction']=r['eligible']/r['observations']
    assert counts['up']['eligible_fraction']==counts['down']['eligible_fraction']==1.
    env.close()
    bank=OUT.parent/'20260914_transition_refine/v7_anchor.npz'
    with np.load(bank) as z:
        archive=dict(rows=len(z['obs']),commands={str(float(k)):int(v) for k,v in zip(*np.unique(z['obs'][:,63],return_counts=True))},
            nonzero_crouch_rows=int(np.count_nonzero(z['obs'][:,72])))
    result=dict(status='PASS',gpu_environment_count=16,control_steps=200,counts=counts,archive=archive,
        finding='All actual terrain observations qualify for the old teacher because body-height command is zero; this demonstrates scope, not causal harm.')
    (OUT/'teacher_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
