"""Real GPU reset check: S288 friction varies, stays nominal-relative and local."""
import json
from pathlib import Path
import mjlab
import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.s288_profile import apply_s288_protocol

root=Path(__file__).resolve().parents[1]
cfg=load_env_cfg('Mjlab-Velocity-Flat-MicroDinosaur',play=False)
cfg.scene.num_envs=8
cfg.scene.entities['robot'].spec_fn=lambda:mujoco.MjSpec.from_file(
    'D:/microduck_rl/src/mjlab_microduck/robot/microdinosaur_v07/robot_microdinosaur_v07.xml')
apply_s288_protocol(cfg)
for name in ('lean_drift','contact_timing'):cfg.rewards.pop(name,None)
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
    env.reset()
    event=env.event_manager.get_term_cfg('randomize_joint_friction')
    assert event.func.__name__=='joint_friction'
    lo,hi=event.params['ranges']
    asset=env.scene['robot']
    # All 19 actuators are one-DOF hinges.
    adr=asset.indexing.joint_v_adr[event.params['asset_cfg'].joint_ids]
    assert len(adr)==19
    field=env.sim.model.dof_frictionloss
    initial=field[:,adr].clone()
    assert float(initial.std())>0
    reset=torch.tensor([1,3,5],device=env.device)
    kept=torch.tensor([0,2,4,6,7],device=env.device)
    for _ in range(15):
        env.reset(env_ids=reset)
        x=field[:,adr]
        assert bool(torch.all(x>=.01*lo-1e-7) and torch.all(x<=.01*hi+1e-7))
        torch.testing.assert_close(x[kept],initial[kept],atol=0,rtol=0)
    assert not torch.equal(field[:,adr][reset],initial[reset])
    report={'status':'PASS','num_envs':8,'actuated_dofs':len(adr),'partial_resets':15,
        'scale_range':[lo,hi],'friction_min':float(field[:,adr].min()),
        'friction_max':float(field[:,adr].max()),'non_reset_worlds_unchanged':True,
        'non_accumulating':True,'applies_after_101_update_candidate':True}
    (root/'research/20260913_hardware_physics/s288_reset_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
finally:env.close()
