"""Test directional reward/history symmetry and frozen-policy mirror responses.

This is a diagnostic, not a training ablation or a causal attribution of yaw.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mjlab
import numpy as np
import onnxruntime as ort
import torch
import yaml
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.symmetry_microdinosaur import _OBS_PERM,_OBS_SIGN,_JOINT_PERM,_JOINT_SIGN
from hardware_sim import HardwareCase,HardwareSim

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260913_direction_bias'
RUN=Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_v07_calibration/20260913_s288_v7recipe_4096x101')
SOURCE=Path('D:/microduck_rl/logs/rsl_rl/velocity_microdinosaur/2026-09-13_13-46-57_velocity_microdinosaur')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def paired_env():
    data=SimpleNamespace(root_link_quat_w=torch.tensor([[1.,0.,0.,0.]]*2),
        body_link_ang_vel_w=torch.zeros(2,3,3),joint_vel=torch.zeros(2,19))
    contact=SimpleNamespace(data=SimpleNamespace(found=torch.zeros(2,2)))
    return SimpleNamespace(num_envs=2,device='cpu',scene={'robot':SimpleNamespace(data=data),'feet':contact},
        command_manager=SimpleNamespace(get_command=lambda name:torch.tensor([[.55,0.,0.]]*2)))


def roll(env,x):
    t=torch.tensor([x,-x],dtype=torch.float)
    env.scene['robot'].data.root_link_quat_w=torch.stack([torch.cos(t/2),torch.sin(t/2),torch.zeros(2),torch.zeros(2)],1)


def main():
    OUT.mkdir(exist_ok=True)
    env=paired_env()
    lean_cfg=RewardTermCfg(func=mdp.lean_drift_penalty,weight=-4.)
    contact_cfg=RewardTermCfg(func=mdp.contact_timing_asymmetry,weight=-4.,params={'sensor_name':'feet'})
    lean=mdp.lean_drift_penalty(lean_cfg,env);contact=mdp.contact_timing_asymmetry(contact_cfg,env)
    rng=np.random.default_rng(13)
    errors={'lean_cost':[],'contact_cost':[],'head_gaze_cost':[]}
    cumulative=np.zeros((2,2))
    selector=SceneEntityCfg('robot',body_ids=[0,1,2])
    for t in range(1000):
        roll(env,.07*np.sin(t*.17)+.013*np.sin(t*.039)+rng.normal(0,.002))
        feet=torch.tensor([float(t%23<12),float(t%23>=10)])
        env.scene['feet'].data.found=torch.stack([feet,feet.flip(0)])
        a=lean(env);b=contact(env,sensor_name='feet')
        cumulative+=np.stack([a.numpy(),b.numpy()])
        angular=torch.tensor(rng.normal(size=(3,3)),dtype=torch.float)
        env.scene['robot'].data.body_link_ang_vel_w=torch.stack([angular,angular*torch.tensor([-1.,1.,-1.])])
        c=mdp.head_gaze_penalty(env,asset_cfg=selector)
        for key,cost in zip(errors,(a,b,c)):errors[key].append(float(abs(cost[0]-cost[1])))
    maxima={k:max(v) for k,v in errors.items()}
    assert max(maxima.values())<1e-6
    # Deliberately inject the SAME signed old history into opposite new motions.
    # That is a history-contamination counterfactual, not the current reset path.
    lean.ema[:]=.03
    stale=[]
    for _ in range(100):roll(env,np.deg2rad(2));stale.append(lean(env).numpy().copy())
    stale=np.asarray(stale)
    lean.reset()
    fresh=[]
    for _ in range(100):roll(env,np.deg2rad(2));fresh.append(lean(env).numpy().copy())
    fresh=np.asarray(fresh)
    np.testing.assert_array_equal(fresh[:,0],fresh[:,1])
    configs={}
    for label,path in [('v7',SOURCE),('s288_v7recipe101',RUN)]:
        cfg=yaml.load((path/'params/env.yaml').read_text(),Loader=yaml.BaseLoader)
        configs[label]={name:cfg['rewards'].get(name,{}).get('weight')
            for name in ('lean_drift','contact_timing','head_gaze','head_world_gaze','body_ang_vel')}

    # A shared bank of on-policy v7 observations avoids comparing unrelated
    # state distributions; mirrored observations are counterfactual inputs.
    class Recorder:
        def __init__(self,session):self.session=session;self.samples=[];self.record=False
        def __getattr__(self,name):return getattr(self.session,name)
        def run(self,outputs,feed):
            if self.record:self.samples.append(next(iter(feed.values())).copy())
            return self.session.run(outputs,feed)
    sim=HardwareSim(ROOT/'20260913_handoff/native_v07',ROOT/'20260913_handoff/v7_reference.onnx',
        HardwareCase(physics_dt=.00125))
    recorder=Recorder(sim.session);sim.session=recorder
    for vx in (0.,.55):
        command=np.zeros(18);command[0]=vx
        for seed in range(3):
            sim.reset(seed,seed>0)
            for tick in range(200):
                recorder.record=tick>=100 and tick%5==0
                sim.step(command)
    bank=np.concatenate(recorder.samples)
    mirror=(bank[:,_OBS_PERM]*np.array(_OBS_SIGN,dtype=np.float32)).astype(np.float32)
    np.testing.assert_array_equal(mirror[:,_OBS_PERM]*np.array(_OBS_SIGN),bank)
    np.savez_compressed(OUT/'shared_observation_bank.npz',original=bank,mirrored=mirror)
    policy_results={}
    for label,path in [('v7',ROOT/'20260913_handoff/v7_reference.onnx'),('s288_v7recipe101',RUN/'candidate.onnx')]:
        session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
        name=session.get_inputs()[0].name
        predict=lambda obs:np.concatenate([session.run(None,{name:x[None]})[0] for x in obs])
        actions=predict(bank);mirrored_actions=predict(mirror)
        residual=mirrored_actions-actions[:,_JOINT_PERM]*np.array(_JOINT_SIGN)
        legs=[0,1,2,3,4,10,11,12,13,14]
        policy_results[label]={'onnx_sha256':sha(path),'observation_count':len(bank),
            'all_action_mirror_residual_rms':float(np.sqrt(np.mean(residual**2))),
            'leg_action_mirror_residual_rms':float(np.sqrt(np.mean(residual[:,legs]**2))),
            'per_joint_residual_rms':np.sqrt(np.mean(residual**2,axis=0)).tolist()}
    report={'status':'DIAGNOSTICS_COMPLETE_NOT_CAUSAL_PROOF',
        'reward_source_sha256':sha(Path(mdp.__file__)),
        'neutral_history_mirror_cost_max_error_over_1000_steps':maxima,
        'neutral_history_raw_cost_sums':cumulative.tolist(),
        'injected_stale_history':{'value':.03,'opposite_roll_deg':[2,-2],
            'first_step_costs':stale[0].tolist(),'raw_100_step_cost_sums':stale.sum(0).tolist(),
            'fresh_reset_100_step_cost_sums':fresh.sum(0).tolist(),
            'current_reset_clears_bias':True},
        'ema_half_life_seconds_at_50hz':{'lean':float(.02*np.log(.5)/np.log(.99)),
            'contact':float(.02*np.log(.5)/np.log(.995))},
        'saved_reward_weights':configs,'frozen_policy_mirror_response':policy_results,
        'mirror_response_is_not_physical_gait_symmetry_or_cause':True,
        'new_training_performed':False}
    (OUT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
