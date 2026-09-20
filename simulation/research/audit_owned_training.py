"""Validate final learned model, actual optimization and normalized export."""
import json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from evaluate_policy import sha
from terrain_lane_cfg import OUT

RUN=Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_imu_owned_terrain/20260914_resume_512x100')
FIRST=RUN.parent/'20260914_train_512x301'


def main():
    p=json.loads((RUN/'run_provenance.json').read_text());assert p['status']=='COMPLETE'
    n=p['iterations'];source=torch.load(p['source_checkpoint'],map_location='cpu',weights_only=False)
    final=torch.load(p['final_checkpoint'],map_location='cpu',weights_only=False);checkpoints=[]
    for path in sorted(RUN.glob('model_*.pt')):
        d=torch.load(path,map_location='cpu',weights_only=False)
        rates=[g['lr'] for g in d['optimizer_state_dict']['param_groups']];assert all(v==3e-5 for v in rates)
        assert all(torch.isfinite(v).all() for name in ('actor_state_dict','critic_state_dict') for v in d[name].values())
        assert all(not torch.is_tensor(v) or torch.isfinite(v).all() for s in d['optimizer_state_dict']['state'].values() for v in s.values())
        checkpoints.append(dict(name=path.name,iteration=d['iter'],infos=d['infos'],learning_rates=rates))
    delta=final['infos']['env_state']['common_step_counter']-source['infos']['env_state']['common_step_counter'];assert delta==n*24
    steps={float(v['step']-source['optimizer_state_dict']['state'][k]['step']) for k,v in final['optimizer_state_dict']['state'].items()}
    assert steps=={n*20,n*22},steps
    obs=np.load(OUT.parent/'20260914_transition_refine/v7_anchor.npz')['obs'][::20]
    d=final['actor_state_dict'];x=torch.tensor(obs)
    x=(x-d['obs_normalizer._mean'])/(d['obs_normalizer._std']+.01)
    for i in (0,2,4,6):
        x=torch.nn.functional.linear(x,d[f'mlp.{i}.weight'],d[f'mlp.{i}.bias'])
        if i<6:x=torch.nn.functional.elu(x)
    session=ort.InferenceSession(p['onnx'],providers=['CPUExecutionProvider'])
    y=np.concatenate([session.run(None,{'obs':row[None]})[0] for row in obs]);error=float(abs(y-x.numpy()).max());assert error<2e-5
    assert sha(Path(p['source_checkpoint']))==p['source_sha256'] and sha(Path(p['teacher_checkpoint']))==p['teacher_sha256']
    assert all(sha(Path(path))==digest for path,digest in p['source_hashes'].items())
    assert sha(Path('D:/microduck_rl/microdinosaur_p2.onnx'))=='0804114efd1e457ec2c297d709c0464fd0c2bfe251cbb36db7de74543857b0e3'
    events=EventAccumulator(str(RUN),size_guidance={'scalars':0});events.Reload();trends={}
    for name in events.Tags()['scalars']:
        values=np.asarray([r.value for r in events.Scalars(name)]);assert np.isfinite(values).all()
        if name.startswith(('Loss/','Episode_Reward/','Episode_Termination/')):
            trends[name]=dict(first10_mean=float(values[:10].mean()),last10_mean=float(values[-10:].mean()),
                minimum=float(values.min()),maximum=float(values.max()))
    result=dict(status='PASS',updates=n,envs=p['envs'],transitions=delta*p['envs'],checkpoints=checkpoints,
        optimizer_step_increments=sorted(steps),export_normalized_fp32_error_rad=error,
        source_teacher_production_unchanged=True,training_source_hashes_unchanged=True,
        wall_seconds=p['finished_unix']-p['started_unix'],onnx_sha256=p['onnx_sha256'],trends=trends)
    original_p=json.loads((FIRST/'run_provenance.json').read_text());assert original_p['status']=='FAILED'
    original=torch.load(original_p['source_checkpoint'],map_location='cpu',weights_only=False)
    kept=source['infos']['env_state']['common_step_counter']-original['infos']['env_state']['common_step_counter']
    assert kept==201*24 and kept+delta==301*24
    original_checkpoints=[]
    for path in sorted(FIRST.glob('model_*.pt')):
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert all(torch.isfinite(v).all() for name in ('actor_state_dict','critic_state_dict') for v in state[name].values())
        assert all(g['lr']==3e-5 for g in state['optimizer_state_dict']['param_groups'])
        original_checkpoints.append(dict(name=path.name,infos=state['infos']))
    import re
    executed=len(re.findall(r'Learning iteration \d+/',(OUT/'training.log').read_text(encoding='utf-8')))
    assert executed==217
    result.update(original_status='FAILED: nonfinite physics followed by Windows symlink permission error',
        original_checkpoints=original_checkpoints,retained_updates=301,retained_transitions=(kept+delta)*512,
        discarded_completed_updates=executed-201,total_logged_updates=executed+n,
        total_logged_transitions=(executed+n)*24*512,physics_rng_state_reinitialized=True,
        original_numeric_root_cause_resolved=False)
    (OUT/'training_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('checkpoints','trends')},indent=2))


if __name__=='__main__':main()
