"""Check actual PPO updates, source hashes, finite state, and normalized deployment export."""
import json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from evaluate_policy import sha
from foot_flight_cfg import OUT
ROOT=OUT.parent

RUN=Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_foot_flight/20260914_train_512x401')


def main():
    p=json.loads((RUN/'run_provenance.json').read_text());assert p['status']=='COMPLETE'
    source=torch.load(p['source_checkpoint'],map_location='cpu',weights_only=False)
    final=torch.load(p['final_checkpoint'],map_location='cpu',weights_only=False)
    checkpoints=[]
    for path in sorted(RUN.glob('model_*.pt')):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        rates=[group['lr'] for group in saved['optimizer_state_dict']['param_groups']]
        assert all(abs(rate-5e-5)<1e-12 for rate in rates)
        for group in ['actor_state_dict','critic_state_dict']:
            assert all(not torch.is_tensor(v) or torch.isfinite(v).all() for v in saved[group].values())
        assert all(not torch.is_tensor(v) or torch.isfinite(v).all()
            for state in saved['optimizer_state_dict']['state'].values() for v in state.values())
        checkpoints.append(dict(path=str(path),sha256=sha(path),iteration=saved['iter'],learning_rates=rates))
    delta=final['infos']['env_state']['common_step_counter']-source['infos']['env_state']['common_step_counter']
    assert delta==401*24
    increments={float(v['step']-source['optimizer_state_dict']['state'][i]['step']) for i,v in final['optimizer_state_dict']['state'].items()}
    assert increments=={401*20}
    obs=np.load(ROOT/'20260914_transition_refine/v7_anchor.npz')['obs'][::30].copy()
    obs[:,63:66]=0
    obs[:,72]=np.resize(np.array([-.05,-.03,.015,0.],np.float32),len(obs))
    actor=final['actor_state_dict'];x=torch.from_numpy(obs)
    x=(x-actor['obs_normalizer._mean'])/(actor['obs_normalizer._std']+.01)
    for i in [0,2,4,6]:
        x=torch.nn.functional.linear(x,actor[f'mlp.{i}.weight'],actor[f'mlp.{i}.bias'])
        if i<6:x=torch.nn.functional.elu(x)
    session=ort.InferenceSession(p['onnx'],providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape==[1,81] and session.get_outputs()[0].shape==[1,19]
    prediction=np.concatenate([session.run(None,{'obs':o[None]})[0] for o in obs])
    error=float(abs(prediction-x.numpy()).max());assert error<2e-5
    assert sha(p['source_checkpoint'])==p['source_sha256']
    assert all(sha(path)==digest for path,digest in p['source_hashes'].items())
    assert sha('D:/microduck_rl/microdinosaur_p2.onnx')=='0804114efd1e457ec2c297d709c0464fd0c2bfe251cbb36db7de74543857b0e3'
    event=EventAccumulator(str(RUN),size_guidance={'scalars':0});event.Reload();trends={}
    for tag in event.Tags()['scalars']:
        values=np.array([v.value for v in event.Scalars(tag)]);assert np.isfinite(values).all(),tag
        if tag in ['Episode_Reward/joint_margin','Episode_Reward/touchdown_speed']:assert (values<=1e-7).all()
        trends[tag]=dict(n=len(values),first10=float(values[:10].mean()),last10=float(values[-10:].mean()))
    report=dict(passed=True,transitions=delta*512,updates=401,optimizer_steps=sorted(increments),
        source_and_production_unchanged=True,training_sources_unchanged=True,checkpoints=checkpoints,
        export_rows=len(obs),normalized_export_max_error=error,onnx_sha256=sha(p['onnx']),
        training_wall_seconds=p['finished_unix']-p['started_unix'],trends=trends)
    (OUT/'training_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['checkpoints','trends']},indent=2))


if __name__=='__main__':main()
