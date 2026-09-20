"""Checkpoint, optimizer, normalizer, export and frozen-source validation."""
import json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from evaluate_policy import sha

ROOT=Path(__file__).parent;OUT=ROOT/'20260914_transition_refine'
RUN=Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_transition_refine/20260914_train_512x201')


def main():
    p=json.loads((RUN/'run_provenance.json').read_text());assert p['status']=='COMPLETE'
    source=torch.load(p['source_checkpoint'],map_location='cpu',weights_only=False)
    final=torch.load(p['final_checkpoint'],map_location='cpu',weights_only=False)
    records=[]
    for path in sorted(RUN.glob('model_*.pt')):
        state=torch.load(path,map_location='cpu',weights_only=False)
        rates=[g['lr'] for g in state['optimizer_state_dict']['param_groups']]
        assert all(x==3e-5 for x in rates)
        for name in ('actor_state_dict','critic_state_dict'):
            assert all(torch.isfinite(v).all() for v in state[name].values())
        assert all(not torch.is_tensor(v) or torch.isfinite(v).all()
            for s in state['optimizer_state_dict']['state'].values() for v in s.values())
        records.append(dict(file=path.name,iteration=state['iter'],infos=state['infos'],learning_rates=rates))
    delta=final['infos']['env_state']['common_step_counter']-source['infos']['env_state']['common_step_counter']
    assert delta==201*24
    increments=[float(v['step']-source['optimizer_state_dict']['state'][k]['step'])
        for k,v in final['optimizer_state_dict']['state'].items()]
    assert set(increments)=={201*20,201*22},set(increments)
    actor=final['actor_state_dict'];obs=np.load(OUT/'v7_anchor.npz')['obs'][::20]
    x=torch.from_numpy(obs)
    x=(x-actor['obs_normalizer._mean'])/(actor['obs_normalizer._std']+.01)
    for i in (0,2,4,6):
        x=torch.nn.functional.linear(x,actor[f'mlp.{i}.weight'],actor[f'mlp.{i}.bias'])
        if i<6:x=torch.nn.functional.elu(x)
    sess=ort.InferenceSession(p['onnx'],providers=['CPUExecutionProvider'])
    # Official deployment export deliberately has batch size 1.
    pred=np.concatenate([sess.run(None,{'obs':row[None]})[0] for row in obs])
    assert pred.shape==(len(obs),19)
    error=float(abs(pred-x.detach().numpy()).max());assert error<2e-5
    assert sha(Path(p['source_checkpoint']))==p['source_sha256']
    assert sha(Path(p['teacher_checkpoint']))==p['teacher_sha256']
    assert all(sha(Path(path))==digest for path,digest in p['source_hashes'].items())
    assert sha(Path(p['onnx']))==p['onnx_sha256']
    published=Path('D:/microduck_rl/microdinosaur_p2.onnx')
    assert sha(published)=='0804114efd1e457ec2c297d709c0464fd0c2bfe251cbb36db7de74543857b0e3'
    event=EventAccumulator(str(RUN),size_guidance={'scalars':0});event.Reload()
    trends={}
    for name in event.Tags()['scalars']:
        values=np.array([r.value for r in event.Scalars(name)])
        assert np.isfinite(values).all(),name
        if name.startswith('Loss/') or name.startswith('Episode_Reward/') or name.startswith('Episode_Termination/'):
            trends[name]=dict(first10_mean=float(values[:10].mean()),last10_mean=float(values[-10:].mean()),
                minimum=float(values.min()),maximum=float(values.max()),n=len(values))
    result=dict(status='PASS',updates=201,environments=512,transitions=delta*512,checkpoints=records,
        optimizer_step_increments=sorted(set(increments)),source_teacher_published_unchanged=True,
        training_source_hashes_unchanged=True,onnx_matches_normalized_checkpoint_max_error_rad=error,
        training_wall_seconds=p['finished_unix']-p['started_unix'],onnx_sha256=p['onnx_sha256'],trends=trends)
    (OUT/'training_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('trends','checkpoints')},indent=2))


if __name__=='__main__':main()
