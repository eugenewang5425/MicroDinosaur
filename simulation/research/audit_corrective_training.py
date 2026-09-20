"""Verify optimizer updates, immutable training inputs and official export."""
import json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from corrective_common import OUT,SOURCE,RUNS,OLD
from evaluate_policy import sha


def main():
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)
    batches=[]
    for path in (OLD,OUT/'recovery_demonstrations.npz',OUT/'stop_demonstrations.npz'):
        with np.load(path) as z:batches.append(z['obs'][::max(1,len(z['obs'])//150)])
    obs=np.concatenate(batches);results={}
    for arm in ('control','corrective'):
        folder=RUNS/f'microdinosaur_corrective_{arm}/20260914_train_512x201'
        p=json.loads((folder/'run_provenance.json').read_text());assert p['status']=='COMPLETE'
        assert (p['envs'],p['iterations'],p['seed'])==(512,201,47)
        assert p['source_sha256']==sha(SOURCE)
        assert all(sha(Path(path))==digest for path,digest in p['source_hashes'].items())
        for path in folder.glob('model_*.pt'):
            saved=torch.load(path,map_location='cpu',weights_only=False)
            assert all(g['lr']==3e-5 for g in saved['optimizer_state_dict']['param_groups'])
            assert all(torch.isfinite(v).all() for k in ('actor_state_dict','critic_state_dict') for v in saved[k].values())
            assert all(not torch.is_tensor(v) or torch.isfinite(v).all() for s in saved['optimizer_state_dict']['state'].values() for v in s.values())
        final=torch.load(p['final_checkpoint'],map_location='cpu',weights_only=False)
        delta=final['infos']['env_state']['common_step_counter']-source['infos']['env_state']['common_step_counter'];assert delta==201*24
        increments={float(v['step']-source['optimizer_state_dict']['state'][i]['step']) for i,v in final['optimizer_state_dict']['state'].items()}
        assert increments=={4020,4824}
        weights=final['actor_state_dict'];x=torch.tensor(obs)
        x=(x-weights['obs_normalizer._mean'])/(weights['obs_normalizer._std']+.01)
        for i in (0,2,4,6):
            x=torch.nn.functional.linear(x,weights[f'mlp.{i}.weight'],weights[f'mlp.{i}.bias'])
            if i<6:x=torch.nn.functional.elu(x)
        session=ort.InferenceSession(p['onnx'],providers=['CPUExecutionProvider'])
        y=np.concatenate([session.run(None,{'obs':row[None]})[0] for row in obs]);error=float(abs(y-x.numpy()).max());assert error<2e-5
        events=EventAccumulator(str(folder),size_guidance={'scalars':0});events.Reload();trends={}
        for name in events.Tags()['scalars']:
            values=np.asarray([row.value for row in events.Scalars(name)]);assert np.isfinite(values).all()
            if 'imitation_' in name:trends[name]=dict(first10=float(values[:10].mean()),last10=float(values[-10:].mean()))
        results[arm]=dict(status='PASS',updates=201,transitions=delta*512,optimizer_step_increments=sorted(increments),
            export_max_error=error,export_observations=len(obs),imitation_trends=trends,
            onnx_sha256=sha(Path(p['onnx'])),checkpoint_sha256=sha(Path(p['final_checkpoint'])))
    production=sha(Path('D:/microduck_rl/microdinosaur_p2.onnx'))
    assert production=='0804114efd1e457ec2c297d709c0464fd0c2bfe251cbb36db7de74543857b0e3'
    (OUT/'training_audit.json').write_text(json.dumps(dict(status='PASS',arms=results,production_sha256=production),indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
