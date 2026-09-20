"""Verify the actual saved optimizer, counters, finite tensors and exports."""
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch

OUT = Path(__file__).parent/'20260914_terrain_skills'
RUNS = Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_terrain_skills')


def main():
    records=[]
    for name in ('20260914_terrain_train_512x151', '20260914_crouch_train_512x151'):
        root=RUNS/name; p=json.loads((root/'run_provenance.json').read_text())
        if p['status']!='COMPLETE':
            records.append(dict(run=name,status=p['status'])); continue
        source=torch.load(p['source_checkpoint'],map_location='cpu',weights_only=False)
        final=torch.load(p['final_checkpoint'],map_location='cpu',weights_only=False)
        rates=[]; counters=[]
        for path in sorted(root.glob('model_*.pt')):
            d=torch.load(path,map_location='cpu',weights_only=False)
            lr=[g['lr'] for g in d['optimizer_state_dict']['param_groups']]
            rates.append(dict(checkpoint=path.name,learning_rates=lr))
            counters.append(dict(checkpoint=path.name,iteration=d['iter'],infos=d['infos']))
            for group in ('actor_state_dict','critic_state_dict'):
                assert all(torch.isfinite(v).all() for v in d[group].values())
        assert len({lr for r in rates for lr in r['learning_rates']})==1
        for key,field in (('final_checkpoint','final_sha256'),('onnx','onnx_sha256')):
            assert hashlib.sha256(Path(p[key]).read_bytes()).hexdigest()==p[field]
        ort_session=ort.InferenceSession(p['onnx'],providers=['CPUExecutionProvider'])
        obs=np.random.default_rng(914).normal(0,.1,(1,81)).astype('float32')
        y=ort_session.run(None,{'obs':obs})[0]
        assert y.shape==(1,19) and np.isfinite(y).all()
        norm=[k for k in final['actor_state_dict'] if 'normalizer' in k]
        assert norm, 'Missing normalized actor'
        records.append(dict(run=name,status='PASS',optimizer_checkpoints=rates,
            checkpoints=counters,source_iteration=source['iter'],final_iteration=final['iter'],
            updates=p['iterations'],new_transitions=p['envs']*p['iterations']*24,
            normalizer_keys=norm,export_shape=list(y.shape),
            learning_rate_note='Initial script configured 3e-5; PPO optimizer restore actually retained source LR. These completed runs used the recorded constant effective LR. Future entry point matches its configuration to that resumed LR.',
            training_head_loop=False,confirmation_head_loop=True))
    (OUT/'training_audit.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps([dict(run=r['run'],status=r['status']) for r in records]))


if __name__=='__main__': main()
