"""Independent numeric check of the normalized 81 -> 19 ONNX export."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from torch import nn
import onnxruntime as ort
from evaluate_policy import sha
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
record=json.loads((a.run/'run_provenance.json').read_text());checkpoint=Path(record['final_checkpoint'])
state=torch.load(checkpoint,map_location='cpu',weights_only=False)['actor_state_dict']
net=nn.Sequential(nn.Linear(81,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,19))
net.load_state_dict({k.removeprefix('mlp.'):v for k,v in state.items() if k.startswith('mlp.')});net.eval()
obs=np.random.default_rng(914).normal(0,.3,(117,81)).astype(np.float32);obs[0]=0
mean=state['obs_normalizer._mean'];std=state['obs_normalizer._std']
with torch.no_grad():expected=net((torch.from_numpy(obs)-mean)/(std+.01)).numpy()
session=ort.InferenceSession(str(a.run/'candidate.onnx'),providers=['CPUExecutionProvider'])
actual=np.concatenate([session.run(None,{'obs':r[None]})[0] for r in obs])
error=float(np.max(abs(actual-expected)));assert error<2e-5,error
a.output.write_text(json.dumps(dict(samples=len(obs),max_absolute_error=error,
    onnx_sha256=sha(a.run/'candidate.onnx'),checkpoint_sha256=sha(checkpoint),official_normalizer_verified=True),indent=2))
print(error)
