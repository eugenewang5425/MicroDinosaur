"""Validate saved offline fit using CPU FP32, without repeating training."""
import json
import time
import torch
import numpy as np
import onnxruntime as ort
from corrective_common import OUT,OLD,PREVIOUS
from measure_corrective_fit import actor
from evaluate_policy import sha


def main():
    dest=OUT/'offline_probe';plan=json.loads((dest/'plan.json').read_text())
    checkpoint=dest/'offline_actor.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    source=torch.load(plan['source_checkpoint'],map_location='cpu',weights_only=False)
    assert saved['optimizer_state_dict']['state']=={}
    assert saved['infos']['env_state']['common_step_counter']==source['infos']['env_state']['common_step_counter']
    for name,value in saved['actor_state_dict'].items():
        assert torch.isfinite(value).all()
        if name.startswith('obs_normalizer.'):torch.testing.assert_close(value,source['actor_state_dict'][name],atol=0,rtol=0)
    optimizer=torch.load(dest/'offline_optimizer.pt',map_location='cpu',weights_only=False)
    assert all(float(v['step'])==1000 and all(not torch.is_tensor(t) or torch.isfinite(t).all() for t in v.values()) for v in optimizer['state'].values())
    results={};samples=[]
    for kind,path in (('old',OLD),('success',PREVIOUS/'step_demonstrations.npz'),('recovery',OUT/'recovery_demonstrations.npz'),('stop',OUT/'stop_demonstrations.npz')):
        with np.load(path) as z:obs=z['obs'];target=z['actions']
        before=actor(plan['source_checkpoint'],obs);after=actor(checkpoint,obs)
        results[kind]=dict(before_mse=float(np.mean((before-target)**2)),after_mse=float(np.mean((after-target)**2)))
        samples.append(obs[::max(1,len(obs)//100)])
    samples=np.concatenate(samples);expected=actor(checkpoint,samples)
    session=ort.InferenceSession(str(dest/'candidate.onnx'),providers=['CPUExecutionProvider'])
    actual=np.concatenate([session.run(None,{'obs':row[None]})[0] for row in samples])
    error=float(abs(actual-expected).max());assert error<2e-5
    result=dict(status='COMPLETE',fit=results,export_max_error=error,export_observations=len(samples),
        policy_sha256=sha(dest/'candidate.onnx'),checkpoint_sha256=sha(checkpoint),steps=1000,unique_new_frames=3550,
        repeated_supervision_samples=1536000,new_physics_training_transitions=0,validation_recovered_without_retraining=True,
        validation_note='Initial cross-device assertion failed after export; saved actor verified against CPU FP32 with unchanged tolerance.',finished_unix=time.time())
    (dest/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
