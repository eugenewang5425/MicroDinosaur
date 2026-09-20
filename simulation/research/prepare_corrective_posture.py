"""Recover already-executed, accepted crouch/return prefixes for rehearsal."""
import json
import numpy as np
import onnxruntime as ort
from corrective_common import OUT,STUDENT
from evaluate_policy import sha


def main():
    records=json.loads((OUT/'collection/stop/matrix.json').read_text());obs=[];actions=[];trial=[]
    session=ort.InferenceSession(str(STUDENT),providers=['CPUExecutionProvider']);accepted=[]
    for i,r in enumerate(records):
        m=r['metrics']
        if abs(m['actual_drop_mm']-40)>6 or abs(m['return_height_error_mm'])>5:continue
        with np.load(OUT/f"collection/stop/{r['name']}.npz") as z:
            f=z['gaze'];prefix=f[(f[:,0]>=6)&(f[:,0]<17)]
            assert prefix[:,3].min()>=.055 and prefix[:,14].max()<=60
            assert not z['teacher_mask'][100:550].any()
            x=z['obs'][100:550];y=z['actions'][100:550]
            predicted=np.concatenate([session.run(None,{'obs':row[None]})[0] for row in x])
            np.testing.assert_array_equal(predicted,y)
            np.testing.assert_array_equal(x[1:,44:63],y[:-1])
            obs.extend(x);actions.extend(y);trial.extend([i]*len(x));accepted.append(r['name'])
    assert len(obs)>=1350
    path=OUT/'posture_retention.npz';np.savez_compressed(path,obs=np.asarray(obs,np.float32),actions=np.asarray(actions,np.float32),trial_id=np.asarray(trial,np.int32))
    result=dict(status='PASS',frames=len(obs),trials=accepted,source_policy_sha256=sha(STUDENT),dataset_sha256=sha(path),
        action_labels_exact=True,new_physics_trials=0,source='Existing student-executed t=2..11s crouch/return prefixes; later stop failures do not relabel earlier motion.')
    (OUT/'posture_dataset.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':main()
