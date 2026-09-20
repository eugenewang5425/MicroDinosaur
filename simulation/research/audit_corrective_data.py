"""Recompute executed expert labels and verify disjoint seed sets/history."""
import json
import numpy as np
import onnxruntime as ort
from corrective_common import OUT,STUDENT,STOP_TEACHER
from demonstration_expert import experiment as make_expert,POLICY
from evaluate_policy import sha


def main():
    results={}
    for kind in ('recovery','stop'):
        rows=json.loads((OUT/f'collection/{kind}/matrix.json').read_text());collected=[];errors=[]
        for r in rows:
            if not r['metrics']['accepted']:continue
            assert r['seed'] not in (1,2,3,201,202,203)
            with np.load(OUT/f"collection/{kind}/{r['name']}.npz") as z:
                obs=z['obs'];actions=z['actions'];mask=z['teacher_mask']
                np.testing.assert_array_equal(obs[1:,44:63],actions[:-1])
                assert mask.sum()==round((len(obs)*.02-r['switch'])/.02)
                if kind=='recovery':
                    unused,teacher=make_expert(.5,r['delay'],.03);del unused
                else:teacher=ort.InferenceSession(str(STOP_TEACHER),providers=['CPUExecutionProvider'])
                y=np.concatenate([teacher.run(None,{'obs':x[None]})[0] for x in obs[mask]])
                np.testing.assert_array_equal(y,actions[mask]);errors.append(float(abs(y-actions[mask]).max()))
                collected.append(obs[mask])
        with np.load(OUT/f'{kind}_demonstrations.npz') as z:
            np.testing.assert_array_equal(z['obs'],np.concatenate(collected))
            assert all(np.isfinite(z[k]).all() for k in z.files)
            if kind=='stop':
                for trial in np.unique(z['trial_id']):
                    command_rows=z['obs'][z['trial_id']==trial]
                    np.testing.assert_allclose(command_rows[:,63],np.maximum(.32-.03*np.arange(len(command_rows)),0),atol=1e-6)
                    np.testing.assert_array_equal(command_rows[50:,63:66],0)
                    assert np.max(np.abs(np.diff(command_rows[:,65])))<=.03001
            else:assert np.allclose(z['obs'][:,63],.2)
            frames=len(z['obs'])
        results[kind]=dict(accepted=len(collected),attempted=len(rows),frames=frames,max_label_error=max(errors),
            dataset_sha256=sha(OUT/f'{kind}_demonstrations.npz'),teacher_sha256=sha(POLICY if kind=='recovery' else STOP_TEACHER))
    result=dict(status='PASS',student_sha256=sha(STUDENT),datasets=results,history_exact=True,holdout_disjoint=True)
    (OUT/'dataset_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
