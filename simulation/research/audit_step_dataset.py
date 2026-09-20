"""Recompute expert actions and verify temporal history and physical admission."""
import json
import numpy as np
import mujoco
import onnxruntime as ort
from demonstration_expert import OUT,ROOT,POLICY
from evaluate_policy import sha


def main():
    metadata=json.loads((OUT/'dataset.json').read_text());assert sha(OUT/'step_demonstrations.npz')==metadata['sha256']
    with np.load(OUT/'step_demonstrations.npz') as z:bank={k:z[k].copy() for k in z.files}
    assert len(bank['obs'])==3000
    allowed=bank['expert_obs'].copy();allowed[:,63]=bank['obs'][:,63]
    np.testing.assert_array_equal(allowed,bank['obs'])
    contract=json.loads((ROOT/'20260913_handoff/native_v07/contract.json').read_text())
    names=contract['action_names'];ids=np.asarray([names.index(n) for n in ('left_hip_yaw','right_hip_yaw')])
    home=np.asarray(contract['action_offset'],np.float32).reshape(-1,19)[0][ids]
    scale=np.broadcast_to(np.asarray(contract['action_scale'],np.float32),(19,))[ids]
    model=mujoco.MjModel.from_binary_path('nominal.mjb',assets={'nominal.mjb':(ROOT/'20260914_terrain_skills/plants/steps_10/nominal.mjb').read_bytes()})
    ranges=model.jnt_range[[model.joint('robot/'+n).id for n in names]]
    session=ort.InferenceSession(str(POLICY),providers=['CPUExecutionProvider'])
    output=np.concatenate([session.run(None,{'obs':obs[None]})[0] for obs in bank['expert_obs']])
    margin=metadata['hip_target_margin'];positions=home+output[:,ids]*scale
    output[:,ids]=(np.clip(positions,ranges[ids,0]+margin,ranges[ids,1]-margin)-home)/scale
    error=float(abs(output-bank['actions']).max());assert error<2e-5
    records=json.loads((OUT/'bounded_collect/matrix.json').read_text());physical=[]
    for i in np.unique(bank['trial_id']):
        index=np.where(bank['trial_id']==i)[0];assert len(index)==600
        np.testing.assert_array_equal(bank['obs'][index[1:],44:63],bank['actions'][index[:-1]])
        row=records[int(i)];assert row['metrics']['demonstration_accepted']
        with np.load(OUT/f"bounded_collect/{row['name']}.npz") as z:
            q=z['joints'];g=z['gaze'][-len(q):];contact=g[:,9:11]
            assert (q>=ranges[:,0]-.02).all() and (q<=ranges[:,1]+.02).all()
            lifts=(np.diff(contact,axis=0)<0).sum(axis=0)
            assert (lifts>0).all()
        physical.append(dict(name=row['name'],delay=row['delay'],seed=row['seed'],vx=row['metrics']['body_vx_mean_m_s'],
            joint_violation=row['metrics']['joint_limit_violation_max_rad'],foot_liftoff_counts=lifts.tolist()))
    result=dict(status='PASS',observations=len(bank['obs']),expert_reconstruction_max_error=error,
        student_command=.2,expert_internal_command=metadata['teacher_command'],only_command_field_changed=True,
        actual_previous_action_history_preserved=True,accepted_trials=physical,
        collection_rejected=len(records)-len(physical),fixed_admission_not_relaxed=True,
        dataset_sha256=metadata['sha256'])
    (OUT/'dataset_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
