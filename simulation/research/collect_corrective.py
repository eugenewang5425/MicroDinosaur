"""Collect executed recovery actions after a student prefix, preserving state."""
import argparse
import json
import numpy as np
import onnxruntime as ort
from corrective_common import OUT,STUDENT,STOP_TEACHER
from demonstration_expert import experiment as make_expert
from evaluate_owned_terrain import cases,experiment,evaluate
from evaluate_policy import sha


class Takeover:
    def __init__(self,student,teacher,tick):
        self.student,self.teacher,self.tick=student,teacher,tick
        self.obs=[];self.actions=[];self.teacher_mask=[]
    def get_inputs(self):return self.student.get_inputs()
    def get_outputs(self):return self.student.get_outputs()
    def run(self,outputs,inputs):
        use_teacher=len(self.obs)>=self.tick
        result=(self.teacher if use_teacher else self.student).run(outputs,inputs)
        self.obs.append(inputs['obs'][0].copy());self.actions.append(result[0][0].copy())
        self.teacher_mask.append(use_teacher)
        return result


def run(kind,delay,seed,switch):
    case=next(c for c in cases() if c['terrain']=='steps_10' and c['speed']==.2) if kind=='recovery' else next(c for c in cases() if c['program']=='resume' and c['delay']==delay)
    case=dict(case,delay=delay);e=experiment(STUDENT,case)
    if kind=='recovery':
        unused,teacher=make_expert(.50,delay,.03);del unused
    else:teacher=ort.InferenceSession(str(STOP_TEACHER),providers=['CPUExecutionProvider'])
    recorder=Takeover(e.sim.session,teacher,round(switch/.02));e.sim.session=recorder
    q=[];before=e.sim.substep_callback
    def capture():
        before()
        if e.active:q.append(e.sim.data.qpos[e.sim.jadr].copy())
    e.sim.substep_callback=capture
    m,t=evaluate(e,case,seed);q=np.asarray(q);f=np.asarray(e.sim.gaze_trace)
    limits=e.sim.model.jnt_range[e.sim.jids];violation=float(np.maximum(np.maximum(limits[:,0]-q,q-limits[:,1]),0).max())
    after=f[f[:,0]>=6+switch-1e-8];progress=float(after[-1,1]-after[0,1])
    torque=float(max(np.max(np.abs(row[1])) for row in e.sim.trace[-len(q):]))
    admitted=not m['fell'] and violation<=.02 and torque<=.6001 and m['physics']['joint_speed_max_rad_s']<=16.5
    if kind=='recovery':admitted=admitted and m['terrain_traversed'] and progress>=.4
    else:admitted=admitted and m['post_stop']['planar_speed_mean_mm_s']<=10 and m['braking_net_displacement_mm']<=80
    obs=np.asarray(recorder.obs,np.float32);actions=np.asarray(recorder.actions,np.float32);mask=np.asarray(recorder.teacher_mask)
    np.testing.assert_array_equal(obs[1:,44:63],actions[:-1])
    m.update(accepted=bool(admitted),joint_limit_violation_max_rad=violation,torque_peak_nm=torque,
        after_switch_progress_m=progress,switch_x=float(after[0,1]),after_switch_vx=float(after[:,13].mean()))
    arrays=dict(obs=obs,actions=actions,teacher_mask=mask,gaze=f,joints=q,trace=t,
        head_controls=np.asarray(e.head_rows),head_physics=np.asarray(e.head_physics))
    if kind=='recovery':arrays['expert_obs']=np.asarray(teacher.expert_obs)
    return m,arrays


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=('recovery','stop'),required=True);a=p.parse_args()
    dest=OUT/'collection'/a.kind;dest.mkdir(parents=True,exist_ok=False)
    jobs=[(d,s,(4.,6.,8.)[(s-31)%3]) for d in (5,10,15) for s in range(31,37)] if a.kind=='recovery' else [(d,s,17.) for d in (5,10,15) for s in (41,42,43)]
    (dest/'plan.json').write_text(json.dumps(dict(jobs=jobs,student_sha256=sha(STUDENT),
        kind=a.kind,holdout_seeds=[201,202,203],state_reset=False),indent=2))
    records=[];bank={k:[] for k in ('obs','actions','trial_id','frontier')}
    for i,(delay,seed,switch) in enumerate(jobs):
        m,arrays=run(a.kind,delay,seed,switch);name=f'lag{delay}_s{seed}_switch{switch:g}'
        record=dict(name=name,delay=delay,seed=seed,switch=switch,metrics=m)
        (dest/f'{name}.json').write_text(json.dumps(record,indent=2));np.savez_compressed(dest/f'{name}.npz',**arrays)
        records.append(record)
        if m['accepted']:
            mask=arrays['teacher_mask'];count=int(mask.sum())
            bank['obs'].extend(arrays['obs'][mask]);bank['actions'].extend(arrays['actions'][mask])
            bank['trial_id'].extend([i]*count);bank['frontier'].extend((np.arange(count)<75).tolist())
        print(json.dumps(dict(kind=a.kind,name=name,accepted=m['accepted'],fell=m['fell'],progress=m['after_switch_progress_m'],limit=m['joint_limit_violation_max_rad'])),flush=True)
        (dest/'matrix.json').write_text(json.dumps(records,indent=2))
    assert len(bank['obs'])>=300,'Too few physically accepted correction frames'
    dataset=OUT/f'{a.kind}_demonstrations.npz'
    np.savez_compressed(dataset,**{k:np.asarray(v,np.float32 if k in ('obs','actions') else np.int32) for k,v in bank.items()})
    (OUT/f'{a.kind}_dataset.json').write_text(json.dumps(dict(frames=len(bank['obs']),accepted=sum(r['metrics']['accepted'] for r in records),attempted=len(records),sha256=sha(dataset),
        meaning='Only physically executed teacher actions after state-preserving takeover; frontier is first 1.5 seconds.'),indent=2))


if __name__=='__main__':main()
