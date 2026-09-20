"""Post-hoc diagnostic: can the verified expert recover student-visited states?"""
import json
import numpy as np
from demonstration_expert import OUT,experiment as make_expert
from evaluate_owned_terrain import cases,key,experiment,evaluate
from evaluate_policy import sha
from pathlib import Path

STUDENT=Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_demo_demonstration/20260914_train_512x101/candidate.onnx')


class Switch:
    def __init__(self,student,expert):self.student=student;self.expert=expert;self.ticks=0;self.obs=[];self.actions=[]
    def get_inputs(self):return self.student.get_inputs()
    def get_outputs(self):return self.student.get_outputs()
    def run(self,outputs,inputs):
        result=(self.student if self.ticks<300 else self.expert).run(outputs,inputs)
        self.ticks+=1;self.obs.append(inputs['obs'][0].copy());self.actions.append(result[0][0].copy())
        return result


def main():
    folder=OUT/'recovery_probe';folder.mkdir(exist_ok=False)
    (folder/'plan.json').write_text(json.dumps(dict(post_hoc=True,reason='New demonstration student still fails low-speed stairs; diagnose teacher ability on student-visited states.',
        seeds=[1,2,3],switch_time_s=6,seconds=12,desired_speed=.2,policy_sha256=sha(STUDENT),
        state_reset=False,expert_parameters='Already selected internal .50, hip margin .03; no tuning.'),indent=2))
    case=next(c for c in cases() if c['terrain']=='steps_10' and c['speed']==.2);records=[]
    for seed in (1,2,3):
        e=experiment(STUDENT,case);unused,expert=make_expert(.5,10,.03);del unused
        switch=Switch(e.sim.session,expert);e.sim.session=switch
        q=[];before=e.sim.substep_callback
        def capture():
            before()
            if e.active:q.append(e.sim.data.qpos[e.sim.jadr].copy())
        e.sim.substep_callback=capture;m,t=evaluate(e,case,seed);f=np.asarray(e.sim.gaze_trace);q=np.asarray(q)
        with np.load(OUT/f'evaluation/demonstration/{key(case,seed)}.npz') as z:
            baseline=z['gaze'];prefix=f[:,0]<12.-1e-8
            np.testing.assert_array_equal(f[prefix],baseline[:int(prefix.sum())])
        ranges=e.sim.model.jnt_range[e.sim.jids];violation=float(np.maximum(np.maximum(ranges[:,0]-q,q-ranges[:,1]),0).max())
        m['joint_limit_violation_max_rad']=violation
        r=dict(seed=seed,metrics=m,first6seconds_identical=True,post_hoc=True);records.append(r)
        (folder/f's{seed}.json').write_text(json.dumps(r,indent=2))
        np.savez_compressed(folder/f's{seed}.npz',gaze=f,joints=q,trace=t,obs=np.asarray(switch.obs),actions=np.asarray(switch.actions),
            expert_obs=np.asarray(expert.expert_obs),expert_actions=np.asarray(expert.actions))
        print(json.dumps(dict(seed=seed,fell=m['fell'],traversed=m['terrain_traversed'],distance=m['forward_displacement_m'],joint_limit=violation)),flush=True)
    (folder/'matrix.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
