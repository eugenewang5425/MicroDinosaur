"""Teacher observations/actions captured without altering inference or delays."""
import json
import numpy as np
from refine_reference_audit import OUT, V7
from terrain_skill_eval import TerrainSkillExperiment
from evaluate_policy import sha


class Recorder:
    def __init__(self, session): self.session=session; self.obs=[]; self.actions=[]
    def get_inputs(self): return self.session.get_inputs()
    def get_outputs(self): return self.session.get_outputs()
    def run(self, outputs, inputs):
        result = self.session.run(outputs, inputs)
        self.obs.append(inputs['obs'].copy()[0]); self.actions.append(result[0].copy()[0])
        return result


def main():
    observations=[]; actions=[]; trials=[]
    for terrain, scenario in [('flat',s) for s in ('stand','straight','left_then_hold','right_then_hold')]+[('steps_10','straight')]:
        for seed in (1,2,3):
            e=TerrainSkillExperiment(V7, terrain=terrain)
            recorder=Recorder(e.sim.session);e.sim.session=recorder
            metrics,_=e.run(scenario,'imu',seed,12.,True)
            assert not metrics['fell'] and len(recorder.obs)==600
            observations.extend(recorder.obs);actions.extend(recorder.actions)
            trials.append(dict(terrain=terrain,scenario=scenario,seed=seed,metrics=metrics))
            print(terrain,scenario,seed,len(observations),flush=True)
    path=OUT/'v7_anchor.npz'
    assert not path.exists()
    np.savez_compressed(path,obs=np.asarray(observations,np.float32),actions=np.asarray(actions,np.float32))
    (OUT/'v7_anchor.json').write_text(json.dumps(dict(policy_sha256=sha(V7),dataset_sha256=sha(path),
        observations=len(observations),trials=trials),indent=2),encoding='utf-8')


if __name__=='__main__':main()
