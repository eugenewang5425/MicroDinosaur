"""Follow-up: calibrate with stable v7 standing before switching locomotion policy.

Triggered by the frozen first matrix rejecting the unstable candidate's own
standing calibration. Never weaken the calibration motion threshold.
"""
import json
from pathlib import Path
import time
import onnxruntime as ort
import numpy as np
from heading_sim import HeadingExperiment,TRACE_COLUMNS
from hardware_sim import HardwareCase
from evaluate_policy import sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_imu_heading/stable_start'
CALIBRATOR=ROOT/'20260913_handoff/v7_reference.onnx'
POLICY=ROOT.parent/'policies/head_candidate.onnx'


class StableStartExperiment(HeadingExperiment):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.calibration_session=ort.InferenceSession(str(CALIBRATOR),providers=['CPUExecutionProvider'])

    def reset(self,seed=0):
        target=self.sim.session;self.sim.session=self.calibration_session
        try:record=super().reset(seed)
        finally:self.sim.session=target
        # Keep actual last action, filtered target and delay buffers at the
        # transition. Clearing them would create an artificial actuator jump.
        record['stationary_policy_sha256']=sha(CALIBRATOR)
        record['policy_switch_preserves_action_and_delay_history']=True
        return record


def main():
    OUT.mkdir(exist_ok=True);(OUT/'trials').mkdir(exist_ok=True);(OUT/'traces').mkdir(exist_ok=True)
    sources={p.name:sha(p) for p in [Path(__file__),ROOT/'imu_heading.py',ROOT/'heading_sim.py',ROOT/'hardware_sim.py',ROOT/'evaluate_policy.py',ROOT/'evaluate_gaze_ablation.py']}
    jobs=[{'seed':seed,'scenario':scenario,'mode':mode,'seconds':8 if scenario=='stand' else 12}
        for seed in (1,2,3) for scenario in ('stand','straight','left_then_hold','right_then_hold') for mode in ('open','imu')]
    plan={'phase':'followup_stable_start','reason':'Candidate standing failed strict calibration in phase A',
        'startup_policy':CALIBRATOR.relative_to(ROOT).as_posix(),'startup_policy_sha256':sha(CALIBRATOR),
        'locomotion_policy':POLICY.relative_to(ROOT.parent).as_posix(),'locomotion_policy_sha256':sha(POLICY),
        'source_hashes':sources,'jobs':jobs,'same_heading_parameters_as_phase_A':True,
        'preserve_action_state_on_switch':True,'hardware':{'physics_dt':.00125,'command_ms':10,'position_ms':20,'velocity_ms':20}}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    records=[];state={'status':'RUNNING','completed':0,'total':24,'started_unix':time.time()}
    def save(): (OUT/'status.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    save()
    try:
        for job in jobs:
            key=f's{job["seed"]}__{job["scenario"]}__{job["mode"]}'
            target=OUT/'trials'/f'{key}.json';assert not target.exists(),'Do not overwrite completed follow-up'
            sim=StableStartExperiment(ROOT/'20260913_handoff/native_v07',POLICY,HardwareCase(physics_dt=.00125))
            metrics,trace=sim.run(job['scenario'],job['mode'],job['seed'],job['seconds'],job['seed']==1)
            record={**job,'metrics':metrics,'status':'COMPLETE','source_hashes':sources}
            records.append(record);target.write_text(json.dumps(record,indent=2),encoding='utf-8')
            if trace is not None:np.savez_compressed(OUT/'traces'/f'{key}.npz',trace=trace,columns=np.array(TRACE_COLUMNS))
            state['completed']=len(records);save()
            print(json.dumps({'completed':len(records),'key':key,'heading_rms':metrics['heading_error_rms_deg'],'fell':metrics['fell']}),flush=True)
        (OUT/'matrix.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        state.update(status='COMPLETE',finished_unix=time.time());save()
    except Exception as exc:state.update(status='FAILED',error=str(exc));save();raise


if __name__=='__main__':main()
