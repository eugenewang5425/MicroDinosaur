"""Timed jump-expert -> frozen v7 return, with continuous physical/IMU history."""
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort

from evaluate_run_jump import MotionExperiment,score,render,OUT
from hardware_sim import HardwareCase
from imu_owned_head import configure_owned
from run_heading_stable_start import CALIBRATOR
from evaluate_policy import sha

JUMP=Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_jump_specialist/20260914_train_512x401/candidate.onnx')
RETURN_TIME=2.8


class ReturningJump(MotionExperiment):
    def __init__(self,case):
        super().__init__(JUMP,'jump',.6,case)
        self.jump_session=self.sim.session
        self.return_session=ort.InferenceSession(str(CALIBRATOR),providers=['CPUExecutionProvider'])
        self.returned=False;self.handoff=None

    def reset(self,seed=0):
        self.sim.session=self.jump_session;self.returned=False;self.handoff=None
        return super().reset(seed)

    def fingerprint(self):
        arrays=[self.sim.raw,self.sim.applied,self.sim.data.qpos,self.sim.data.qvel,
                np.asarray(self.sim.command_history),np.asarray(self.sim.obs_history),
                self.head_controller.correction,np.array([self.controller.reference,self.controller.integral])]
        return hashlib.sha256(b''.join(np.asarray(a).tobytes() for a in arrays)).hexdigest()

    def user_command(self,t,scenario):
        command=super().user_command(t,scenario)
        if t>=RETURN_TIME-1e-9 and not self.returned:
            before=self.fingerprint();self.sim.session=self.return_session
            after=self.fingerprint();assert before==after
            self.returned=True
            self.handoff=dict(time_s=t,history_before_sha256=before,history_after_sha256=after,
                sensor_or_physical_state_reset=False,return_policy_sha256=sha(CALIBRATOR))
        return command


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fresh',action='store_true');args=parser.parse_args()
    parent=OUT/'evaluation/jump';dest=OUT/'evaluation'/('jump_return_fresh' if args.fresh else 'jump_return')
    dest.mkdir(parents=True,exist_ok=False)
    cases=json.loads((parent/'matrix.json').read_text());rows=[]
    if args.fresh:
        cases=[{**r,'seed':seed,'name':r['name'].replace('_s101_','_s'+str(seed)+'_'),'status':'NO_PREFIX_REFERENCE'}
               for r in cases if r['seed']==101 for seed in [201,202,203]]
    for source in cases:
        case=HardwareCase(**source['hardware']);seed=source['seed'];name=source['name']
        e=configure_owned(ReturningJump(case),(1,))
        r={k:source[k] for k in ['name','skill','speed','seed','hardware','policy_sha256']}
        r.update(return_time_s=RETURN_TIME,return_policy=str(CALIBRATOR),return_policy_sha256=sha(CALIBRATOR))
        try:
            m,trace=e.run('stand','imu',seed,6.,True);score(e,m)
            if source['status']=='COMPLETE':
                original=np.load(parent/(name+'.npz'))['physics'];actual=np.asarray(e.rows)
                prefix=original[:,0]<RETURN_TIME-1e-7
                error=float(np.max(np.abs(original[prefix]-actual[prefix])))
                assert error<1e-9, ('pre-return trajectory changed',error)
                r['prefix_max_error']=error
            r.update(status='COMPLETE',metrics=m,handoff=e.handoff)
            np.savez_compressed(dest/(name+'.npz'),physics=np.asarray(e.rows),qpos=np.asarray(e.qpos_frames),
                gaze=np.asarray(e.sim.gaze_trace),head_controls=np.asarray(e.head_rows),heading=trace)
            if seed==101 and case.command_ms==10:
                try:render(e,dest/'jump_return.mp4');r['video_status']='COMPLETE'
                except Exception as exc:r.update(video_status='FAILED',video_error=repr(exc))
            print(json.dumps(dict(name=name,flight=m['max_clear_flight_s'],jump=m['jump_success'],
                final_speed=m['final_second_speed_m_s'],failed=m['motion_failed'])),flush=True)
        except Exception as exc:
            r.update(status='FAILED',error=repr(exc));print(json.dumps(r),flush=True)
        rows.append(r)
        (dest/(name+'.json')).write_text(json.dumps(r,indent=2),encoding='utf-8')
        (dest/'matrix.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':main()
