"""Paired camera-axis and gait diagnostics, with finite positive hardware delays."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from evaluate_policy import sha
from hardware_sim import HardwareCase,HardwareSim

ROOT=Path(__file__).resolve().parent
CASES=[
    HardwareCase(name='nominal_1p25ms',physics_dt=.00125),
    HardwareCase(name='nominal_5ms',physics_dt=.005),
    HardwareCase(name='nominal_0p625ms',physics_dt=.000625),
    HardwareCase(name='delay_15_40_20',physics_dt=.00125,command_ms=15,position_ms=40),
]
COLUMNS=['time','base_x','base_y','base_z','base_yaw','camera_yaw','camera_pitch',
    'camera_yaw_rate','camera_relative_y','left_contact','right_contact',
    'left_foot_z','right_foot_z','base_vx','base_tilt_deg']


def optical_angles_and_yaw_rate(forward,omega_world):
    horizontal_sq=float(forward[0]**2+forward[1]**2)
    if horizontal_sq<1e-6:raise ValueError('Camera optical axis is vertical; yaw is undefined')
    change=np.cross(omega_world,forward)
    return np.array([np.arctan2(forward[1],forward[0]),
        np.arctan2(forward[2],np.sqrt(horizontal_sq)),
        (forward[0]*change[1]-forward[1]*change[0])/horizontal_sq])


def complete_stance_durations(contact,dt):
    # Exclude runs truncated by the beginning/end of the measurement window.
    transitions=np.diff(np.asarray(contact,dtype=int))
    starts=list(np.where(transitions==1)[0]+1);ends=list(np.where(transitions==-1)[0]+1)
    values=[]
    for start in starts:
        end=next((end for end in ends if end>start),None)
        if end is not None:values.append((end-start)*dt)
    return values


class GazeSim(HardwareSim):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.camera=self.model.site('robot/head_camera').id
        assert self.model.site_bodyid[self.camera]==self.head
        forward=-self.data.site_xmat[self.camera].reshape(3,3)[:,2]
        self.camera_home_angles=optical_angles_and_yaw_rate(forward,np.zeros(3))[:2]

    def reset(self,*args,**kwargs):
        super().reset(*args,**kwargs);self.gaze_trace=[]

    def view(self,sample_time):
        d,m=self.data,self.model
        rotation=d.xmat[self.body].reshape(3,3)
        forward=-d.site_xmat[self.camera].reshape(3,3)[:,2]
        head_velocity=np.zeros(6)
        mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,self.head,head_velocity,0)
        yaw,pitch,yaw_rate=optical_angles_and_yaw_rate(forward,head_velocity[:3])
        relative_y=(rotation.T@(d.site_xpos[self.camera]-d.xpos[self.body]))[1]
        contact=np.zeros(2);force=np.zeros(6)
        for index,c in enumerate(d.contact):
            if c.efc_address<0:continue
            if not any(g in (c.geom1,c.geom2) for g in self.foot_geoms):continue
            mujoco.mj_contactForce(m,d,index,force)
            if force[0]<=.05:continue
            for i,g in enumerate(self.foot_geoms):
                if g in (c.geom1,c.geom2):contact[i]=1
        return np.r_[sample_time,d.xpos[self.body],np.arctan2(rotation[1,0],rotation[0,0]),
            yaw,pitch,yaw_rate,relative_y,contact,d.xpos[self.feet,2],
            (rotation.T@d.qvel[:3])[0],np.rad2deg(np.arccos(np.clip(rotation[2,2],-1,1)))]

    def capture_physics(self):
        super().capture_physics()
        # mj_step leaves kinematic transforms from the beginning of the step.
        # Timestamp those transforms accordingly, without extra physics calls.
        self.gaze_trace.append(self.view(self.data.time-self.model.opt.timestep))


def run_gaze_trial(sim,command,seed,seconds):
    sim.reset(seed,seed>0)
    initial_heading=np.arctan2(sim.data.xmat[sim.body].reshape(3,3)[1,0],
                               sim.data.xmat[sim.body].reshape(3,3)[0,0])
    for _ in range(round(2/sim.dt)):sim.step(command)
    for _ in range(round(seconds/sim.dt)):sim.step(command)
    count=round(seconds/sim.model.opt.timestep)
    a=np.vstack([sim.gaze_trace[-count:],sim.view(sim.data.time)])
    assert a.shape==(count+1,len(COLUMNS)) and np.isfinite(a).all()
    expected=initial_heading+sim.camera_home_angles[0]+command[2]*a[:,0]
    error=np.unwrap(a[:,5])-expected
    # Select one continuous 2pi branch; preserve all subsequent heading drift.
    error-=2*np.pi*round(float(error[0]/(2*np.pi)))
    detrended=error-np.polyval(np.polyfit(a[:,0]-a[0,0],error,1),a[:,0]-a[0,0])
    yaw=np.unwrap(a[:,4]);heading=yaw[0]
    delta=a[-1,1:3]-a[0,1:3]
    contact=a[:-1,9:11]
    durations=[complete_stance_durations(contact[:,i],sim.model.opt.timestep) for i in range(2)]
    stance=[float(np.mean(d)) if d else None for d in durations]
    fall_samples=np.zeros((len(a),7));fall_samples[:,:3]=a[:,1:4];fall_samples[:,6]=a[:,14]
    rms=lambda x:float(np.sqrt(np.mean(np.square(x))))
    report={'seed':seed,'seconds':seconds,'command':command.tolist(),
        'camera_yaw_error_rms_deg':float(np.rad2deg(rms(error))),
        'camera_yaw_error_mean_deg':float(np.rad2deg(error.mean())),
        'camera_yaw_error_p95_deg':float(np.rad2deg(np.percentile(np.abs(error),95))),
        'camera_yaw_detrended_rms_deg':float(np.rad2deg(rms(detrended))),
        'camera_yaw_rate_error_rms_deg_s':float(np.rad2deg(rms(a[:,7]-command[2]))),
        'camera_pitch_error_rms_deg':float(np.rad2deg(rms(a[:,6]-sim.camera_home_angles[1]))),
        'camera_relative_lateral_p90_range_mm':float((np.percentile(a[:,8],95)-np.percentile(a[:,8],5))*1000),
        'body_yaw_drift_deg':float(np.rad2deg(yaw[-1]-yaw[0])),
        'body_yaw_rate_mean_rad_s':float((yaw[-1]-yaw[0])/seconds),
        'body_vx_mean_m_s':float(a[:,13].mean()),
        'body_lateral_displacement_mm':float((-np.sin(heading)*delta[0]+np.cos(heading)*delta[1])*1000),
        'foot_contact_fraction_LR':contact.mean(0).tolist(),
        'contact_duty_diff_pp':float(100*(contact[:,0].mean()-contact[:,1].mean())),
        'complete_stance_duration_mean_LR_s':stance,
        'complete_stance_count_LR':[len(d) for d in durations],
        'foot_vertical_range_LR_mm':(np.ptp(a[:,11:13],axis=0)*1000).tolist(),
        'fell':sim.fall_test(fall_samples),'physics':sim.physics_metrics(seconds)}
    return report,a


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',action='append',required=True)
    p.add_argument('--out',required=True);p.add_argument('--cases',default='all')
    p.add_argument('--seeds',type=int,default=3);p.add_argument('--resume',action='store_true')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    selected=[c for c in CASES if args.cases=='all' or c.name in args.cases.split(',')]
    assert selected
    model=ROOT/'20260913_handoff/native_v07'
    hashes={p.name:sha(p) for p in [Path(__file__),ROOT/'hardware_sim.py',ROOT/'evaluate_policy.py']}
    records=[];started=time.time()
    for spec in args.policy:
        label,onnx=spec.split('=',1)
        for case in selected:
            dest=out/f'{label}__{case.name}.json'
            if dest.exists():
                assert args.resume
                old=json.loads(dest.read_text());assert old['implementation_hashes']==hashes
                assert old['onnx_sha256']==sha(onnx) and old['case']==asdict(case)
                records.append(old);continue
            sim=GazeSim(model,onnx,case);trials=[]
            tasks=[('stand',0.,0.,5),('forward',.55,0.,6)]
            if case.name=='nominal_1p25ms':tasks += [('turn_left',0.,.45,4),('turn_right',0.,-.45,4)]
            for name,vx,wz,seconds in tasks:
                command=np.zeros(18);command[0]=vx;command[2]=wz
                for seed in range(args.seeds):
                    trial,trace=run_gaze_trial(sim,command,seed,seconds);trial['task']=name;trials.append(trial)
                    if case.name=='nominal_1p25ms' and seed==0:
                        np.savez_compressed(out/f'{label}__{name}.npz',trace=trace,columns=np.array(COLUMNS))
            record={'policy':label,'onnx_sha256':sha(onnx),'case':asdict(case),'plant_sha256':sha(model/'nominal.mjb'),
                'implementation_hashes':hashes,'trials':trials,'contact_force_threshold_N':.05,
                'camera_reference':'nominal optical forward + initial base heading + integrated commanded yaw rate',
                'camera_home_yaw_pitch_rad':sim.camera_home_angles.tolist(),'wall_seconds':time.time()-started}
            dest.write_text(json.dumps(record,indent=2),encoding='utf-8');records.append(record)
            print(json.dumps({'policy':label,'case':case.name,'trials':len(trials),
                'falls':sum(t['fell'] for t in trials),'elapsed_s':round(time.time()-started)}),flush=True)
    (out/'matrix.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':main()
