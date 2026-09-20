"""Timestamped outer-loop IMU transport and fixed-policy heading evaluation."""
from collections import deque
from dataclasses import dataclass,asdict
import numpy as np
from evaluate_gaze_ablation import GazeSim
from imu_heading import RelativeImuHeading,HeadingController,matrix,wrap
from mjlab_microduck.s288_protocol import JY61P_GYRO_STEP,JY61P_GYRO_LIMIT

WARMUP_SECONDS=6.


@dataclass(frozen=True)
class ImuTransportConfig:
    period_ms:int=20
    latency_ms:int=20
    gyro_noise_deg_s:float=.05
    accel_noise_m_s2:float=.02
    constant_z_bias_deg_s:float=0.
    residual_z_bias_deg_s:float=0.
    outage_start_s:float=5.
    outage_duration_s:float=0.


class ImuStream:
    """Only sensor buffers enter this adapter; no orientation or root state."""
    def __init__(self,sim,config,seed):
        self.config=config;self.rng=np.random.default_rng(30000+seed)
        self.slices=[]
        for name in ('imu_ang_vel','imu_accel','head_imu_ang_vel','head_imu_accel'):
            sensor=sim.model.sensor('robot/'+name).id
            start=sim.model.sensor_adr[sensor];assert sim.model.sensor_dim[sensor]==3
            self.slices.append(slice(start,start+3))
        if config.latency_ms<=0 or config.period_ms<20 or config.period_ms%20:raise ValueError('Use positive latency and a supported sensor period')
        self.queue=deque();self.next_sample=0.;self.latest=None;self.delivered=[];self.dropped=0

    def poll(self,timestamp,sensordata):
        c=self.config
        if timestamp+1e-9>=self.next_sample:
            values=np.array([sensordata[s].copy() for s in self.slices])
            self.next_sample+=c.period_ms/1000
            for i in (0,2):
                values[i]+=self.rng.normal(0,np.deg2rad(c.gyro_noise_deg_s),3)
                values[i,2]+=np.deg2rad(c.constant_z_bias_deg_s+(c.residual_z_bias_deg_s if timestamp>=WARMUP_SECONDS else 0.))
                values[i]=np.round(np.clip(values[i],-JY61P_GYRO_LIMIT,JY61P_GYRO_LIMIT)/JY61P_GYRO_STEP)*JY61P_GYRO_STEP
            for i in (1,3):values[i]+=self.rng.normal(0,c.accel_noise_m_s2,3)
            relative=timestamp-WARMUP_SECONDS
            if c.outage_start_s<=relative<c.outage_start_s+c.outage_duration_s:self.dropped+=1
            else:self.queue.append((timestamp+c.latency_ms/1000,timestamp,values))
        new=[]
        while self.queue and self.queue[0][0]<=timestamp+1e-9:
            _,source,values=self.queue.popleft();self.latest=(source,values);self.delivered.append(self.latest);new.append(self.latest)
        return new


class HeadingExperiment:
    def __init__(self,plant,policy,hardware,transport=None,controller=None,estimator=None):
        self.sim=GazeSim(plant,policy,hardware);self.transport=transport or ImuTransportConfig()
        self.controller_config=controller;self.estimator_config=estimator
        # Verify the fixed CAD mounting convention at nominal HOME once.
        for name in ('imu','head_imu'):
            r=self.sim.data.site_xmat[self.sim.model.site('robot/'+name).id].reshape(3,3)
            np.testing.assert_allclose(r,np.eye(3),atol=1e-5)
        camera_r=self.sim.data.site_xmat[self.sim.camera].reshape(3,3)
        np.testing.assert_allclose(-camera_r[:,2],[1,0,0],atol=1e-5)
        self.stream=None;self.body_est=None;self.head_est=None

    def poll(self):
        for timestamp,values in self.stream.poll(self.sim.data.time,self.sim.data.sensordata):
            if self.body_est.ready:
                self.body_est.update(values[0],values[1],timestamp)
                self.head_est.update(values[2],values[3],timestamp)

    def reset(self,seed=0):
        self.sim.reset(seed,seed>0);self.stream=ImuStream(self.sim,self.transport,seed)
        self.body_est=RelativeImuHeading(self.estimator_config);self.head_est=RelativeImuHeading(self.estimator_config)
        self.controller=HeadingController(self.controller_config)
        for _ in range(round(WARMUP_SECONDS/self.sim.dt)):self.poll();self.sim.step(np.zeros(18))
        self.poll()
        samples=[(t,v) for t,v in self.stream.delivered if t>=WARMUP_SECONDS-1.-1e-8]
        timestamps=[t for t,_ in samples];values=np.array([v for _,v in samples])
        self.body_est.calibrate(values[:,0],values[:,1],timestamps[-1])
        self.head_est.calibrate(values[:,2],values[:,3],timestamps[-1])
        self.controller.reset()
        return {'body_bias_deg_s':np.rad2deg(self.body_est.bias).tolist(),
            'head_bias_deg_s':np.rad2deg(self.head_est.bias).tolist(),'calibration_samples':len(samples),
            'calibration_end_timestamp':timestamps[-1]}

    def user_command(self,t,scenario):
        user=np.zeros(18)
        if scenario=='stand':pass
        else:user[0]=.55 if scenario=='straight' else .35
        if scenario=='left_then_hold' and 2<=t<4:user[2]=.45
        if scenario=='right_then_hold' and 2<=t<4:user[2]=-.45
        if scenario=='s_turn':user[2]=.45 if 2<=t<4 else (-.45 if 6<=t<8 else 0.)
        if scenario not in ('stand','straight','left_then_hold','right_then_hold','s_turn'):raise ValueError(scenario)
        return user

    def run(self,scenario,mode='imu',seed=0,seconds=12.,capture=False):
        if mode not in ('open','imu','oracle'):raise ValueError(mode)
        calibration=self.reset(seed);sim=self.sim;start=sim.view(sim.data.time)
        body_zero,head_zero=start[4],start[5];rows=[]
        # Scoring reads ground truth below; only explicitly labelled oracle mode
        # passes it to the controller. IMU mode supplies only sensor estimates.
        for tick in range(round(seconds/sim.dt)):
            t=tick*sim.dt;self.poll();user=self.user_command(t,scenario)
            heading=self.body_est.yaw
            if mode=='oracle':heading=wrap(sim.view(sim.data.time)[4]-body_zero)
            age=max(0.,sim.data.time-self.body_est.last_time)
            corrected=self.controller.update(heading,user[2],sim.dt,moving=abs(user[0])>.05,measurement_age=age)
            command=user.copy();command[2]=self.controller.feedforward if mode=='open' else corrected
            sim.step(command);view=sim.view(sim.data.time)
            rows.append(np.r_[t+sim.dt,self.controller.reference,view[4],view[5],self.body_est.yaw,self.head_est.yaw,
                command[2],self.controller.feedforward,self.controller.integral,age,self.controller.stale,
                self.controller.saturated,view[1:3],view[13]])
        a=np.array(rows);body=np.unwrap(a[:,2])-body_zero;head=np.unwrap(a[:,3])-body_zero
        # Sensor estimate corresponds to its source timestamp, not current truth.
        # Compare to interpolated truth at that source time to separate delay/error.
        body_at_sample=np.interp(a[:,0]-sim.dt-a[:,9],np.r_[0,a[:,0]],np.r_[0,body])
        head_relative=np.unwrap(a[:,3])-head_zero
        head_at_sample=np.interp(a[:,0]-sim.dt-a[:,9],np.r_[0,a[:,0]],np.r_[0,head_relative])
        count=round(seconds/sim.model.opt.timestep)
        full=np.vstack([sim.gaze_trace[-count:],sim.view(sim.data.time)])
        relative_time=full[:,0]-WARMUP_SECONDS
        reference=np.interp(relative_time,np.r_[0,a[:,0]],np.r_[0,a[:,1]])
        ff=np.interp(relative_time,np.r_[0,a[:,0]],np.r_[0,a[:,7]])
        camera_error=np.unwrap(full[:,5])-body_zero-reference
        camera_error-=2*np.pi*round(camera_error[0]/(2*np.pi))
        detrended=camera_error-np.polyval(np.polyfit(relative_time,camera_error,1),relative_time)
        rms=lambda x:float(np.sqrt(np.mean(np.square(x))))
        delta=a[-1,12:14]-start[1:3]
        fall=np.zeros((len(full),7));fall[:,:3]=full[:,1:4];fall[:,6]=full[:,14]
        metrics={'scenario':scenario,'mode':mode,'seed':seed,'seconds':seconds,'calibration':calibration,
            'heading_error_rms_deg':float(np.rad2deg(rms(body-a[:,1]))),
            'heading_error_final_deg':float(np.rad2deg(body[-1]-a[-1,1])),
            'heading_error_last2s_mean_deg':float(np.rad2deg(np.mean((body-a[:,1])[-100:]))),
            'body_estimate_error_rms_deg':float(np.rad2deg(rms(a[:,4]-body_at_sample))),
            'head_relative_estimate_error_rms_deg':float(np.rad2deg(rms(a[:,5]-head_at_sample))),
            'camera_heading_error_rms_deg':float(np.rad2deg(rms(camera_error))),
            'camera_heading_detrended_rms_deg':float(np.rad2deg(rms(detrended))),
            'camera_yaw_rate_error_rms_deg_s':float(np.rad2deg(rms(full[:,7]-ff))),
            'camera_relative_lateral_p90_range_mm':float(np.diff(np.percentile(full[:,8],[5,95]))[0]*1000),
            'body_vx_mean_m_s':float(np.mean(a[:,14])),
            'body_lateral_displacement_mm':float((-np.sin(body_zero)*delta[0]+np.cos(body_zero)*delta[1])*1000),
            'reference_final_deg':float(np.rad2deg(a[-1,1])),
            'command_wz_max_rad_s':float(np.max(np.abs(a[:,6]))),
            'command_wz_slew_max_rad_s2':float(np.max(abs(np.diff(np.r_[0,a[:,6]])))/sim.dt),
            'feedback_stale_fraction':float(np.mean(a[:,10])),
            'controller_limited_fraction':float(np.mean(a[:,11])) if mode!='open' else None,
            'sensor_age_max_ms':float(np.max(a[:,9])*1000),'sensor_packets_dropped':self.stream.dropped,
            'body_accel_rejection_fraction':self.body_est.accel_rejections/max(self.body_est.updates,1),
            'estimator_gap_count':self.body_est.gap_count,'fell':sim.fall_test(fall)}
        assert np.isfinite(a).all() and metrics['command_wz_max_rad_s']<=.700001
        assert metrics['command_wz_slew_max_rad_s2']<=1.50001
        return metrics,a if capture else None


TRACE_COLUMNS=['time','reference','body_yaw_world','camera_yaw_world','body_yaw_estimate',
    'head_relative_yaw_estimate','applied_wz','user_wz_shaped','integral','sensor_age','stale','limited',
    'base_x','base_y','body_vx']
