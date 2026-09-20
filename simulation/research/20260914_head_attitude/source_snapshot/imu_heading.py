"""Sensor-only relative attitude estimation and a bounded heading command loop.

No MuJoCo/state access. Quaternions are wxyz, mapping sensor-aligned body axes
to a local Z-up frame. Heading is relative to the stationary calibration pose.
This small complementary filter is not a reproduction of JY61P firmware AHRS.
"""
from dataclasses import dataclass
import numpy as np


def wrap(angle):return float((angle+np.pi)%(2*np.pi)-np.pi)


def multiply(q,p):
    w,x,y,z=q;a,b,c,d=p
    return np.array([w*a-x*b-y*c-z*d,w*b+x*a+y*d-z*c,w*c-x*d+y*a+z*b,w*d+x*c-y*b+z*a])


def matrix(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
        [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
        [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def from_rpy(roll,pitch,yaw=0.):
    cr,sr=np.cos(roll/2),np.sin(roll/2);cp,sp=np.cos(pitch/2),np.sin(pitch/2)
    cy,sy=np.cos(yaw/2),np.sin(yaw/2)
    return np.array([cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy])


@dataclass(frozen=True)
class EstimatorConfig:
    tilt_gain:float=.3
    accel_filter_tau:float=.10
    accel_norm_tolerance:float=.15
    accel_angle_limit_deg:float=20.
    max_sample_gap_s:float=.15


class RelativeImuHeading:
    def __init__(self,config=None):self.config=config or EstimatorConfig();self.ready=False

    def calibrate(self,gyro_samples,accel_samples,timestamp):
        gyro=np.asarray(gyro_samples,float);accel=np.asarray(accel_samples,float)
        if gyro.ndim!=2 or gyro.shape[1]!=3 or gyro.shape!=accel.shape or len(gyro)<20:
            raise ValueError('Stationary calibration requires >=20 matching 3-axis samples')
        if not np.isfinite(gyro).all() or not np.isfinite(accel).all():raise ValueError('Nonfinite calibration')
        if np.max(np.std(gyro,axis=0))>np.deg2rad(.5):raise ValueError('Calibration angular motion is excessive')
        self.bias=gyro.mean(0);self.accel_filtered=accel.mean(0)
        if not .8*9.81<np.linalg.norm(self.accel_filtered)<1.2*9.81:raise ValueError('Calibration acceleration is not near 1 g')
        ax,ay,az=self.accel_filtered
        self.q=from_rpy(np.arctan2(ay,az),np.arctan2(-ax,np.hypot(ay,az)))
        self.last_time=float(timestamp);self.yaw=0.;self.previous_yaw=0.;self.ready=True
        self.updates=0;self.accel_rejections=0;self.gap_count=0

    def update(self,gyro,accel,timestamp):
        if not self.ready:raise RuntimeError('Calibrate first')
        gyro=np.asarray(gyro,float);accel=np.asarray(accel,float)
        if gyro.shape!=(3,) or accel.shape!=(3,) or not np.isfinite(np.r_[gyro,accel,timestamp]).all():
            raise ValueError('Invalid IMU packet')
        dt=float(timestamp)-self.last_time
        if dt<0:raise ValueError('IMU timestamps must be monotonic')
        if dt==0:return self.yaw  # Held packet is not integrated twice.
        if dt>self.config.max_sample_gap_s:
            self.gap_count+=1;self.last_time=float(timestamp)
            # Motion during a long outage is unobserved, never invent it.
            return self.yaw
        self.last_time=float(timestamp)
        self.accel_filtered += (1-np.exp(-dt/self.config.accel_filter_tau))*(accel-self.accel_filtered)
        up=matrix(self.q)[2,:]
        length=np.linalg.norm(self.accel_filtered);measured_up=self.accel_filtered/max(length,1e-9)
        angle=np.arccos(np.clip(measured_up@up,-1,1))
        use_accel=(abs(np.linalg.norm(accel)/9.81-1)<=self.config.accel_norm_tolerance and
                   angle<=np.deg2rad(self.config.accel_angle_limit_deg))
        correction=self.config.tilt_gain*np.cross(measured_up,up) if use_accel else np.zeros(3)
        self.accel_rejections+=int(not use_accel)
        rotation=(gyro-self.bias+correction)*dt;angle=np.linalg.norm(rotation)
        dq=np.r_[np.cos(angle/2),rotation*(np.sin(angle/2)/angle if angle>1e-12 else .5)]
        self.q=multiply(self.q,dq);self.q/=np.linalg.norm(self.q)
        r=matrix(self.q);yaw=np.arctan2(r[1,0],r[0,0])
        self.yaw+=wrap(yaw-self.previous_yaw);self.previous_yaw=yaw;self.updates+=1
        return self.yaw


@dataclass(frozen=True)
class HeadingConfig:
    kp:float=1.2
    ki:float=.15
    heading_filter_tau:float=.15
    deadband_rad:float=float(np.deg2rad(.5))
    integral_limit_rad_s:float=.25
    max_yaw_rate_rad_s:float=.7
    yaw_accel_limit_rad_s2:float=1.5
    max_measurement_age_s:float=.12


class HeadingController:
    def __init__(self,config=None):self.config=config or HeadingConfig();self.reset()

    def reset(self,heading=0.):
        self.reference=float(heading);self.filtered_heading=float(heading)
        self.feedforward=0.;self.integral=0.;self.output=0.;self.feedback=0.;self.error=0.
        self.saturated=False;self.stale=False

    def update(self,heading,user_yaw_rate,dt,moving=True,measurement_age=0.):
        if not np.isfinite([heading,user_yaw_rate,dt,measurement_age]).all() or dt<=0 or measurement_age<0:
            raise ValueError('Invalid heading update')
        c=self.config;delta=c.yaw_accel_limit_rad_s2*dt
        requested=float(np.clip(user_yaw_rate,-c.max_yaw_rate_rad_s,c.max_yaw_rate_rad_s))
        self.feedforward+=float(np.clip(requested-self.feedforward,-delta,delta))
        # Integrate USER turn intent, never the corrected output command.
        self.reference+=self.feedforward*dt
        self.filtered_heading+=(1-np.exp(-dt/c.heading_filter_tau))*wrap(heading-self.filtered_heading)
        self.error=wrap(self.reference-self.filtered_heading)
        self.stale=measurement_age>c.max_measurement_age_s
        enabled=not self.stale and (moving or abs(self.feedforward)>.01)
        error=np.sign(self.error)*max(0.,abs(self.error)-c.deadband_rad) if enabled else 0.
        integral=float(np.clip(self.integral+c.ki*error*dt,-c.integral_limit_rad_s,c.integral_limit_rad_s)) if enabled else 0.
        raw=self.feedforward+c.kp*error+integral
        bounded=float(np.clip(raw,-c.max_yaw_rate_rad_s,c.max_yaw_rate_rad_s))
        output=self.output+float(np.clip(bounded-self.output,-delta,delta))
        self.saturated=abs(raw-output)>1e-10
        # Freeze integration if the error would push farther into either limit.
        if not (self.saturated and error*(raw-output)>0):self.integral=integral
        if not enabled:self.integral=0.
        self.output=output;self.feedback=output-self.feedforward
        return output
