"""A physically executed old gait with explicit command calibration for teaching."""
from pathlib import Path
import numpy as np
from terrain_skill_eval import TerrainSkillExperiment
from hardware_sim import HardwareCase
from imu_owned_head import configure_owned

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_demonstrations'
SOURCE=Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_transition_refine/20260914_train_512x201/model_16150.pt')
POLICY=SOURCE.with_name('candidate.onnx')


class CalibratedExpert:
    def __init__(self,session,drive,hip_limit=None):
        self.session=session;self.drive=drive
        self.hip_limit=hip_limit
        self.obs=[];self.expert_obs=[];self.actions=[]
    def get_inputs(self):return self.session.get_inputs()
    def get_outputs(self):return self.session.get_outputs()
    def run(self,outputs,inputs):
        student=inputs['obs'].copy();expert=student.copy()
        # The desired physical speed is .2m/s. The old gait under-tracks its
        # command; explicitly calibrate its internal command, never its physics.
        if self.drive is not None and expert[0,63]>.05:expert[0,63]=self.drive
        result=self.session.run(outputs,{'obs':expert})
        if self.hip_limit is not None:
            ids,home,scale,lower,upper=self.hip_limit
            raw=result[0].copy();q=home+raw[:,ids]*scale
            raw[:,ids]=(np.clip(q,lower,upper)-home)/scale;result=[raw]
        self.obs.append(student[0]);self.expert_obs.append(expert[0]);self.actions.append(result[0][0].copy())
        return result


def experiment(drive,delay=10,hip_margin=None):
    e=configure_owned(TerrainSkillExperiment(POLICY,terrain='steps_10',walking_speed=.2,
        hardware_case=HardwareCase(physics_dt=.00125,command_ms=delay)),(1,))
    hip_limit=None
    if hip_margin is not None:
        ids=np.asarray([e.sim.names.index(n) for n in ('left_hip_yaw','right_hip_yaw')])
        limits=e.sim.model.jnt_range[np.asarray(e.sim.jids)[ids]]
        scale=np.broadcast_to(e.sim.scale,(19,))[ids]
        hip_limit=(ids,e.sim.home[ids],scale,limits[:,0]+hip_margin,limits[:,1]-hip_margin)
    recorder=CalibratedExpert(e.sim.session,drive,hip_limit);e.sim.session=recorder
    before=e.sim.substep_callback;e.joint_rows=[];e.foot_rows=[]
    def capture():
        before()
        if e.active:
            e.joint_rows.append(e.sim.data.qpos[e.sim.jadr].copy())
            e.foot_rows.append(e.sim.data.geom_xpos[e.sim.foot_geoms].copy())
    e.sim.substep_callback=capture
    return e,recorder


def run(drive,delay,seed,hip_margin=None):
    e,recorder=experiment(drive,delay,hip_margin);m,t=e.run('straight','imu',seed,12.,True)
    q=np.asarray(e.joint_rows);limits=e.sim.model.jnt_range[e.sim.jids]
    violation=float(np.maximum(np.maximum(limits[:,0]-q,q-limits[:,1]),0).max())
    m.update(joint_limit_violation_max_rad=violation,
        torque_peak_nm=float(np.max(np.abs([r[1] for r in e.sim.trace[-9600:]]))),
        desired_speed_m_s=.2,expert_internal_command_m_s=drive,
        expert_hip_target_margin_rad=hip_margin,
        demonstration_accepted=bool(not m['fell'] and m['forward_displacement_m']>=.8
            and .15<=m['body_vx_mean_m_s']<=.25 and violation<=.02
            and m['physics']['joint_speed_max_rad_s']<=16.5))
    assert m['torque_peak_nm']<=.6001
    arrays=dict(obs=np.asarray(recorder.obs,np.float32),expert_obs=np.asarray(recorder.expert_obs,np.float32),
        actions=np.asarray(recorder.actions,np.float32),trace=t,gaze=np.asarray(e.sim.gaze_trace),
        joints=q,feet=np.asarray(e.foot_rows),head_controls=np.asarray(e.head_rows),head_physics=np.asarray(e.head_physics))
    assert arrays['obs'].shape==(600,81) and arrays['actions'].shape==(600,19)
    assert np.allclose(arrays['obs'][:,63],.2) and np.all(arrays['obs'][:,72]==0)
    assert all(np.isfinite(v).all() for v in arrays.values())
    return m,arrays
