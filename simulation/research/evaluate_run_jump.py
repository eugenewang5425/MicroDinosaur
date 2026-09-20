"""Independent CPU physics: true flight, stable landing, speed and head metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco

from run_jump_cfg import ROOT,OUT,SOURCE
from head_attitude_sim import HeadExperiment
from head_attitude import HeadConfig
from imu_owned_head import configure_owned
from hardware_sim import HardwareCase
from heading_sim import WARMUP_SECONDS
from evaluate_policy import sha


class MotionExperiment(HeadExperiment):
    def __init__(self,policy,skill,speed=.6,case=None,plant=None,operating_envelope=False):
        self.skill=skill;self.speed=speed
        super().__init__(plant or OUT/'plant',policy,case or HardwareCase(physics_dt=.00125),
                         head_config=HeadConfig(max_measurement_age_s=.04))
        if operating_envelope:
            previous_target=self.sim.transform_target
            def bounded(target,obs):
                target=target.copy()
                for name in ('left_ankle','right_ankle'):
                    index=self.sim.names.index(name)
                    target[index]=np.clip(target[index],-np.deg2rad(51),np.deg2rad(51))
                target[self.sim.names.index('jaw_hinge')]=.04
                return previous_target(target,obs)
            self.sim.transform_target=bounded
        self.rows=[];self.qpos_frames=[];self.shaped=0.
        self.floor=self.sim.model.geom('terrain').id
        self.vertices=[]
        for gid in self.sim.foot_geoms:
            mesh=self.sim.model.geom_dataid[gid]
            a=self.sim.model.mesh_vertadr[mesh];n=self.sim.model.mesh_vertnum[mesh]
            self.vertices.append(self.sim.model.mesh_vert[a:a+n].copy())
        previous=self.sim.substep_callback
        def capture():
            previous()
            if not self.active:return
            d=self.sim.data;m=self.sim.model;force=np.zeros(6);total=0.;feet=[0.,0.];nonfoot=0.
            for k,c in enumerate(d.contact):
                if self.floor not in (c.geom1,c.geom2):continue
                mujoco.mj_contactForce(m,d,k,force);normal=max(0.,force[0]);total+=normal
                other=c.geom2 if c.geom1==self.floor else c.geom1
                if other in self.sim.foot_geoms:feet[self.sim.foot_geoms.index(other)]+=normal
                elif m.geom(other).name.endswith('_floor_proxy'):nonfoot+=normal
            minimum=[]
            for gid,v in zip(self.sim.foot_geoms,self.vertices):
                minimum.append(float(np.min(v@d.geom_xmat[gid].reshape(3,3)[2])+d.geom_xpos[gid,2]))
            rot=d.xmat[self.sim.body].reshape(3,3)
            vel=rot.T@d.qvel[:3]
            tilt=np.degrees(np.arccos(np.clip(rot[2,2],-1,1)))
            q=d.qpos[self.sim.jadr];limits=m.jnt_range[self.sim.jids]
            excess=np.maximum(limits[:,0]-q,q-limits[:,1]).clip(min=0).max()
            self.rows.append([d.time-WARMUP_SECONDS,d.qpos[2],d.qvel[2],d.subtree_com[self.sim.body,2],
                total,*feet,nonfoot,tilt,*vel[:2],max(abs(d.actuator_force[self.sim.aids])),
                max(abs(d.qvel[self.sim.vadr])),excess,*minimum,
                max([-c.dist for c in d.contact if self.floor in (c.geom1,c.geom2)],default=0.),
                np.linalg.norm(d.qvel[3:6]),
                np.max(np.abs(q[[self.sim.names.index('left_ankle'),self.sim.names.index('right_ankle')]])),
                q[self.sim.names.index('head_pitch')]])
            if len(self.rows)%32==0:self.qpos_frames.append(d.qpos.copy())
        self.sim.substep_callback=capture

    def reset(self,seed=0):
        self.rows=[];self.qpos_frames=[];self.shaped=0.
        return super().reset(seed)

    def user_command(self,t,scenario):
        user=np.zeros(18)
        if self.skill=='run':
            target=self.speed if 1.<=t<7. else 0.
            self.shaped+=float(np.clip(target-self.shaped,-.03,.03));user[0]=self.shaped
        elif self.skill=='jump':user[9]=-.03 if t<1.5 else (.02 if t<2.2 else 0.)
        return user


def flight_segments(a,dt):
    # No robot-ground supporting force AND both complete foot meshes >2mm.
    raw_air=a[:,4]<.05
    mask=raw_air&(np.min(a[:,14:16],axis=1)>.002)&(a[:,8]<30)
    com_velocity=np.gradient(a[:,3],dt)
    changes=np.diff(np.r_[False,mask,False].astype(int));runs=[]
    for start,end in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
        window=a[max(0,start-round(.08/dt)):start+1]
        up=float(np.max(window[:,2]))
        raw_start=start
        while raw_start>0 and raw_air[raw_start-1]:raw_start-=1
        raw_end=end
        while raw_end<len(a) and raw_air[raw_end]:raw_end+=1
        runs.append(dict(start_s=float(a[start,0]),end_s=float(a[end-1,0]+dt),
                         duration_s=(end-start)*dt,takeoff_up_velocity_m_s=up,
                         takeoff_com_velocity_m_s=float(com_velocity[raw_start]),
                         support_free_start_s=float(a[raw_start,0]),support_free_duration_s=(raw_end-raw_start)*dt,
                         com_apex_above_takeoff_m=float(np.max(a[raw_start:raw_end,3])-a[raw_start,3]),
                         minimum_foot_clearance_m=float(np.min(a[start:end,14:16]))))
    return runs


def score(e,m):
    a=np.asarray(e.rows);dt=e.sim.model.opt.timestep
    segments=flight_segments(a,dt)
    valid=[s for s in segments if s['duration_s']>=.04 and s['takeoff_com_velocity_m_s']>=.1]
    late=a[a[:,0]>=a[-1,0]-1.]
    stable=bool(np.mean(np.hypot(late[:,9],late[:,10]))<.03 and np.max(late[:,8])<15
                and np.mean((late[:,5]>.1)&(late[:,6]>.1))>.9)
    body_hit=bool(np.any(a[:,7]>.1))
    failed=bool(m['fell'] or body_hit or np.max(a[:,13])>.02)
    moving=a[(a[:,0]>=2.)&(a[:,0]<7.)]
    metrics=dict(flight_segments=segments,qualified_flight_segments=valid,
        max_clear_flight_s=max([s['duration_s'] for s in segments],default=0.),
        max_upward_velocity_m_s=float(np.max(a[:,2])),max_com_rise_m=float(np.max(a[:,3])-a[0,3]),
        min_root_height_m=float(np.min(a[:,1])),peak_tilt_deg=float(np.max(a[:,8])),
        nonfoot_ground_contact=body_hit,peak_ground_force_n=float(np.max(a[:,4])),
        peak_joint_torque_nm=float(np.max(a[:,11])),peak_joint_speed_rad_s=float(np.max(a[:,12])),
        joint_limit_excess_rad=float(np.max(a[:,13])),settled_final_second=stable,
        final_second_speed_m_s=float(np.mean(np.hypot(late[:,9],late[:,10]))),
        moving_speed_m_s=float(np.mean(moving[:,9])),motion_failed=failed,
        jump_success=bool(e.skill=='jump' and valid and stable and not failed),
        running_with_flight=bool(e.skill=='run' and valid and np.mean(moving[:,9])>.35 and not failed))
    m.update(metrics);return m


def render(e,dest):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    renderer=mujoco.Renderer(e.sim.model,height=640,width=960)
    cam=mujoco.MjvCamera();cam.distance=.8;cam.azimuth=135;cam.elevation=-12
    option=mujoco.MjvOption();option.geomgroup[3]=0
    writer=imageio.get_writer(str(dest),fps=getattr(e,'video_fps',25),codec='libx264',quality=8)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    data=mujoco.MjData(e.sim.model)
    for i,qpos in enumerate(e.qpos_frames):
        data.qpos[:]=qpos;mujoco.mj_forward(e.sim.model,data)
        cam.lookat[:]=data.subtree_com[e.sim.body];cam.lookat[2]=.12
        renderer.update_scene(data,camera=cam,scene_option=option)
        frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
        draw.rectangle((0,0,960,44),fill=(22,28,33))
        mode='RUN TASK' if e.skill=='run' else 'JUMP TASK'
        if e.skill=='getup':mode='GET-UP TASK - independent recovery attempt'
        if hasattr(e,'return_time'):mode='JUMP EXPERT' if (i+1)/25<e.return_time else 'V7 BASE RETURN'
        draw.text((15,10),f'{mode} | sim t={(i+1)/25:.2f}s | S288 / IMU / delay {e.sim.case.command_ms}ms',font=font,fill='white')
        draw.rectangle((0,600,960,640),fill=(22,28,33))
        footer='RESEARCH SIMULATION - flight requires measured ground clearance'
        if hasattr(e,'rows') and len(e.rows):
            row=e.rows[min((i+1)*32-1,len(e.rows)-1)]
            footer=f'Both-foot gap >= {min(row[14:16])*1000:.1f} mm | Ground support {row[4]:.2f} N | Research simulation'
            if e.skill=='run':
                requested=getattr(e,'speed',.6) if 1.<=(i+1)/25<7. else 0.
                footer=f'Request {requested:.2f} m/s | Body vx {row[9]:.2f} m/s | Both-foot gap {min(row[14:16])*1000:.1f} mm'
            elif e.skill=='getup':
                footer=f'Body tilt {row[8]:.1f} deg | Body-floor support {row[7]:.2f} N | Foot support {row[5]+row[6]:.2f} N'
        draw.text((15,610),footer,font=font,fill='white')
        writer.append_data(np.asarray(frame))
    writer.close();renderer.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--label',required=True);p.add_argument('--skill',choices=['run','jump','both'],default='both')
    p.add_argument('--quick',action='store_true');p.add_argument('--sensitivity',action='store_true')
    p.add_argument('--video',action='store_true');args=p.parse_args()
    dest=OUT/'evaluation'/args.label;dest.mkdir(parents=True,exist_ok=False)
    cases=[]
    for skill in (['run','jump'] if args.skill=='both' else [args.skill]):
        for speed in ([.45,.6,.8] if skill=='run' and not args.quick else [.6]):
            for delay in ([10] if args.quick else [5,10,15]):
                for seed in ([101] if args.quick else [101,102,103]):
                    cases.append((skill,speed,seed,HardwareCase(physics_dt=.00125,command_ms=delay)))
        if args.sensitivity:
            for dt,curve,voltage in [(.000625,False,12.6),(.00125,True,11.1),(.00125,True,9.9)]:
                cases.append((skill,.6,104,HardwareCase(physics_dt=dt,motor_curve=curve,voltage=voltage)))
            for case in [HardwareCase(name='feedback40',physics_dt=.00125,position_ms=40),
                         HardwareCase(name='low_friction',physics_dt=.00125,ground_friction=.6),
                         HardwareCase(name='head_carrier_plus10pct',physics_dt=.00125,head_mass_scale=1.1),
                         HardwareCase(name='feedback_drop2pct',physics_dt=.00125,feedback_drop_probability=.02)]:
                cases.append((skill,.6,104,case))
    rows=[]
    for skill,speed,seed,case in cases:
        name=f'{skill}_v{speed}_d{case.command_ms}_s{seed}_dt{case.physics_dt}_curve{int(case.motor_curve)}_V{case.voltage}'
        if case.name!='s288_protocol':name+='__'+case.name
        e=configure_owned(MotionExperiment(args.policy,skill,speed,case),(1,))
        r=dict(name=name,skill=skill,speed=speed,seed=seed,hardware=vars(case),policy_sha256=sha(args.policy))
        try:
            m,trace=e.run('stand','imu',seed,10. if skill=='run' else 6.,True)
            score(e,m);r.update(status='COMPLETE',metrics=m)
            np.savez_compressed(dest/(name+'.npz'),physics=np.asarray(e.rows),qpos=np.asarray(e.qpos_frames),
                gaze=np.asarray(e.sim.gaze_trace),head_controls=np.asarray(e.head_rows),heading=trace)
            if args.video and seed==101 and case.command_ms==10 and speed==.6:
                try:
                    render(e,dest/(skill+'.mp4'));r['video_status']='COMPLETE'
                except Exception as video_error:
                    r.update(video_status='FAILED',video_error=repr(video_error))
            print(json.dumps(dict(name=name,speed=m['moving_speed_m_s'],flight=m['max_clear_flight_s'],
                                  jump=m['jump_success'],running=m['running_with_flight'],failed=m['motion_failed'])),flush=True)
        except Exception as exc:
            r.update(status='FAILED',error=repr(exc));print(json.dumps(r),flush=True)
        rows.append(r);(dest/(name+'.json')).write_text(json.dumps(r,indent=2),encoding='utf-8')
        (dest/'matrix.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':main()
