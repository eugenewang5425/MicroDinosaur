"""Exact 100 Hz replay showing both sole clearance and support, no COM headline."""
import argparse
import json
from pathlib import Path
import sys
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from evaluate_foot_flight import FeetExperiment
from hardware_sim import HardwareCase
from imu_owned_head import configure_owned
from foot_flight_cfg import OUT


def capture(label,depth):
    folder=OUT/'evaluation'/label;name=f'c{depth}_d10_s601_s288_protocol_return0'
    r=json.loads((folder/(name+'.json')).read_text())
    e=configure_owned(FeetExperiment(Path(r['policy']),HardwareCase(**r['hardware']),False,depth/1000),(1,))
    frames=[];previous=e.sim.substep_callback
    def callback():
        previous()
        if e.active and len(e.rows)%8==0:frames.append((e.rows[-1][0],e.sim.data.qpos.copy(),np.array(e.rows[-1])))
    e.sim.substep_callback=callback;e.run('stand','imu',601,6.,True)
    a=np.asarray(e.rows);old=np.load(folder/(name+'.npz'))['physics'];error=float(abs(a-old).max());assert error==0
    streak=np.zeros(len(a));n=0
    for i,row in enumerate(a):
        good=min(row[14:16])>.005 and row[4]<.05 and row[8]<30 and row[13]<=.02
        n=n+1 if good else 0;streak[i]=n*e.sim.model.opt.timestep*1000
    return e,frames,streak,error


def main():
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--depth',type=int,choices=[30,50],default=50);args=p.parse_args()
    captures=[capture(label,args.depth) for label in ['previous','candidate']]
    renderers=[mujoco.Renderer(e.sim.model,height=576,width=640) for e,*_ in captures]
    states=[mujoco.MjData(e.sim.model) for e,*_ in captures]
    camera=mujoco.MjvCamera();camera.distance=.64;camera.azimuth=150;camera.elevation=-10
    option=mujoco.MjvOption();option.geomgroup[3]=0
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);title=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23)
    path=OUT/f'feet_pair_c{args.depth}.mp4';writer=imageio.get_writer(str(path),fps=20,codec='libx264',quality=8)
    schedule=[(i,False) for i in range(4,len(captures[0][1]),5)]
    schedule += [(i,True) for i,(t,*_) in enumerate(captures[0][1]) if 1.5<=t<=2.7]
    for i,slow in schedule:
        panels=[]
        for column,((e,frames,streak,_),renderer,data) in enumerate(zip(captures,renderers,states)):
            t,q,row=frames[i];data.qpos[:]=q;mujoco.mj_forward(e.sim.model,data)
            camera.lookat[:]=data.subtree_com[e.sim.body];camera.lookat[2]=.135
            renderer.update_scene(data,camera=camera,scene_option=option);frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,640,83),fill=(22,28,33))
            draw.text((12,7),'PREVIOUS POLICY' if column==0 else 'FEET-OBJECTIVE TRAINING',font=title,fill='white')
            phase='CROUCH REQUEST' if t<1.8 else ('JUMP REQUEST' if t<2.4 else 'ZERO COMMAND / SETTLE')
            draw.text((12,37),f'{phase} | {args.depth} mm crouch | t={t:.2f}s',font=font,fill='white')
            draw.text((12,60),'0.2x SLOW REPLAY' if slow else '1x | POLICY ALONE',font=font,fill='#b1dfd8')
            draw.rectangle((0,495,640,576),fill=(22,28,33))
            draw.text((12,501),f'Both soles minimum: {min(row[14:16])*1000:.1f} mm | Support: {row[4]:.2f} N',font=font,fill='white')
            draw.text((12,525),f'Continuous >5mm: {streak[min((i+1)*8-1,len(streak)-1)]:.1f} ms | Goal: 60ms + 10mm peak',font=font,fill='white')
            draw.text((12,551),'S288 provisional curve / 12V / 10ms / SIMULATION',font=font,fill='white')
            panels.append(np.asarray(frame))
        writer.append_data(np.concatenate(panels,axis=1))
    writer.close()
    for r in renderers:r.close()
    (OUT/f'video_c{args.depth}_audit.json').write_text(json.dumps(dict(trace_max_errors=[x[3] for x in captures],capture_hz=100,
        simulation_seconds=6,playback_fps=20,slow_speed=.2,seed=601,depth_mm=args.depth),indent=2),encoding='utf-8')
    print(path)


if __name__=='__main__':main()
