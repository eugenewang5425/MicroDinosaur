"""Physics-preserving 100 Hz replay capture; complete action and 0.2x launch replay."""
import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_jump_refine import RefineExperiment
from hardware_sim import HardwareCase
from imu_owned_head import configure_owned
from jump_refine_cfg import OUT


def capture(label, curve):
    directory = OUT/'evaluation'/label
    name = f'curve{curve}_d10_s401_s288_protocol_return0'
    record = json.loads((directory/(name+'.json')).read_text())
    e = configure_owned(RefineExperiment(Path(record['policy']), HardwareCase(**record['hardware']), False), (1,))
    frames = []; previous = e.sim.substep_callback
    def callback():
        previous()
        if e.active and len(e.rows)%8 == 0:
            frames.append((e.rows[-1][0],e.sim.data.qpos.copy(),np.array(e.rows[-1])))
    e.sim.substep_callback = callback
    e.run('stand', 'imu', 401, 6., True)
    old = np.load(directory/(name+'.npz'))['physics']
    error = float(np.max(abs(np.asarray(e.rows)-old)))
    assert error == 0, error
    return e,frames,record,error


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--curve',type=int,choices=[0,1],default=1);args=p.parse_args()
    captures = [capture(label,args.curve) for label in ['source','candidate']]
    width,height=640,576
    renderers=[mujoco.Renderer(e.sim.model,height=height,width=width) for e,*_ in captures]
    states=[mujoco.MjData(e.sim.model) for e,*_ in captures]
    camera=mujoco.MjvCamera();camera.distance=.64;camera.azimuth=150;camera.elevation=-10
    options=mujoco.MjvOption();options.geomgroup[3]=0
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
    big=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
    destination=OUT/f'jump_pair_curve{args.curve}.mp4'
    writer=imageio.get_writer(str(destination),fps=20,codec='libx264',quality=8)
    # Native state at 100 Hz. Play every fifth frame for normal 20 fps, then
    # every frame for a 0.2x replay of the commanded launch and first landing.
    schedule=[(i,False) for i in range(4,len(captures[0][1]),5)]
    schedule += [(i,True) for i,(t,*_) in enumerate(captures[0][1]) if 1.35<=t<=2.3]
    for i,slow in schedule:
        panels=[]
        for column,((e,frames,record,_),renderer,state) in enumerate(zip(captures,renderers,states)):
            t,q,row=frames[i];state.qpos[:]=q;mujoco.mj_forward(e.sim.model,state)
            camera.lookat[:]=state.subtree_com[e.sim.body];camera.lookat[2]=.135
            renderer.update_scene(state,camera=camera,scene_option=options)
            frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,width,84),fill=(22,28,33))
            draw.text((14,7),'SOURCE' if column==0 else 'AFTER 401 PPO UPDATES',font=big,fill='white')
            phase='CROUCH' if t<1.5 else ('EXTEND / JUMP' if t<2.2 else 'ZERO COMMAND / SETTLE')
            draw.text((14,39),f'{phase} | t={t:.2f}s',font=font,fill='white')
            draw.text((14,62),'0.2x SLOW REPLAY' if slow else '1x | JUMP EXPERT ALONE',font=font,fill='#b1dfd8')
            draw.rectangle((0,height-78,width,height),fill=(22,28,33))
            draw.text((14,height-72),f'COM above floor: {row[3]*1000:.1f} mm',font=font,fill='white')
            draw.text((14,height-48),f'Both-foot gap: {min(row[14:16])*1000:.1f} mm | Support: {row[4]:.2f} N',font=font,fill='white')
            draw.text((14,height-24),'12V speed-torque assumption / 10ms / simulation' if args.curve else 'Legacy peak torque / 10ms / simulation',font=font,fill='white')
            panels.append(np.asarray(frame))
        writer.append_data(np.concatenate(panels,axis=1))
    writer.close()
    for renderer in renderers:renderer.close()
    (OUT/f'video_curve{args.curve}_audit.json').write_text(json.dumps(dict(
        source_and_candidate_trace_errors=[x[3] for x in captures],capture_hz=100,playback_fps=20,
        complete_seconds=6,slow_replay_speed=.2,curve=bool(args.curve),seed=401,command_ms=10,
        filename=destination.name),indent=2),encoding='utf-8')
    print(destination)


if __name__=='__main__':main()
