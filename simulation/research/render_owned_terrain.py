"""Predeclared CPU replay clips with exact metrics and explicit failure labels."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from evaluate_owned_terrain import OUT,cases,key,experiment,evaluate
from terrain_skill_eval import ground_height
from evaluate_policy import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--scene',choices=('steps','resume','slow_steps'),required=True);a=p.parse_args()
    if a.scene=='resume':case=next(c for c in cases() if c['program']=='resume' and c['delay']==10)
    else:case=next(c for c in cases() if c['terrain']=='steps_10' and c['speed']==(.2 if a.scene=='slow_steps' else .35))
    name=key(case,1);expected=json.loads((OUT/f'evaluation/{a.label}/{name}.json').read_text());assert expected['policy_sha256']==sha(a.policy)
    dest=OUT/f'videos/{a.label}/{a.scene}';dest.mkdir(parents=True,exist_ok=False)
    e=experiment(a.policy,case);s=e.sim;s.model.vis.global_.offwidth=640;s.model.vis.global_.offheight=448
    renderer=mujoco.Renderer(s.model,height=448,width=640)
    camera=mujoco.MjvCamera();camera.azimuth=125;camera.elevation=-12;camera.distance=.75
    writer=imageio.get_writer(str(dest/'simulation.mp4'),fps=25,codec='libx264',quality=8)
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',16)
    original=s.step;count=[0];failed=[None];last_scene=[None];body_zero=[None]
    def step(command,*args,**kwargs):
        if e.active and body_zero[0] is None:body_zero[0]=s.view(s.data.time)[4]
        original(command,*args,**kwargs);t=s.data.time-6.
        if t<=1e-8 or round(t/s.dt)%2==0:return
        v=s.view(s.data.time);clearance=v[3]-float(ground_height(e.terrain_kind,v[1],v[2]))
        if failed[0] is None and (clearance<.055 or v[14]>60):failed[0]=t
        camera.lookat[:]=s.data.xpos[s.body]+[0,0,.015];renderer.update_scene(s.data,camera=camera)
        scene=Image.fromarray(renderer.render())
        if failed[0] is not None and last_scene[0] is not None:scene=last_scene[0]
        else:last_scene[0]=scene.copy()
        frame=Image.new('RGB',(640,576),'#111a28');frame.paste(scene,(0,64));d=ImageDraw.Draw(frame)
        phase=f"10MM STEPS / CMD {case['speed']:.2f}m/s" if a.scene!='resume' else ('STAND' if t<2 else ('CROUCH' if t<7 else ('RETURN' if t<11 else ('WALK' if t<17 else 'STOP'))))
        d.text((12,10),f'{a.label} | {phase}',font=font,fill='white')
        d.text((12,36),f't={t:5.2f}s | v07 / S288 / IMU owns head yaw',font=font,fill='#bfcede')
        yaw=(v[5]-body_zero[0]-e.controller.reference+np.pi)%(2*np.pi)-np.pi
        d.text((12,514),f'clearance {clearance*1000:5.1f}mm | vx {v[13]:+.3f}m/s',font=font,fill='white')
        d.text((12,539),f'camera yaw {np.rad2deg(yaw):+.1f} / pitch {np.rad2deg(v[6]):+.1f}deg | 10/20/20ms',font=font,fill='#bfcede')
        if failed[0] is not None:d.text((12,80),f'FALL at {failed[0]:.2f}s; last valid scene held',font=font,fill='#ff5050')
        if a.scene=='slow_steps':d.text((12,104),'POST-HOC DIAGNOSTIC: low-speed progress failure',font=font,fill='#ffb95a')
        writer.append_data(np.asarray(frame))
        if count[0] in (49,149,249,449):frame.save(dest/f'frame_{count[0]:03d}.png')
        count[0]+=1
    s.step=step
    try:m,_=evaluate(e,case,1)
    finally:writer.close();renderer.close()
    assert json.loads(json.dumps(m))==expected['metrics'],'Replay differs from original evaluation'
    assert count[0]==case['seconds']*25
    r=dict(status='PASS',frames=count[0],fps=25,metrics_match=True,preselected=a.scene!='slow_steps',case=name,fall_time_s=failed[0],policy_sha256=sha(a.policy))
    (dest/'verification.json').write_text(json.dumps(r,indent=2));print(r)


if __name__=='__main__':main()
