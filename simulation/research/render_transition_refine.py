"""Preselected full 20s crouch-walk-stop video; exact confirmation replay."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from evaluate_transition_refine import TransitionExperiment,OUT,score
from heading_sim import WARMUP_SECONDS
from evaluate_policy import sha
from terrain_skill_eval import TerrainSkillExperiment,ground_height


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--case',choices=('resume','steps10'),default='resume')
    a=p.parse_args();steps=a.case=='steps10';seconds=12. if steps else 20.
    name='legacy_steps_10_straight_d0_lag10_s1' if steps else 'resume_flat_stand_d40_lag10_s1'
    expected=json.loads((OUT/'evaluation'/a.label/('confirmation' if steps else 'transitions')/(name+'.json')).read_text())
    assert expected['policy_sha256']==sha(a.policy)
    dest=OUT/'videos'/(a.label+('_steps10_failure' if steps else ''));dest.mkdir(parents=True,exist_ok=False)
    e=TerrainSkillExperiment(a.policy,terrain='steps_10') if steps else TransitionExperiment(a.policy,posture='crouch40',program='resume')
    s=e.sim
    s.model.vis.global_.offwidth=640;s.model.vis.global_.offheight=448
    renderer=mujoco.Renderer(s.model,height=448,width=640)
    camera=mujoco.MjvCamera();camera.azimuth=125;camera.elevation=-12;camera.distance=.68
    writer=imageio.get_writer(str(dest/'simulation.mp4'),fps=25,codec='libx264',quality=8)
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',16)
    original=s.step;frame_count=[0];fall_time=[None];last_scene=[None]
    def step(command,*args,**kwargs):
        original(command,*args,**kwargs);t=s.data.time-WARMUP_SECONDS
        if t<=1e-8 or round(t/s.dt)%2==0:return
        v=s.view(s.data.time)
        if fall_time[0] is None and (v[3]-float(ground_height(e.terrain_kind,v[1],v[2]))<.055 or v[14]>60):fall_time[0]=t
        camera.lookat[:]=s.data.xpos[s.body]+[0,0,.015]
        renderer.update_scene(s.data,camera=camera)
        scene=Image.fromarray(renderer.render())
        if fall_time[0] is not None and last_scene[0] is not None:scene=last_scene[0]
        else:last_scene[0]=scene.copy()
        frame=Image.new('RGB',(640,576),'#111a28');frame.paste(scene,(0,64))
        d=ImageDraw.Draw(frame)
        phase='STAND' if t<2 else ('CROUCH' if t<7 else ('RETURN / HOLD' if t<11 else ('WALK' if t<17 else 'STOP')))
        if steps:phase='10MM STEPS'
        d.text((12,10),f'{a.label} | {phase} | t={t:5.2f}s',font=font,fill='white')
        d.text((12,36),'v07 / S288 / calibrated dual-IMU control',font=font,fill='#bfcede')
        user_vx=.55 if steps else e.shaped_user[0]
        d.text((12,514),f'height {v[3]*1000:5.1f}mm  vx {v[13]:+.3f}m/s  user {user_vx:.2f}',font=font,fill='white')
        d.text((12,539),f'camera pitch {np.rad2deg(v[6]):+.1f}deg | delays 10/20/20ms',font=font,fill='#bfcede')
        if fall_time[0] is not None:d.text((12,80),f'FALL at {fall_time[0]:.2f}s; last valid scene held',font=font,fill='#ff5050')
        writer.append_data(np.asarray(frame))
        if frame_count[0] in (24,149,249,374,474):frame.save(dest/f'frame_{frame_count[0]:03d}.png')
        frame_count[0]+=1
    s.step=step
    try:m,_=e.run('straight' if steps else 'stand','imu',1,seconds,True);score(e,m)
    finally:writer.close();renderer.close()
    assert json.loads(json.dumps(m))==expected['metrics'],'Replay differs from confirmation'
    assert frame_count[0]==round(seconds*25)
    record=dict(status='PASS',frames=frame_count[0],fps=25,metrics_match=True,case=name,fall_time_s=fall_time[0],policy_sha256=sha(a.policy),
        selection='Failure diagnostic selected after seeing terrain failure' if steps else 'Preselected before formal training')
    (dest/'verification.json').write_text(json.dumps(record,indent=2));print(record)


if __name__=='__main__':main()
