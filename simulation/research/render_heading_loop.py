"""Paired video of preselected initial state 1, preserving evaluated dynamics."""
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from heading_sim import HeadingExperiment,WARMUP_SECONDS
from run_heading_stable_start import StableStartExperiment,POLICY,CALIBRATOR
from hardware_sim import HardwareCase
from evaluate_policy import sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_imu_heading/videos'


def render(policy_label,policy,kind):
    dest=OUT/policy_label;dest.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',17)
    small=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',14)
    frame_counts=[];metrics_equal=[]
    for mode in ('open','imu'):
        experiment=kind(ROOT/'20260913_handoff/native_v07',policy,HardwareCase(physics_dt=.00125))
        sim=experiment.sim;sim.model.vis.global_.offwidth=640;sim.model.vis.global_.offheight=576
        renderer=mujoco.Renderer(sim.model,height=576,width=640)
        camera=mujoco.MjvCamera();camera.azimuth=130;camera.elevation=-15;camera.distance=.70
        writer=imageio.get_writer(str(dest/f'{mode}.mp4'),fps=25,codec='libx264',quality=8)
        original=sim.step;frames=[];zero=[None]
        def step(command,*args,**kwargs):
            original(command,*args,**kwargs)
            if abs(sim.data.time-WARMUP_SECONDS)<1e-8:zero[0]=sim.view(sim.data.time)[4]
            if sim.data.time<=WARMUP_SECONDS+1e-8:return
            tick=round((sim.data.time-WARMUP_SECONDS)/sim.dt)
            if tick%2==0:return
            view=sim.view(sim.data.time);target=zero[0]+experiment.controller.reference
            camera.lookat[:]=sim.data.xpos[sim.body]+[0,0,.045]
            renderer.update_scene(sim.data,camera=camera)
            origin=sim.data.site_xpos[sim.camera].copy()
            forward=-sim.data.site_xmat[sim.camera].reshape(3,3)[:,2]
            for direction,color in ((np.array([np.cos(target),np.sin(target),0]),[.15,.75,.35,1]),(forward,[1,.4,.1,1])):
                geom=renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,dtype=float))
                mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.004,origin,origin+.14*direction)
                renderer.scene.ngeom+=1
            image=Image.new('RGB',(640,720),(15,23,34));image.paste(Image.fromarray(renderer.render()),(0,65))
            draw=ImageDraw.Draw(image)
            draw.text((16,10),f'{policy_label} | '+('Open loop' if mode=='open' else 'IMU heading feedback'),font=font,fill='white')
            draw.text((16,35),f'vx command 0.55 m/s | time {sim.data.time-WARMUP_SECONDS:.2f}s',font=small,fill='#c7d1dc')
            draw.text((16,650),f'Body error {np.rad2deg(view[4]-target):+.1f} deg | applied wz {command[2]:+.3f}',font=font,fill='white')
            draw.text((16,678),'Green: desired forward; orange: camera optical axis',font=small,fill='#c7d1dc')
            draw.text((16,701),'S288; physics 1.25ms; motor/fbk 10/20/20ms; outer IMU +20ms',font=small,fill='#c7d1dc')
            writer.append_data(np.asarray(image));frames.append(tick)
        sim.step=step
        try:metrics,_=experiment.run('straight',mode,1,12.)
        finally:writer.close();renderer.close()
        expected=ROOT/('20260914_imu_heading/trials/nominal__v7__1__straight__'+mode+'.json') if policy_label=='v7' else ROOT/f'20260914_imu_heading/stable_start/trials/s1__straight__{mode}.json'
        assert metrics==json.loads(expected.read_text())['metrics'],'Rendering changed trajectory'
        assert len(frames)==300;frame_counts.append(len(frames));metrics_equal.append(True)
    readers=[imageio.get_reader(str(dest/f'{mode}.mp4')) for mode in ('open','imu')]
    writer=imageio.get_writer(str(dest/'comparison.mp4'),fps=25,codec='libx264',quality=8)
    try:
        for index,(left,right) in enumerate(zip(*readers)):
            combined=np.concatenate([left,right],axis=1);writer.append_data(combined)
            if index in (0,150,299):Image.fromarray(combined).save(dest/f'frame_{index:03d}.png')
    finally:
        writer.close()
        for reader in readers:reader.close()
    contract={'policy_sha256':sha(policy),'startup_policy_sha256':sha(CALIBRATOR),
        'seed':1,'scenario':'straight','seconds':12,'frames_per_arm':frame_counts,'fps':25,'size':[1280,720],
        'all_rendered_metrics_equal_saved_evaluation':metrics_equal,'renderer_sha256':sha(Path(__file__))}
    (dest/'contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    print(policy_label,'PASS',flush=True)


if __name__=='__main__':
    render('v7',CALIBRATOR,HeadingExperiment)
    render('s42_no_neck',POLICY,StableStartExperiment)
