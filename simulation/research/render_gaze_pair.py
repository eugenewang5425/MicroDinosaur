"""Render preselected seed 42 under the exact primary forward evaluation."""
from dataclasses import asdict
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from evaluate_gaze_ablation import GazeSim,CASES
from evaluate_policy import sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_head_gaze_ablation/video_seed42'
RUNS=Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_v07_calibration')


def main():
    OUT.mkdir(exist_ok=True)
    case=CASES[0];assert case.name=='nominal_1p25ms'
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',18)
    small=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',15)
    policies=[RUNS/f'20260914_gaze_train_s42_{a}_1024x101/candidate.onnx' for a in ('control','no_neck_cost')]
    sims=[GazeSim(ROOT/'20260913_handoff/native_v07',p,case) for p in policies]
    renderers=[];cameras=[];command=np.zeros(18);command[0]=.55
    for sim in sims:
        sim.model.vis.global_.offwidth=640;sim.model.vis.global_.offheight=576
        renderers.append(mujoco.Renderer(sim.model,height=576,width=640))
        cam=mujoco.MjvCamera();cam.azimuth=130;cam.elevation=-15;cam.distance=.70;cameras.append(cam)
        sim.reset(0,False)
        for _ in range(100):sim.step(command)
    start=[sim.view(sim.data.time) for sim in sims]
    writer=imageio.get_writer(str(OUT/'forward_pair.mp4'),fps=25,codec='libx264',quality=8)
    frames=0;last_samples=[]
    try:
        for tick in range(300):
            for sim in sims:sim.step(command)
            if tick%2:continue
            canvas=Image.new('RGB',(1280,720),(15,23,34));draw=ImageDraw.Draw(canvas)
            draw.text((20,12),f'Paired seed 42 | forward 0.55 m/s | yaw command 0 | t={(tick+1)*.02:.2f}s',font=font,fill='white')
            draw.text((20,38),'Physics 1.25 ms; command / position / velocity delay 10 / 20 / 20 ms; nominal kd 0.8',font=small,fill='#c2cad5')
            last_samples=[]
            for i,(sim,renderer,cam) in enumerate(zip(sims,renderers,cameras)):
                cam.lookat[:]=sim.data.xpos[sim.body]+[0,0,.045]
                renderer.update_scene(sim.data,camera=cam)
                pos=sim.data.site_xpos[sim.camera].copy()
                forward=-sim.data.site_xmat[sim.camera].reshape(3,3)[:,2]
                # Scene-only arrows, no model or physics modification.
                for direction,color in ((np.array([1.,0,0]),[.15,.75,.35,1.]),(forward,[1.,.4,.1,1.])):
                    geom=renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color))
                    mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.004,pos,pos+.14*direction)
                    renderer.scene.ngeom+=1
                canvas.paste(Image.fromarray(renderer.render()),(i*640,70))
                sample=sim.view(sim.data.time);last_samples.append(sample)
                color='#7fb7e0' if i==0 else '#ffad70'
                draw.text((i*640+20,652),'Neck cost -0.2 (control)' if i==0 else 'Neck cost 0 (ablation)',font=font,fill=color)
                draw.text((i*640+20,677),f'Camera yaw {np.rad2deg(sample[5]):+.1f} deg | body turn {np.rad2deg(sample[4]-start[i][4]):+.1f} deg',font=small,fill='white')
                draw.text((i*640+20,700),'Green: requested forward; orange: camera optical axis',font=small,fill='#c2cad5')
            writer.append_data(np.asarray(canvas));frames+=1
            if tick in (0,150,298):canvas.save(OUT/f'frame_{tick:03d}.png')
    finally:
        writer.close()
        for renderer in renderers:renderer.close()
    # Check that visual scene updates did not alter either trajectory.
    endpoint_checks=[]
    for arm,sim in zip(('control','no_neck_cost'),sims):
        # Both final samples follow the actor step's mj_forward at t=8.0.
        # Intermediate stored samples are the beginning of physical substeps.
        sample=sim.view(sim.data.time)
        trace=np.load(ROOT/f'20260914_head_gaze_ablation/evaluation/s42_{arm}/s42_{arm}__forward.npz')['trace']
        expected=trace[-1]
        error=float(np.max(abs(sample-expected)));assert error<1e-8,error
        endpoint_checks.append(error)
    contract={'case':asdict(case),'training_seed':42,'evaluation_seed':0,'task':'forward',
        'seconds':6,'warmup_seconds':2,'frame_count':frames,'fps':25,'frame_size':[1280,720],
        'command':command.tolist(),'policies':[{'path':str(p),'sha256':sha(p)} for p in policies],
        'camera_arrows':'green world +X target, orange optical -Z; visual scene only',
        'rendered_endpoint_max_error_vs_saved_trace':endpoint_checks,'renderer_source_sha256':sha(Path(__file__))}
    (OUT/'contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    print(json.dumps(contract,indent=2),flush=True)


if __name__=='__main__':main()
