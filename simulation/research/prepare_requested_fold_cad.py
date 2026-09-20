"""Prepare sampled actual motion transforms and a clearly labelled probe video."""
from pathlib import Path
import json
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).parent;OUT=ROOT/'20260914_requested_fold'
PLANT=ROOT/'20260914_contact_motion/preferred_plant'
m=mujoco.MjModel.from_binary_path('x',assets={'x':(PLANT/'nominal.mjb').read_bytes()});d=mujoco.MjData(m)
dest=OUT/'dynamic_cad';dest.mkdir(exist_ok=False)
frames=[json.loads((ROOT/'20260914_squat_specialist/reference_geometry/cad_pose_frames.json').read_text())[0]]
joints=[m.jnt_qposadr[m.joint('robot/'+n).id] for n in ('left_ankle','right_ankle','left_knee','right_knee','jaw_hinge')]
sources=[]
for path in sorted((OUT/'dynamic_ankle54').glob('*.npz')):
    z=np.load(path)
    if 'qpos' not in z:continue
    qs=z['qpos'];indices={round(t/.04)-1 for t in (1,2,3,4,5,6,7,8,9,10,13)}
    indices.update(int(np.argmax(abs(qs[:,j]))) for j in joints)
    for k in sorted(indices):
        d.qpos[:]=qs[k];mujoco.mj_forward(m,d);bodies={}
        for b in range(m.nbody):
            n=m.body(b).name
            if not n.startswith('robot/'):continue
            t=np.eye(4);t[:3,:3]=d.xmat[b].reshape(3,3);t[:3,3]=d.xpos[b]
            bodies[n.removeprefix('robot/')]=t.tolist()
        frames.append(dict(name=f'{path.stem}_f{k}',bodies=bodies))
    sources.append(dict(path=str(path),indices=sorted(indices)))
(dest/'cad_pose_frames.json').write_text(json.dumps(frames))
(dest/'sources.json').write_text(json.dumps(sources,indent=2))

path=next((OUT/'dynamic_ankle54').glob('*s962.npz'))
z=np.load(path);row=json.loads(path.with_suffix('.json').read_text())
renderer=mujoco.Renderer(m,width=960,height=640);option=mujoco.MjvOption();option.geomgroup[3]=0
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
writer=imageio.get_writer(str(OUT/'deeper_fold_probe.mp4'),fps=25,codec='libx264',quality=8)
for k,q in enumerate(z['qpos']):
    d.qpos[:]=q;mujoco.mj_forward(m,d);t=(k+1)*.04
    cam=mujoco.MjvCamera();cam.azimuth=90;cam.elevation=-3;cam.distance=.65;cam.lookat[:]=[-.045,0,.14]
    renderer.update_scene(d,camera=cam,scene_option=option)
    im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
    phase='站立' if t<1 else ('下蹲' if t<4 else ('保持折腿姿态' if t<7 else ('站起' if t<10 else '站稳检查')))
    draw.rectangle((0,0,960,84),fill=(21,32,43));draw.text((20,10),f'重新检查图示目标 · {phase} · {t:.2f} s',font=font,fill='white')
    draw.text((20,49),'慢速轨迹探针；不是新训练模型，也未宣称完整复现参考图',font=small,fill=(245,195,120))
    draw.rectangle((0,548,960,640),fill=(21,32,43))
    draw.text((20,559),f"实际下降 {row['actual_depth_mm']:.2f} mm · 膝关节约 75° · 实际踝角 <55°",font=font,fill='white')
    draw.text((20,602),'两次保持和站回通过；配件、舵机能力、接触配方均沿用现有模型',font=small,fill='white')
    writer.append_data(np.asarray(im))
    if k==124:im.save(OUT/'deeper_fold_probe.png')
writer.close();renderer.close();print(len(frames),'CAD poses')
