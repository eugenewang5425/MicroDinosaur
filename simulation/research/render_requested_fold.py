"""Side-view evidence: measured shallow squat versus static folded candidates."""
from pathlib import Path
import json
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).parent;OUT=ROOT/'20260914_requested_fold'
m=mujoco.MjModel.from_binary_path('x',assets={'x':(ROOT/'20260914_contact_motion/preferred_plant/nominal.mjb').read_bytes()})
d=mujoco.MjData(m);renderer=mujoco.Renderer(m,width=560,height=640)
option=mujoco.MjvOption();option.geomgroup[3]=0
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
old=next((ROOT/'20260914_contact_motion/preferred_squat_delays/baseline').glob('d25_mu1_c10*_s962.npz'))
rows=json.loads((OUT/'kinematic_candidates.json').read_text())
poses=[('之前交付：浅蹲',np.load(old)['qpos'][124],'实际动力学帧；约下降 25 mm'),
    ('更紧折腿候选：膝 75°',next(r['qpos'] for r in rows if r['requested_knee_deg']==75),'静态求解；尚未通过动态保持'),
    ('更紧折腿候选：膝 85°',next(r['qpos'] for r in rows if r['requested_knee_deg']==85),'静态求解；髋部舵机壳与脚架干涉')]
panel=Image.new('RGB',(1680,640))
for i,(label,q,note) in enumerate(poses):
    d.qpos[:]=q;mujoco.mj_forward(m,d)
    cam=mujoco.MjvCamera();cam.azimuth=90;cam.elevation=-3;cam.distance=.57;cam.lookat[:]=[-.045,0,.13]
    renderer.update_scene(d,camera=cam,scene_option=option);im=Image.fromarray(renderer.render())
    draw=ImageDraw.Draw(im);draw.rectangle((0,0,560,75),fill=(21,32,43))
    draw.text((15,10),label,font=font,fill='white');draw.text((15,45),note,font=small,fill=(245,195,120))
    draw.rectangle((0,595,560,640),fill=(21,32,43));draw.text((15,605),'照片只能指定形态，不能直接读取关节角',font=small,fill='white')
    panel.paste(im,(i*560,0))
renderer.close();panel.save(OUT/'pose_comparison.png')
solids=json.loads((OUT/'solid_intersections.json').read_text());home={tuple(sorted(r['parts'])):r['intersection_mm3'] for r in solids[0]['pairs']}
gate=[]
for f in solids[1:]:
    bad=[dict(parts=r['parts'],volume_mm3=r['intersection_mm3'],increase_mm3=r['intersection_mm3']-home[tuple(sorted(r['parts']))])
        for r in f['pairs'] if r['intersection_mm3']-home[tuple(sorted(r['parts']))]>.001]
    gate.append(dict(frame=f['frame'],passed=not bad,intersections=bad))
(OUT/'static_geometry_gate.json').write_text(json.dumps(gate,indent=2))
print(json.dumps(gate))
