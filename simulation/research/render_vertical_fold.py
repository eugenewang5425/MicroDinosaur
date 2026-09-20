"""Static shape explanation; conspicuous limit and collision qualifications."""
from pathlib import Path
import json
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).parent;OUT=ROOT/'20260915_vertical_fold'
m=mujoco.MjModel.from_binary_path('x',assets={'x':(ROOT/'20260914_contact_motion/preferred_plant/nominal.mjb').read_bytes()})
d=mujoco.MjData(m);renderer=mujoco.Renderer(m,width=800,height=640)
option=mujoco.MjvOption();option.geomgroup[3]=0
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
rows=json.loads((OUT/'poses.json').read_text())
panel=Image.new('RGB',(1600,800),(21,32,43))
selected=[rows[0],rows[2]]
for i,row in enumerate(selected):
    d.qpos[:]=row['qpos'];mujoco.mj_forward(m,d)
    cam=mujoco.MjvCamera();cam.azimuth=90;cam.elevation=-2;cam.distance=.54;cam.lookat[:]=[-.048,0,.085]
    renderer.update_scene(d,camera=cam,scene_option=option);im=Image.fromarray(renderer.render())
    draw=ImageDraw.Draw(im);draw.rectangle((0,0,800,100),fill=(21,32,43))
    title='现有限位下：小腿竖直，大腿仍斜着' if i==0 else '你要的方向：大、小腿折回并排竖直'
    draw.text((16,10),title,font=font,fill='white')
    subtitle='静态诊断 · 膝 90° · 大小腿仍差约 31.6°' if i==0 else '仅作目标示意 · 髋/膝约 121.6°，超出现模型 ±90°'
    draw.text((16,48),subtitle,font=small,fill=(255,185,100))
    draw.text((16,76),'两图均不是训练结果，也不是可执行动作',font=small,fill=(255,185,100))
    panel.paste(im,(i*800,0))
    draw=ImageDraw.Draw(panel)
    x=i*800+135;y=692;s=1.8
    h,k,a=[np.array(row['legs'][0][n]) for n in ('hip_mm','knee_mm','ankle_mm')]
    # Schematic uses measured sagittal coordinates; slight display separation
    # of coincident segments makes the two links readable, not a geometry edit.
    def pt(p,shift=0):return (x+(p[0]-k[0])*s+shift,y-(p[2]-k[2])*s)
    draw.line([pt(a),pt(k)],fill=(100,215,255),width=7)
    draw.line([pt(k,10),pt(h,10)],fill=(255,164,88),width=7)
    for p,color,shift in [(a,(100,215,255),0),(k,(240,240,240),5),(h,(255,164,88),10)]:
        xx,yy=pt(p,shift);draw.ellipse((xx-5,yy-5,xx+5,yy+5),fill=color)
    draw.text((i*800+260,678),'蓝：小腿   橙：大腿   顶点：膝',font=small,fill='white')
    draw.text((i*800+260,714),f"髋轴高出踝轴：{row['legs'][0]['hip_above_ankle_mm']:.2f} mm",font=small,fill='white')
    draw.text((i*800+260,750),'下方示意线做了少量横向错开以便辨认',font=small,fill=(180,190,200))
renderer.close();panel.save(OUT/'vertical_fold_comparison.png')
