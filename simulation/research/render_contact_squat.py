"""Show the completed independent squat on the selected contact hypothesis."""
from pathlib import Path
import json
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
OUT=Path(__file__).parent/'20260914_contact_motion'
path=next((OUT/'preferred_squat_delays/baseline').glob('d25_mu1_c10*_s962.npz'))
z=np.load(path);row=json.loads(path.with_suffix('.json').read_text());assert row['passed']
m=mujoco.MjModel.from_binary_path('x',assets={'x':(OUT/'preferred_plant/nominal.mjb').read_bytes()});d=mujoco.MjData(m)
renderer=mujoco.Renderer(m,width=960,height=640);option=mujoco.MjvOption();option.geomgroup[3]=0
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
writer=imageio.get_writer(str(OUT/'squat_contact_optimized.mp4'),fps=25,codec='libx264',quality=8)
for k,q in enumerate(z['qpos']):
    d.qpos[:]=q;mujoco.mj_forward(m,d);t=(k+1)*.04
    cam=mujoco.MjvCamera();cam.azimuth=90;cam.elevation=-5;cam.distance=.61;cam.lookat[:]=[-.05,0,.14]
    renderer.update_scene(d,camera=cam,scene_option=option)
    panel=Image.fromarray(renderer.render());draw=ImageDraw.Draw(panel)
    phase='站立' if t<1 else ('缓慢下蹲' if t<4 else ('保持蹲姿' if t<7 else ('站起' if t<10 else '站稳检查')))
    draw.rectangle((0,0,960,85),fill=(21,32,43));draw.text((22,12),f'独立蹲起 · {phase} · {t:.2f} s',font=font,fill='white')
    draw.text((22,48),'已学残差 + 蹲姿参考 · S288 12 V · 指令延迟 10 ms · 双 IMU',font=small,fill=(185,225,240))
    draw.rectangle((0,545,960,640),fill=(26,67,55))
    draw.text((22,557),f"实际下降 {row['actual_depth_mm']:.2f} mm · 保持和站回均通过",font=font,fill='white')
    draw.text((22,597),'接触参数为待实测假设；没有提高舵机能力，也没有移动电池',font=small,fill='white')
    writer.append_data(np.asarray(panel))
    if k==124:panel.save(OUT/'squat_contact_optimized.png')
writer.close();renderer.close()
