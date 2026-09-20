"""Side-by-side measured reference and independent-policy trajectories."""
from pathlib import Path
import argparse,json
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio

p=argparse.ArgumentParser();p.add_argument('--evaluation',default='evaluation');p.add_argument('--label',default='独立蹲起策略')
a=p.parse_args();OUT=Path(__file__).resolve().parent/'20260914_squat_specialist'
m=mujoco.MjModel.from_binary_path('x',assets={'x':(OUT/'plant/nominal.mjb').read_bytes()});d=mujoco.MjData(m)
paths=[next((OUT/'friction_contact_speed').glob('d25_mu1_c10*.npz')),
       next((OUT/a.evaluation).glob('d25_mu1_c10*_s701.npz'))]
data=[np.load(f) for f in paths];rows=[json.loads(f.with_suffix('.json').read_text()) for f in paths]
renderer=mujoco.Renderer(m,width=640,height=576)
option=mujoco.MjvOption();option.geomgroup[3]=0
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
writer=imageio.get_writer(str(OUT/'squat_comparison.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=16)
for k in range(min(len(z['qpos']) for z in data)):
    canvas=Image.new('RGB',(1280,576));t=(k+1)*.04
    phase='站立' if t<1 else ('下蹲' if t<4 else ('保持' if t<7 else ('站回' if t<10 else '站稳检查')))
    for i,(trace,row) in enumerate(zip(data,rows)):
        d.qpos[:]=trace['qpos'][k];d.qvel[:]=0;mujoco.mj_forward(m,d)
        cam=mujoco.MjvCamera();cam.azimuth=90;cam.elevation=-4;cam.distance=.63
        cam.lookat[:]=[-.055,0,.14]
        renderer.update_scene(d,camera=cam,scene_option=option)
        panel=Image.fromarray(renderer.render());draw=ImageDraw.Draw(panel)
        draw.rectangle((0,0,640,83),fill=(20,29,38))
        draw.text((16,10),'脚本参考（非学习成绩）' if i==0 else a.label,font=font,fill='white')
        draw.text((16,48),f'25 mm 下蹲请求 · {phase} · {t:.2f} s',font=small,fill=(180,220,245))
        passed=row['passed'];draw.rectangle((0,482,640,576),fill=(24,66,55) if passed else (86,37,34))
        draw.text((16,493),f"实际保持下降 {row['actual_depth_mm']:.1f} mm · 最大倾斜 {row['hold_tilt_max_deg']:.1f}°",font=small,fill='white')
        draw.text((16,527),f"保持稳定 {100*row['hold_stable_fraction']:.0f}% · 站回稳定 {100*row['return_stable_fraction']:.0f}%",font=small,fill='white')
        canvas.paste(panel,(640*i,0))
    writer.append_data(np.asarray(canvas))
    if k==124:canvas.save(OUT/'squat_hold_comparison.png')
writer.close();renderer.close()
