"""Show actual dynamic trajectories with the failed CAD gate visible."""
from pathlib import Path
import json
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
OUT=Path(__file__).resolve().parent/'20260914_compact_fold'
plant=OUT/'floor_only_plant'
m=mujoco.MjModel.from_binary_path('x',assets={'x':(plant/'nominal.mjb').read_bytes()});d=mujoco.MjData(m)
data=[np.load(next((OUT/'floor_only_nominal').glob(f'd{dep}_*.npz'))) for dep in (30,40,50)]
rows=[json.loads(next((OUT/'floor_only_nominal').glob(f'd{dep}_*.json')).read_text()) for dep in (30,40,50)]
renderer=mujoco.Renderer(m,width=512,height=512)
option=mujoco.MjvOption();option.geomgroup[3]=0
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
writer=imageio.get_writer(str(OUT/'compact_fold_trial.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=16)
for k in range(len(data[0]['qpos'])):
    canvas=Image.new('RGB',(1536,512));t=(k+1)*.04
    phase='站立' if t<1 else ('缓慢收腿' if t<4 else ('保持' if t<7 else ('恢复站立' if t<10 else '站稳检查')))
    for i,(trace,row) in enumerate(zip(data,rows)):
        d.qpos[:]=trace['qpos'][k];d.qvel[:]=0;mujoco.mj_forward(m,d)
        cam=mujoco.MjvCamera();cam.azimuth=90;cam.elevation=-4;cam.distance=.66
        cam.lookat[:]=[-.015,0,.143]
        renderer.update_scene(d,camera=cam,scene_option=option)
        panel=Image.fromarray(renderer.render());draw=ImageDraw.Draw(panel)
        draw.rectangle((0,0,512,74),fill=(20,29,38))
        draw.text((14,9),f"目标下降 {row['depth_mm']} mm · {phase}",font=font,fill='white')
        draw.text((14,42),f"t={t:.2f}s  实际保持下降 {row['actual_depth_mm']:.1f} mm",font=small,fill=(174,219,244))
        draw.rectangle((0,440,512,512),fill=(71,26,30))
        draw.text((12,450),'双脚支撑、保持及站回：动力学初检通过',font=small,fill='white')
        draw.text((12,480),'CAD 零件干涉未通过 · 非成功动作示范',font=small,fill=(255,195,185))
        canvas.paste(panel,(512*i,0))
    writer.append_data(np.asarray(canvas))
    if k==124:canvas.save(OUT/'compact_fold_hold.png')
writer.close();renderer.close()
print('Rendered actual 13-second trajectories at 25 fps, with CAD failure labels.')
