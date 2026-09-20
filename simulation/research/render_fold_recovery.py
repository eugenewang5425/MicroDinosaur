"""Replay saved simulated positions without re-running dynamics."""
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from fold_recovery_probe import Probe,OUT
from evaluate_policy import sha


def render(stem,title):
    path=OUT/'trials'/(stem+'.npz');a=np.load(path);trace=a['trace'];qpos=a['qpos']
    e=Probe();m,d=e.model,e.data;m.vis.global_.offwidth=720;m.vis.global_.offheight=512
    renderer=mujoco.Renderer(m,height=512,width=720);cam=mujoco.MjvCamera()
    cam.azimuth=130;cam.elevation=-16;cam.distance=.72;cam.lookat[:]=[-.055,0,.1]
    dest=OUT/'videos';dest.mkdir(exist_ok=True);video=dest/(stem+'.mp4')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    writer=imageio.get_writer(str(video),fps=25,codec='libx264',quality=8)
    try:
        for counter,i in enumerate(range(0,len(qpos),2)):
            d.qpos[:]=qpos[i];d.qvel[:]=0;d.time=trace[i,0]+.00125;mujoco.mj_forward(m,d)
            if not stem.startswith('fold_'):
                cam.lookat[:]=d.subtree_com[e.trunk]+[0,0,.01];cam.distance=.9
            renderer.update_scene(d,camera=cam)
            frame=Image.new('RGB',(720,640),(15,23,34));frame.paste(Image.fromarray(renderer.render()),(0,64));draw=ImageDraw.Draw(frame)
            draw.text((12,8),title,font=font,fill='white')
            draw.text((12,35),'v07 / S288 | 有界位置轨迹试验，非RL策略',font=font,fill='#c1cfdf')
            draw.text((12,585),f'时间 {trace[i,0]:.2f}s   身体高度 {trace[i,1]*1000:.1f}mm   倾斜 {trace[i,2]:.1f}°',font=font,fill='white')
            draw.text((12,612),'重力、正阻尼、10ms目标延迟；身体触地代理待完善',font=font,fill='#c1cfdf')
            writer.append_data(np.asarray(frame))
            if abs(trace[i,0]-4)<.01:frame.save(dest/(stem+'.png'))
    finally:writer.close();renderer.close()
    (dest/(stem+'_replay.json')).write_text(json.dumps(dict(source_trace_sha256=sha(path),
        replay=True,physics_re_run=False,frames=counter+1,fps=25),indent=2),encoding='utf-8')
    return video


def main():
    left=render('fold_40mm_mesh__s0__lag10','降低4厘米，再回站姿：本次完成')
    right=render('fold_70mm_mesh__s0__lag10','要求降低7厘米：失稳，未回到站姿')
    readers=[imageio.get_reader(str(p)) for p in (left,right)]
    writer=imageio.get_writer(str(OUT/'videos/fold_comparison.mp4'),fps=25,codec='libx264',quality=8)
    try:
        for i,frames in enumerate(zip(*readers)):
            frame=np.concatenate(frames,axis=1);writer.append_data(frame)
            if i==100:Image.fromarray(frame).save(OUT/'videos/fold_comparison.png')
    finally:
        writer.close()
        for r in readers:r.close()
    render('supine_fold_extend__s0__lag10','仰卧收腿—伸腿：未能稳定站起')
    print('Replay rendering complete',flush=True)


if __name__=='__main__':main()
