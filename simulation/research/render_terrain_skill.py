"""Render a preselected seed, asserting equality with its confirmation record."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from heading_sim import WARMUP_SECONDS
from terrain_skill_eval import TerrainSkillExperiment, ground_height, OUT
from evaluate_policy import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--suite', required=True)
    p.add_argument('--terrain', required=True)
    p.add_argument('--posture', default='none')
    p.add_argument('--scenario', default='straight')
    p.add_argument('--seed', type=int, default=1)
    a = p.parse_args()
    name = f'{a.terrain}__{a.posture}__{a.scenario}__s{a.seed}'
    expected = json.loads((OUT/'evaluation'/a.label/a.suite/(name+'.json')).read_text())
    assert expected['policy_sha256'] == sha(a.policy)
    dest = OUT/'videos'/a.label/name; dest.mkdir(parents=True, exist_ok=True)
    e = TerrainSkillExperiment(a.policy, a.terrain, a.posture); s = e.sim
    s.model.vis.global_.offwidth = 800; s.model.vis.global_.offheight = 512
    renderer = mujoco.Renderer(s.model, height=512, width=800)
    camera = mujoco.MjvCamera(); camera.azimuth=130; camera.elevation=-15; camera.distance=.8
    writer = imageio.get_writer(str(dest/'simulation.mp4'), fps=25, codec='libx264', quality=8)
    font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 18)
    original = s.step; counter = [0]; last_valid=[None]; failed_at=[None]
    def step(command, *args, **kwargs):
        original(command, *args, **kwargs)
        t = s.data.time-WARMUP_SECONDS
        if t<=1e-8 or round(t/s.dt)%2 == 0: return
        v = s.view(s.data.time)
        height=v[3]-float(ground_height(a.terrain,v[1],v[2]))
        if failed_at[0] is None and (height<.055 or v[14]>60): failed_at[0]=t
        if failed_at[0] is not None and last_valid[0] is not None:
            # The locomotion plant terminates before full-body ground contact.
            # Freeze the last valid picture; do not present post-fall clipping
            # from the feet-only plant as physical behavior.
            frame=last_valid[0].copy();d=ImageDraw.Draw(frame)
            d.rectangle((0,64,800,101),fill='#7c2028')
            d.text((12,73),f'FALL at {failed_at[0]:.2f}s - last valid frame held',font=font,fill='white')
            writer.append_data(np.asarray(frame))
            if counter[0] in (0,125,249):frame.save(dest/f'frame_{counter[0]:03d}.png')
            counter[0]+=1;return
        camera.lookat[:] = s.data.xpos[s.body]+[0, 0, .025]
        renderer.update_scene(s.data, camera=camera)
        frame = Image.new('RGB', (800, 640), (15, 23, 34))
        frame.paste(Image.fromarray(renderer.render()), (0, 64)); d = ImageDraw.Draw(frame)
        d.text((12, 8), f'{a.label} | {a.terrain} | {a.posture}', font=font, fill='white')
        d.text((12, 34), f't={t:5.2f}s | v07 / S288 / body + head IMU', font=font, fill='#bbc9d8')
        d.text((12, 584), f'Height {height*1000:5.1f} mm | target {(0.117182+e.height_reference)*1000:5.1f} mm', font=font, fill='white')
        d.text((12, 612), f'vx {v[13]:+.3f} m/s | motor / feedback 10 / 20 / 20 ms', font=font, fill='#bbc9d8')
        writer.append_data(np.asarray(frame))
        last_valid[0]=frame.copy()
        if counter[0] in (0, 125, 249): frame.save(dest/f'frame_{counter[0]:03d}.png')
        counter[0]+=1
    s.step=step
    try: metrics,_=e.run(a.scenario,'imu',a.seed,12.,True)
    finally: writer.close(); renderer.close()
    assert json.loads(json.dumps(metrics)) == expected['metrics'], 'Rendering changed the trajectory'
    assert counter[0]==300
    (dest/'verification.json').write_text(json.dumps(dict(frames=300, fps=25,
        metrics_match_confirmation=True, failed_at_s=failed_at[0],
        post_failure_frames_hold_last_valid=True, policy_sha256=sha(a.policy)),indent=2),encoding='utf-8')
    print(str(dest/'simulation.mp4'), 'render PASS')


if __name__=='__main__': main()
