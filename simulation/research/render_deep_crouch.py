"""Preselected 40mm/seed1 clip, with exact replay of confirmation metrics."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from terrain_skill_eval import TerrainSkillExperiment
from heading_sim import WARMUP_SECONDS
from evaluate_policy import sha
from evaluate_deep_crouch import OUT


def main():
    p = argparse.ArgumentParser(); p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--label', required=True); a = p.parse_args()
    expected = json.loads((OUT/'evaluation'/a.label/'d40_stand_lag10_s1.json').read_text())
    assert expected['policy_sha256'] == sha(a.policy)
    dest = OUT/'videos'/a.label; dest.mkdir(parents=True, exist_ok=False)
    e = TerrainSkillExperiment(a.policy, posture='crouch40'); s = e.sim
    s.model.vis.global_.offwidth = 640; s.model.vis.global_.offheight = 448
    renderer = mujoco.Renderer(s.model, height=448, width=640)
    camera = mujoco.MjvCamera(); camera.azimuth = 125; camera.elevation = -12; camera.distance = .68
    writer = imageio.get_writer(str(dest/'simulation.mp4'), fps=25, codec='libx264', quality=8)
    font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 17)
    original = s.step; frames = [0]; last = [None]; failed = [None]
    def step(command, *args, **kwargs):
        original(command, *args, **kwargs)
        t = s.data.time-WARMUP_SECONDS
        if t <= 1e-8 or round(t/s.dt)%2 == 0: return
        v = s.view(s.data.time)
        if failed[0] is None and (v[3]<.055 or v[14]>60): failed[0] = t
        if failed[0] is not None and last[0] is not None:
            frame = last[0].copy(); draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 48, 640, 90), fill='#7c2028')
            draw.text((12, 60), f'FALL at {failed[0]:.2f}s; last valid frame held', font=font, fill='white')
        else:
            camera.lookat[:] = s.data.xpos[s.body]+[0, 0, .015]
            renderer.update_scene(s.data, camera=camera)
            frame = Image.new('RGB', (640, 560), '#111a28')
            frame.paste(Image.fromarray(renderer.render()), (0, 56)); draw = ImageDraw.Draw(frame)
            draw.text((12, 8), f'{a.label} | stand - crouch 40mm - return', font=font, fill='white')
            draw.text((12, 32), f't={t:4.2f}s | v07 / S288 / both IMU loops', font=font, fill='#bfcede')
            draw.text((12, 510), f'height {v[3]*1000:5.1f}mm | command {e.height_reference*1000:+5.1f}mm', font=font, fill='white')
            draw.text((12, 536), 'motor / position / velocity delay: 10 / 20 / 20 ms', font=font, fill='#bfcede')
        writer.append_data(np.asarray(frame)); last[0] = frame.copy()
        if frames[0] in (24, 149, 274): frame.save(dest/f'frame_{frames[0]:03d}.png')
        frames[0] += 1
    s.step = step
    try: metrics, _ = e.run('stand', 'imu', 1, 12., True)
    finally: writer.close(); renderer.close()
    metrics = json.loads(json.dumps(metrics))
    assert all(v == expected['metrics'][k] for k, v in metrics.items()), 'Replay changed original metrics'
    assert frames[0] == 300
    (dest/'verification.json').write_text(json.dumps(dict(metrics_match=True, frames=300,
        preselected_case='40mm stand, nominal delay, seed1', failed_at_s=failed[0], policy_sha256=sha(a.policy)), indent=2))
    print(dest/'simulation.mp4')


if __name__ == '__main__': main()
