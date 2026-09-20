"""Render the exact nominal evaluation model, with commands visible in each frame."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from evaluate_policy import Sim, sha

p = argparse.ArgumentParser()
p.add_argument('--plant', required=True)
p.add_argument('--onnx', required=True)
p.add_argument('--out', required=True)
p.add_argument('--seconds', type=float, default=4)
p.add_argument('--command-lag', type=int, default=0)
p.add_argument('--obs-lag', type=int, default=0)
p.add_argument('--hardware-case', default=None, help='Named case from evaluate_hardware.py; supplies its own delays')
args = p.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
if args.hardware_case:
    from evaluate_hardware import cases
    from hardware_sim import HardwareSim
    selected = {c.name: c for c in cases()}[args.hardware_case]
    if args.command_lag or args.obs_lag:
        raise ValueError('Use the hardware case delays without separate legacy lag flags')
    sim = HardwareSim(args.plant, args.onnx, selected)
else:
    sim = Sim(args.plant, args.onnx, args.command_lag, args.obs_lag)
(out / 'evaluation_contract.json').write_text(json.dumps({
    'onnx': str(Path(args.onnx).resolve()), 'onnx_sha256': sha(Path(args.onnx)),
    'plant_sha256': sha(Path(args.plant) / 'nominal.mjb'),
    'hardware_case': asdict(selected) if args.hardware_case else None,
    'physics_dt': sim.model.opt.timestep, 'policy_dt': sim.dt,
    'command_delay_ms': sim.command_lag * sim.model.opt.timestep * 1000,
    'position_delay_ms': sim.obs_lag * sim.dt * 1000,
    'velocity_delay_ms': sim.velocity_lag * sim.dt * 1000,
    'contract': sim.contract,
}, indent=2), encoding='utf-8')
font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 16)
sim.model.vis.global_.offwidth = 960
sim.model.vis.global_.offheight = 720
renderer = mujoco.Renderer(sim.model, height=720, width=960)
camera = mujoco.MjvCamera()
camera.azimuth = 135
camera.elevation = -15
camera.distance = .8
writer = imageio.get_writer(str(out / 'rollouts.mp4'), fps=25, codec='libx264', quality=8)
scenarios = [('Stand: all commands zero', 0., 0., 0.),
             ('Forward: vx=0.55 m/s', .55, 0., 0.),
             ('Turn: yaw_rate=0.9 rad/s', 0., .9, 0.),
             ('Tail up: vx=0.35 m/s, tail_pitch=1.57 rad', .35, 0., 1.57)]
samples = []
try:
    for index, (label, vx, wz, tail) in enumerate(scenarios):
        sim.reset()
        cmd = np.zeros(18)
        cmd[0], cmd[2], cmd[16] = vx, wz, tail
        for _ in range(100):
            sim.step(cmd)
        total = round(args.seconds / sim.dt)
        for frame in range(total):
            sim.step(cmd)
            if frame % 2:
                continue
            xyz = sim.data.xpos[sim.body]
            camera.lookat[:] = xyz + [0, 0, .045]
            renderer.update_scene(sim.data, camera=camera)
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            draw.rectangle((0, 0, 960, 65), fill=(12, 18, 28))
            draw.text((18, 8), label, fill='white', font=font)
            metric = sim.sample()
            draw.text((18, 31), f'v07 sim | vx={metric[4]:+.3f} m/s | tilt={metric[6]:.1f} deg | '
                      f'cmd/fbk delay={sim.command_lag*sim.model.opt.timestep*1000:g}/{sim.obs_lag*sim.dt*1000:g} ms | t={frame*sim.dt:.2f}s',
                      fill='white', font=font)
            writer.append_data(np.asarray(img))
            if frame == 0 or abs(frame - total // 2) <= 1:
                img.save(out / f'scene_{index}_{frame:03d}.png')
            if abs(frame - total // 2) <= 1:
                samples.append(img.copy())
finally:
    writer.close()
    renderer.close()
grid = Image.new('RGB', (960, 720))
for i, img in enumerate(samples[:4]):
    grid.paste(img.resize((480, 360)), ((i % 2) * 480, (i // 2) * 360))
grid.save(out / 'overview.png')
print(out / 'rollouts.mp4')
