"""Preselected seed 1, body-loop-only versus body+head IMU feedback."""
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from heading_sim import WARMUP_SECONDS
from run_head_attitude import make_experiment, POLICIES
from evaluate_policy import sha

OUT = Path(__file__).parent/'20260914_head_attitude'


def render(policy):
    dest = OUT/'videos'/policy; dest.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 17)
    small = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 14)
    for mode in ('off', 'imu'):
        e = make_experiment(policy, mode); s = e.sim
        s.model.vis.global_.offwidth = 640; s.model.vis.global_.offheight = 576
        renderer = mujoco.Renderer(s.model, height=576, width=640)
        camera = mujoco.MjvCamera(); camera.azimuth = 130; camera.elevation = -15; camera.distance = .70
        writer = imageio.get_writer(str(dest/f'{mode}.mp4'), fps=25, codec='libx264', quality=8)
        original = s.step; count = [0]; zero = [None]
        def step(command, *args, **kwargs):
            original(command, *args, **kwargs)
            if abs(s.data.time-WARMUP_SECONDS) < 1e-8:
                zero[0] = s.view(s.data.time)[4]
            if s.data.time <= WARMUP_SECONDS+1e-8 or round((s.data.time-WARMUP_SECONDS)/s.dt)%2 == 0:
                return
            view = s.view(s.data.time); target = zero[0]+e.controller.reference
            camera.lookat[:] = s.data.xpos[s.body]+[0, 0, .045]
            renderer.update_scene(s.data, camera=camera)
            origin = s.data.site_xpos[s.camera].copy()
            forward = -s.data.site_xmat[s.camera].reshape(3, 3)[:, 2]
            for direction, color in ((np.array([np.cos(target), np.sin(target), 0]), [.15, .75, .35, 1]),
                                     (forward, [1, .4, .1, 1])):
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(color))
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .004, origin, origin+.14*direction)
                renderer.scene.ngeom += 1
            image = Image.new('RGB', (640, 720), (15, 23, 34))
            image.paste(Image.fromarray(renderer.render()), (0, 65)); draw = ImageDraw.Draw(image)
            draw.text((16, 10), policy+' | '+('Body heading only' if mode == 'off' else 'Body + HEAD IMU feedback'), font=font, fill='white')
            draw.text((16, 36), f't={s.data.time-WARMUP_SECONDS:.2f}s | same frozen gait | vx cmd 0.55 m/s', font=small, fill='#c7d1dc')
            draw.text((16, 648), f'Camera yaw error {np.rad2deg(view[5]-target):+.1f} deg | vx {view[13]:.3f}', font=font, fill='white')
            draw.text((16, 678), 'Green: desired forward; orange: actual camera axis', font=small, fill='#c7d1dc')
            draw.text((16, 701), '19 S288 | motor/fbk 10/20/20ms | head IMU +20ms', font=small, fill='#c7d1dc')
            writer.append_data(np.asarray(image)); count[0] += 1
        s.step = step
        try:
            metrics, _ = e.run('straight', 'imu', 1, 12.)
        finally:
            writer.close(); renderer.close()
        expected = json.loads((OUT/'trials'/f'nominal__{policy}__1__straight__{mode}.json').read_text())['metrics']
        # JSON stores dataclass tuple fields as lists; compare canonical records.
        assert json.loads(json.dumps(metrics)) == expected, 'Rendering changed simulation trajectory'
        assert count[0] == 300
    readers = [imageio.get_reader(str(dest/f'{mode}.mp4')) for mode in ('off', 'imu')]
    writer = imageio.get_writer(str(dest/'comparison.mp4'), fps=25, codec='libx264', quality=8)
    try:
        for index, frames in enumerate(zip(*readers)):
            combined = np.concatenate(frames, axis=1); writer.append_data(combined)
            if index in (0, 150, 299):
                Image.fromarray(combined).save(dest/f'frame_{index:03d}.png')
    finally:
        writer.close()
        for reader in readers:
            reader.close()
    (dest/'contract.json').write_text(json.dumps(dict(policy_sha256=sha(POLICIES[policy]), seed=1,
        seconds=12, fps=25, frames=300, size=[1280, 720], metrics_match_confirmation=True), indent=2), encoding='utf-8')
    print(policy, 'render PASS', flush=True)


if __name__ == '__main__':
    for policy in POLICIES:
        render(policy)
