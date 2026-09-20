"""Annotated playback from saved nominal physics; never re-simulates a policy."""
from types import SimpleNamespace
import json
import hashlib

import imageio.v2 as imageio
import mujoco
import numpy as np

from evaluate_run_jump import OUT,render


def main():
    audit=[]
    jobs=[('jump_return','jump',.6,'jump_return_slow.mp4',10,42),
          ('run','run',.8,'fast_walk.mp4',25,99)]
    for label,skill,speed,video,fps,preview_index in jobs:
        path=OUT/'evaluation'/label/f'{skill}_v{speed}_d10_s101_dt0.00125_curve0_V12.6.npz'
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        saved=np.load(path)
        model=mujoco.MjModel.from_binary_path('nominal.mjb',
                    {'nominal.mjb':(OUT/'plant/nominal.mjb').read_bytes()})
        e=SimpleNamespace(sim=SimpleNamespace(model=model,body=model.body('robot/trunk_base').id,
                            case=SimpleNamespace(command_ms=10)),skill=skill,speed=speed,
                          qpos_frames=saved['qpos'],rows=saved['physics'],video_fps=fps)
        if label=='jump_return':e.return_time=2.8
        dest=path.parent/video
        render(e,dest)
        reader=imageio.get_reader(str(dest))
        imageio.imwrite(str(OUT/(skill+'_final_preview.png')),reader.get_data(preview_index))
        metadata=reader.get_meta_data();reader.close()
        after=hashlib.sha256(path.read_bytes()).hexdigest();assert before==after
        audit.append(dict(input=str(path),input_sha256=before,physics_unchanged=True,
                          video=str(dest),fps=metadata['fps'],duration_s=metadata['duration'],
                          playback_rate=fps/25,preview_frame_index=preview_index))
    (OUT/'video_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
