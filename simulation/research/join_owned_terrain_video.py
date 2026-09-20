"""Join preselected before/after replays and verify every decoded frame."""
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np


def main():
    base=Path(__file__).parent/'20260914_imu_owned_terrain/videos'
    results={}
    for scene,expected in (('steps',300),('resume',500)):
        for label in ('frozen_yaw','trained_yaw'):
            r=json.loads((base/label/scene/'verification.json').read_text())
            assert r['metrics_match'] and r['preselected'] and r['frames']==expected
        left=imageio.get_reader(str(base/'frozen_yaw'/scene/'simulation.mp4'))
        right=imageio.get_reader(str(base/'trained_yaw'/scene/'simulation.mp4'))
        name=f'{scene}_comparison.mp4'
        writer=imageio.get_writer(str(base/name),fps=25,codec='libx264',quality=8)
        count=0
        try:
            for a,b in zip(left,right,strict=True):
                assert a.shape==b.shape==(576,640,3)
                writer.append_data(np.concatenate((a,b),axis=1));count+=1
        finally:left.close();right.close();writer.close()
        assert count==expected
        for relative in (name,f'frozen_yaw/{scene}/simulation.mp4',f'trained_yaw/{scene}/simulation.mp4'):
            reader=imageio.get_reader(str(base/relative));meta=reader.get_meta_data();decoded=0
            try:
                for frame in reader:
                    assert frame.shape==(576,1280 if relative==name else 640,3)
                    decoded+=1
            finally:reader.close()
            assert decoded==expected and meta['fps']==25
            results[relative]=dict(decoded_frames=decoded,fps=meta['fps'],size=meta['size'])
    diagnostic='trained_yaw/slow_steps/simulation.mp4'
    verification=json.loads((base/'trained_yaw/slow_steps/verification.json').read_text())
    assert verification['metrics_match'] and not verification['preselected']
    reader=imageio.get_reader(str(base/diagnostic));meta=reader.get_meta_data();decoded=0
    try:
        for frame in reader:
            assert frame.shape==(576,640,3);decoded+=1
    finally:reader.close()
    assert decoded==300 and meta['fps']==25
    results[diagnostic]=dict(decoded_frames=decoded,fps=meta['fps'],size=meta['size'])
    (base/'decoder_verification.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
