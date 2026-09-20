"""Three-arm low-speed comparison and full decoder verification."""
import json
import argparse
from pathlib import Path
import numpy as np
import imageio.v2 as imageio


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--marked-steps',action='store_true');args=parser.parse_args()
    base=Path(__file__).parent/'20260914_demonstrations'/('videos_marked' if args.marked_steps else 'videos')
    width,height=(960,816) if args.marked_steps else (640,576)
    arms=('legacy','archive','demonstration');readers=[]
    for arm in arms:
        r=json.loads((base/arm/'slow/verification.json').read_text());assert r['common_metrics_match'] and r['frames']==300
        readers.append(imageio.get_reader(str(base/arm/'slow/simulation.mp4')))
    writer=imageio.get_writer(str(base/'low_speed_comparison.mp4'),fps=25,codec='libx264',quality=8);count=0
    try:
        for frames in zip(*readers,strict=True):writer.append_data(np.concatenate(frames,axis=1));count+=1
    finally:
        writer.close()
        for r in readers:r.close()
    assert count==300
    left=imageio.get_reader(str(base/'demonstration/slow/simulation.mp4'))
    right=imageio.get_reader(str(base/'recovery/recovery/simulation.mp4'))
    writer=imageio.get_writer(str(base/'recovery_comparison.mp4'),fps=25,codec='libx264',quality=8);count=0
    try:
        for a,b in zip(left,right,strict=True):writer.append_data(np.concatenate((a,b),axis=1));count+=1
    finally:left.close();right.close();writer.close()
    assert count==300
    paths={'low_speed_comparison.mp4':(300,3*width),
        'recovery/recovery/simulation.mp4':(300,width),'recovery_comparison.mp4':(300,2*width)}
    if not args.marked_steps:paths.update({'expert/expert/simulation.mp4':(300,width),'demonstration/resume/simulation.mp4':(500,width)})
    paths.update({f'{arm}/slow/simulation.mp4':(300,width) for arm in arms});results={}
    for path,(expected,expected_width) in paths.items():
        reader=imageio.get_reader(str(base/path));meta=reader.get_meta_data();decoded=0
        try:
            for frame in reader:assert frame.shape==(height,expected_width,3);decoded+=1
        finally:reader.close()
        assert decoded==expected and meta['fps']==25
        results[path]=dict(decoded_frames=decoded,fps=meta['fps'],size=meta['size'])
    (base/'decoder_verification.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))


if __name__=='__main__':main()
