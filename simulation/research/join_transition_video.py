"""Side-by-side preselected clips and full decoder verification."""
import json
import imageio.v2 as imageio
import numpy as np
from pathlib import Path


def main():
    base=Path(__file__).parent/'20260914_transition_refine/videos'
    for label in ('previous','candidate'):
        r=json.loads((base/label/'verification.json').read_text());assert r['metrics_match'] and r['frames']==500
    a=imageio.get_reader(str(base/'previous/simulation.mp4'));b=imageio.get_reader(str(base/'candidate/simulation.mp4'))
    writer=imageio.get_writer(str(base/'comparison.mp4'),fps=25,codec='libx264',quality=8)
    count=0
    try:
        for left,right in zip(a,b,strict=True):
            writer.append_data(np.concatenate((left,right),axis=1));count+=1
    finally:a.close();b.close();writer.close()
    assert count==500
    results={}
    for name in ('comparison.mp4','candidate_steps10_failure/simulation.mp4'):
        r=imageio.get_reader(str(base/name));metadata=r.get_meta_data();decoded=0
        try:
            for frame in r:assert frame.shape[0]==576;decoded+=1
        finally:r.close()
        assert decoded==(500 if name=='comparison.mp4' else 300) and metadata['fps']==25
        results[name]=dict(decoded_frames=decoded,fps=metadata['fps'],size=metadata['size'])
    (base/'decoder_verification.json').write_text(json.dumps(results,indent=2));print(results)


if __name__=='__main__':main()
