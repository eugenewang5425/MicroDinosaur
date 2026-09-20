"""Join unchanged-time marked replays and fully decode all deliverable videos."""
import json
import imageio.v2 as imageio
import numpy as np
from corrective_common import OUT


def main():
    base=OUT/'videos';arms=('source','control','corrective');readers=[]
    for arm in arms:
        record=json.loads((base/arm/'verification.json').read_text())
        assert record['physics_trace_bitwise_match'] and record['frames']==300
        readers.append(imageio.get_reader(str(base/arm/'simulation.mp4')))
    writer=imageio.get_writer(str(base/'comparison.mp4'),fps=25,codec='libx264',quality=8);count=0
    try:
        for frames in zip(*readers,strict=True):writer.append_data(np.concatenate(frames,axis=1));count+=1
    finally:
        writer.close()
        for reader in readers:reader.close()
    assert count==300;results={}
    for name,width in [('comparison.mp4',2880)]+[(f'{arm}/simulation.mp4',960) for arm in arms]:
        reader=imageio.get_reader(str(base/name));meta=reader.get_meta_data();decoded=0
        try:
            for frame in reader:assert frame.shape==(816,width,3);decoded+=1
        finally:reader.close()
        assert decoded==300 and meta['fps']==25
        results[name]=dict(frames=decoded,fps=25,size=meta['size'])
    (base/'decoder_verification.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))


if __name__=='__main__':main()
