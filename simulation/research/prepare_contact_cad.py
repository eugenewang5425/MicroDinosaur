"""Build actual simulation body transforms for sampled full-CAD checks."""
from pathlib import Path
import json,argparse
import numpy as np
import mujoco
from evaluate_policy import sha
OUT=Path(__file__).parent/'20260914_contact_motion'
BASE=OUT.parent/'20260914_squat_specialist'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folders',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    dest=OUT/a.out;dest.mkdir(exist_ok=False)
    selected=json.loads((OUT/'selected_contact.json').read_text());plant=OUT/selected['plant_directory']
    m=mujoco.MjModel.from_binary_path('x',assets={'x':(plant/'nominal.mjb').read_bytes()});d=mujoco.MjData(m)
    frames=[json.loads((BASE/'reference_geometry/cad_pose_frames.json').read_text())[0]];sources=[]
    ankles=[m.jnt_qposadr[m.joint('robot/'+n).id] for n in ('left_ankle','right_ankle')]
    jaw=m.jnt_qposadr[m.joint('robot/jaw_hinge').id]
    for folder in a.folders:
      for path in (OUT/folder).glob('*.npz'):
        z=np.load(path)
        if 'qpos' not in z:continue
        qs=z['qpos'];indices={len(qs)-1,int(np.argmax(abs(qs[:,ankles]).max(-1))),int(np.argmin(qs[:,jaw]))}
        if 'bank' in folder:indices.update(range(len(qs)))
        if 'fall_preparation' not in folder:indices.update(np.linspace(0,len(qs)-1,5).astype(int))
        for k in sorted(indices):
            d.qpos[:]=qs[k];mujoco.mj_forward(m,d);bodies={}
            for b in range(m.nbody):
                name=m.body(b).name
                if not name.startswith('robot/'):continue
                t=np.eye(4);t[:3,:3]=d.xmat[b].reshape(3,3);t[:3,3]=d.xpos[b]
                bodies[name.removeprefix('robot/')]=t.tolist()
            frames.append(dict(name=f'{folder}_{path.stem}_f{k}',bodies=bodies))
        sources.append(dict(path=str(path),sha256=sha(path),indices=[int(i) for i in sorted(indices)]))
    (dest/'cad_pose_frames.json').write_text(json.dumps(frames))
    (dest/'sources.json').write_text(json.dumps(dict(plant_sha256=sha(plant/'nominal.mjb'),sources=sources),indent=2))
    print(len(frames),'frames')
