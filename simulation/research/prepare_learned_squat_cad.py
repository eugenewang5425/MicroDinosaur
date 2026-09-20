"""Audit measured learned poses, including maximum ankle flexion and jaw closure."""
from pathlib import Path
import json
import mujoco
import numpy as np
from evaluate_policy import sha
OUT=Path(__file__).resolve().parent/'20260914_squat_specialist'
m=mujoco.MjModel.from_binary_path('x',assets={'x':(OUT/'plant/nominal.mjb').read_bytes()});d=mujoco.MjData(m)
frames=[json.loads((OUT/'reference_geometry/cad_pose_frames.json').read_text())[0]]
sources=[]
ankles=[m.jnt_qposadr[m.joint('robot/'+n).id] for n in ('left_ankle','right_ankle')]
jaw=m.jnt_qposadr[m.joint('robot/jaw_hinge').id]
for folder in ('evaluation_residual','confirmation_residual'):
    for row in json.loads((OUT/folder/'summary.json').read_text()):
        if row['status']!='COMPLETE':continue
        path=OUT/folder/(row['key']+'.npz');z=np.load(path);q=z['qpos']
        indices={round(t/.04)-1 for t in (1,3,5,8,10,13)}
        indices.add(int(np.argmax(np.abs(q[:,ankles]).max(-1))))
        indices.add(int(np.argmin(q[:,jaw])))
        for k in sorted(indices):
            d.qpos[:]=q[k];mujoco.mj_forward(m,d);bodies={}
            for b in range(m.nbody):
                n=m.body(b).name
                if not n.startswith('robot/'):continue
                t=np.eye(4);t[:3,:3]=d.xmat[b].reshape(3,3);t[:3,3]=d.xpos[b]
                bodies[n.removeprefix('robot/')]=t.tolist()
            frames.append(dict(name=f'{folder}_{row["key"]}_f{k}',bodies=bodies))
        sources.append(dict(path=str(path),sha256=sha(path),frames=sorted(indices)))
(OUT/'cad_pose_frames.json').write_text(json.dumps(frames),encoding='utf-8')
(OUT/'learned_geometry_sources.json').write_text(json.dumps(dict(plant_sha256=sha(OUT/'plant/nominal.mjb'),
    frames=len(frames),sources=sources,sampling='6 phase samples plus ankle/jaw extrema per executed trial; not a continuous-time proof'),indent=2))
print(len(frames),'frames',len(sources),'trials')
