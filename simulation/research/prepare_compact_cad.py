from pathlib import Path
import json
import mujoco
import numpy as np
OUT=Path(__file__).resolve().parent/'20260914_compact_fold'
plant=OUT/'floor_only_plant'
m=mujoco.MjModel.from_binary_path('x',assets={'x':(plant/'nominal.mjb').read_bytes()})
d=mujoco.MjData(m);c=json.loads((plant/'contract.json').read_text());frames=[]
d.qpos[:3]=[0,0,.117182];d.qpos[3:7]=[1,0,0,0]
for name,q in zip(c['action_names'],c['action_offset'][0]):d.qpos[m.jnt_qposadr[m.joint('robot/'+name).id]]=q
def capture(name):
    mujoco.mj_forward(m,d);bodies={}
    for b in range(m.nbody):
        n=m.body(b).name
        if not n.startswith('robot/'):continue
        t=np.eye(4);t[:3,:3]=d.xmat[b].reshape(3,3);t[:3,3]=d.xpos[b]
        bodies[n.removeprefix('robot/')]=t.tolist()
    frames.append(dict(name=name,bodies=bodies))
capture('home')
for depth in (30,40,50):
    trace=np.load(next((OUT/'floor_only_nominal').glob(f'd{depth}_*.npz')))
    for t in ([5] if depth<50 else [1,2,3,4,5,8,9,10,13]):
        d.qpos[:]=trace['qpos'][round(t/.04)-1];capture(f'd{depth}_t{t}')
(OUT/'cad_pose_frames.json').write_text(json.dumps(frames),encoding='utf-8')
