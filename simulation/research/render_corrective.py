"""Marked stair replays of the fixed independent confirmation initial state."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw
import step_render_overlay as overlay
from corrective_common import OUT
from evaluate_owned_terrain import cases,key,experiment,evaluate
from evaluate_policy import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--seed',type=int,default=201);p.add_argument('--delay',type=int,default=10);p.add_argument('--record-folder',type=Path);a=p.parse_args()
    case=dict(next(c for c in cases() if c['terrain']=='steps_10' and c['speed']==.2),delay=a.delay);seed=a.seed
    folder='source_holdout' if a.label=='source' else a.label
    record_folder=a.record_folder or OUT/f'evaluation/{folder}'
    name=key(case,seed);expected=json.loads((record_folder/f'{name}.json').read_text())
    assert sha(a.policy)==expected['policy_sha256']
    dest=OUT/'videos'/a.label;dest.mkdir(parents=True,exist_ok=False)
    e=experiment(a.policy,case);s=e.sim
    s.model.vis.global_.offwidth=overlay.WIDTH;s.model.vis.global_.offheight=overlay.SCENE_HEIGHT
    renderer=mujoco.Renderer(s.model,height=overlay.SCENE_HEIGHT,width=overlay.WIDTH)
    camera=mujoco.MjvCamera();camera.azimuth=115;camera.elevation=-24;camera.distance=.90
    writer=imageio.get_writer(str(dest/'simulation.mp4'),fps=25,codec='libx264',quality=8)
    count=0;original=s.step;failed=None;last=None
    names={'source':'原策略','control':'成功示范对照组','corrective':'卡阶纠正示范组',
        'fitted':'纯示范拟合诊断（未晋级）','fitted_15ms':'纯示范拟合诊断（未晋级）',
        'rehearsal':'补全蹲起回放（未晋级）','rehearsal_5ms':'补全蹲起回放（未晋级）'}
    def step(command,*args,**kwargs):
        nonlocal count,failed,last
        original(command,*args,**kwargs);t=s.data.time-6
        if t<=1e-8 or round(t/s.dt)%2==0:return
        v=s.view(s.data.time)
        from terrain_skill_eval import ground_height
        if failed is None and (v[3]-ground_height('steps_10',v[1],v[2])<.055 or v[14]>60):failed=t
        camera.lookat[:]=[max(.40,v[1]-.04),0.,.09];renderer.update_scene(s.data,camera=camera)
        overlay.mark_scene(renderer,s.model);scene=renderer.render()
        if failed is not None and last is not None:scene=last
        else:last=scene.copy()
        frame=overlay.frame(scene,'demonstration','slow',t,v[1],v[13]);draw=ImageDraw.Draw(frame)
        draw.rectangle((0,0,overlay.WIDTH,42),fill='#111a28')
        initial=f'诊断初态{seed}' if a.record_folder else f'独立初态{seed}'
        draw.text((20,10),f'{names[a.label]}  |  {initial}，{a.delay} ms延迟',font=overlay.FONT,fill='white')
        if failed is not None:draw.text((20,140),f'跌倒于 {failed:.2f}s，画面保留最后有效姿态',font=overlay.FONT,fill='#ff5050')
        writer.append_data(np.asarray(frame))
        if count in (49,149,249):frame.save(dest/f'frame_{count:03d}.png')
        count+=1
    s.step=step
    try:m,_=evaluate(e,case,seed)
    finally:writer.close();renderer.close()
    assert count==300
    assert all(json.loads(json.dumps(v))==expected['metrics'][k] for k,v in m.items())
    with np.load(record_folder/f'{name}.npz') as z:np.testing.assert_array_equal(s.gaze_trace,z['gaze'])
    result=dict(status='PASS',frames=count,physics_trace_bitwise_match=True,policy_sha256=sha(a.policy),seed=seed,case=case,fall_time=failed)
    (dest/'verification.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':main()
