"""Exact replay of preselected expert and student clips."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from demonstration_expert import OUT,experiment as expert_experiment,POLICY
from evaluate_owned_terrain import cases,key,experiment,evaluate
from terrain_skill_eval import ground_height
from evaluate_policy import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--scene',choices=('slow','resume','expert','recovery'),required=True)
    p.add_argument('--policy',type=Path);p.add_argument('--marked-steps',action='store_true');a=p.parse_args()
    if a.marked_steps:
        if a.scene=='resume':p.error('--marked-steps requires a stair scene')
        import step_render_overlay as overlay
    if a.scene=='expert':
        selected=json.loads((OUT/'bounded_probe/selection.json').read_text());drive=selected['drive'];margin=selected['margin']
        name=f'drive{drive}_margin{margin}_lag10_s11';expected=json.loads((OUT/f'bounded_collect/{name}.json').read_text())
        e,_=expert_experiment(drive,10,margin);seed=11;seconds=12;policy=POLICY
    elif a.scene=='recovery':
        from probe_expert_recovery import Switch,STUDENT
        case=next(c for c in cases() if c['terrain']=='steps_10' and c['speed']==.2)
        e=experiment(STUDENT,case);unused,expert=expert_experiment(.5,10,.03);del unused
        e.sim.session=Switch(e.sim.session,expert);seed=1;seconds=12;policy=STUDENT;name='student_6s_then_expert_seed1'
        expected=json.loads((OUT/'recovery_probe/s1.json').read_text())
        expected['policy_sha256']=json.loads((OUT/'recovery_probe/plan.json').read_text())['policy_sha256']
    else:
        case=next(c for c in cases() if c['terrain']=='steps_10' and c['speed']==.2) if a.scene=='slow' else next(c for c in cases() if c['program']=='resume' and c['delay']==10)
        name=key(case,1);expected=json.loads((OUT/f'evaluation/{a.label}/{name}.json').read_text())
        assert a.policy is not None;e=experiment(a.policy,case);seed=1;seconds=case['seconds'];policy=a.policy
    assert expected['policy_sha256']==sha(policy)
    video_folder='videos_marked' if a.marked_steps else 'videos'
    dest=OUT/f'{video_folder}/{a.label}/{a.scene}';dest.mkdir(parents=True,exist_ok=False)
    width,height=(overlay.WIDTH,overlay.SCENE_HEIGHT) if a.marked_steps else (640,448)
    s=e.sim;s.model.vis.global_.offwidth=width;s.model.vis.global_.offheight=height
    renderer=mujoco.Renderer(s.model,height=height,width=width)
    camera=mujoco.MjvCamera();camera.azimuth=125;camera.elevation=-12;camera.distance=.75
    if a.marked_steps:camera.azimuth=115;camera.elevation=-24;camera.distance=.90
    writer=imageio.get_writer(str(dest/'simulation.mp4'),fps=25,codec='libx264',quality=8)
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',16)
    original=s.step;count=[0];zero=[None];failed=[None];last=[None]
    def step(command,*args,**kwargs):
        if e.active and zero[0] is None:zero[0]=s.view(s.data.time)[4]
        original(command,*args,**kwargs);t=s.data.time-6.
        if t<=1e-8 or round(t/s.dt)%2==0:return
        v=s.view(s.data.time);height=v[3]-float(ground_height(e.terrain_kind,v[1],v[2]))
        if failed[0] is None and (height<.055 or v[14]>60):failed[0]=t
        camera.lookat[:]=s.data.xpos[s.body]+[0,0,.015]
        if a.marked_steps:camera.lookat[:]=[max(.40,v[1]-.04),0.,.09]
        renderer.update_scene(s.data,camera=camera)
        if a.marked_steps:overlay.mark_scene(renderer,s.model)
        scene=Image.fromarray(renderer.render())
        if failed[0] is not None and last[0] is not None:scene=last[0]
        else:last[0]=scene.copy()
        frame=Image.new('RGB',(640,576),'#111a28');frame.paste(scene,(0,64));d=ImageDraw.Draw(frame)
        phase='10mm STEPS / DESIRED 0.20m/s' if a.scene!='resume' else ('STAND' if t<2 else ('CROUCH' if t<7 else ('RETURN' if t<11 else ('WALK' if t<17 else 'STOP'))))
        d.text((12,10),f'{a.label} | {phase}',font=font,fill='white')
        detail=f'expert internal {drive:.2f}; hip margin {margin:.2f}' if a.scene=='expert' else ('POST-HOC: '+('STUDENT' if t<6 else 'EXPERT TAKES OVER') if a.scene=='recovery' else 'student policy / unchanged IMU yaw control')
        d.text((12,36),detail,font=font,fill='#bfcede')
        yaw=(v[5]-zero[0]-e.controller.reference+np.pi)%(2*np.pi)-np.pi
        d.text((12,514),f't={t:5.2f}s | clearance {height*1000:5.1f}mm | vx {v[13]:+.3f}',font=font,fill='white')
        d.text((12,539),f'camera yaw {np.rad2deg(yaw):+.1f} / pitch {np.rad2deg(v[6]):+.1f}deg | 10/20/20ms',font=font,fill='#bfcede')
        if a.marked_steps:
            frame=overlay.frame(np.asarray(scene),a.label,a.scene,t,v[1],v[13]);d=ImageDraw.Draw(frame)
        if failed[0] is not None:d.text((12,80),f'FALL at {failed[0]:.2f}s; scene held',font=font,fill='#ff5050')
        writer.append_data(np.asarray(frame))
        if count[0] in (49,149,249,449):frame.save(dest/f'frame_{count[0]:03d}.png')
        count[0]+=1
    s.step=step
    try:m,_=e.run('straight','imu',seed,seconds,True) if a.scene=='expert' else evaluate(e,case,seed)
    finally:writer.close();renderer.close()
    assert all(json.loads(json.dumps(v))==expected['metrics'][k] for k,v in m.items()),'Replay changed'
    assert count[0]==seconds*25
    if a.marked_steps:
        if a.scene=='recovery':trace_path=OUT/'recovery_probe/s1.npz'
        elif a.scene=='expert':trace_path=OUT/f'bounded_collect/{name}.npz'
        else:trace_path=OUT/f'evaluation/{a.label}/{name}.npz'
        with np.load(trace_path) as z:np.testing.assert_array_equal(np.asarray(s.gaze_trace),z['gaze'])
    result=dict(status='PASS',frames=count[0],fps=25,preselected=a.scene!='recovery',common_metrics_match=True,
        matched_metric_count=len(m),policy_sha256=sha(policy),case=name,fall_time_s=failed[0])
    if a.marked_steps:
        result.update(marked_steps=True,physics_trace_bitwise_match=True,frame_size=[overlay.WIDTH,overlay.HEIGHT],
            source_hashes={n:sha(Path(__file__).parent/n) for n in ('render_demonstrations.py','step_render_overlay.py')})
    (dest/'verification.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':main()
