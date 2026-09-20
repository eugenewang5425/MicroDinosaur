"""Audit finished specialist runs and their independent CPU confirmations."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from run_jump_cfg import OUT,ROOT,SOURCE
from evaluate_policy import sha

RUNS=Path('D:/microduck_rl/logs/rsl_rl')


def main():
    summaries=[];audits={};groups=[]
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)
    anchor=np.load(ROOT/'20260914_transition_refine/v7_anchor.npz')['obs'][::100][:100].copy()
    obs=np.tile(anchor,(3,1));obs[:,63]=np.repeat([0.,.6,.8],len(anchor))
    obs[:,72]=np.tile(np.resize(np.array([-.03,.02,0.],np.float32),len(anchor)),3)
    for skill,iters in [('jump',401),('run',301)]:
        run=RUNS/f'microdinosaur_{skill}_specialist'/f'20260914_train_512x{iters}'
        p=json.loads((run/'run_provenance.json').read_text())
        assert p['status']=='COMPLETE'
        checkpoint=torch.load(p['final_checkpoint'],map_location='cpu',weights_only=False)
        rates=[g['lr'] for g in checkpoint['optimizer_state_dict']['param_groups']]
        assert all(abs(x-1e-4)<1e-12 for x in rates)
        assert all(torch.isfinite(v).all() for group in ['actor_state_dict','critic_state_dict']
                   for v in checkpoint[group].values() if torch.is_tensor(v))
        session=ort.InferenceSession(str(run/'candidate.onnx'),providers=['CPUExecutionProvider'])
        assert session.get_inputs()[0].shape==[1,81] and session.get_outputs()[0].shape==[1,19]
        assert np.isfinite(session.run(None,{'obs':np.zeros((1,81),np.float32)})[0]).all()
        delta=checkpoint['infos']['env_state']['common_step_counter']-source['infos']['env_state']['common_step_counter']
        assert delta==24*iters
        increments={float(v['step']-source['optimizer_state_dict']['state'][i]['step'])
                    for i,v in checkpoint['optimizer_state_dict']['state'].items()}
        assert increments=={20*iters}
        weights=checkpoint['actor_state_dict'];x=torch.tensor(obs)
        x=(x-weights['obs_normalizer._mean'])/(weights['obs_normalizer._std']+.01)
        for i in [0,2,4,6]:
            x=torch.nn.functional.linear(x,weights[f'mlp.{i}.weight'],weights[f'mlp.{i}.bias'])
            if i<6:x=torch.nn.functional.elu(x)
        y=np.concatenate([session.run(None,{'obs':row[None]})[0] for row in obs])
        export_error=float(np.max(np.abs(y-x.numpy())))
        assert export_error<2e-5
        event=EventAccumulator(str(run),size_guidance={'scalars':0});event.Reload();tags=event.Tags()['scalars']
        nonfinite=[];positive_cost=[];last={}
        for tag in tags:
            values=event.Scalars(tag)
            if values:last[tag]=values[-1].value
            if any(not np.isfinite(x.value) for x in values):nonfinite.append(tag)
            if tag.startswith('Episode_Reward/') and any(k in tag for k in ['action_rate','dof_pos_limits','self_collisions','translation','foot_slip']):
                if any(x.value>1e-7 for x in values):positive_cost.append(tag)
        assert not nonfinite and not positive_cost
        audits[skill]=dict(run=str(run),iterations=iters,envs=512,transitions=512*24*iters,
            checkpoint_sha256=sha(p['final_checkpoint']),onnx_sha256=sha(run/'candidate.onnx'),
            optimizer_rates=rates,all_scalars_finite=True,penalty_signs_valid=True,last_scalars=last,
            verified_env_steps=delta,optimizer_step_increments=sorted(increments),
            export_max_error=export_error,export_test_rows=len(obs),
            trained_actor_shape=[81,19],head_control_in_training=p['head_control_in_gpu_training'])
    for label in ['source','source_stress','jump','jump_return','jump_return_fresh','run']:
        records=json.loads((OUT/'evaluation'/label/'matrix.json').read_text())
        for skill in ([] if label=='source_stress' else ['run','jump']):
            speeds=[.45,.6,.8] if skill=='run' else [.6]
            for speed in speeds:
                selected=[r for r in records if r['skill']==skill and r['speed']==speed and r['seed'] in [101,102,103,201,202,203]]
                if not selected:continue
                good=[r['metrics'] for r in selected if r['status']=='COMPLETE']
                mean=lambda key:statistics.mean(m[key] for m in good) if good else None
                row=dict(label=label,skill=skill,command_speed=speed,planned=len(selected),
                    completed=len(good),startup_or_execution_failures=len(selected)-len(good),
                    motion_failures=sum(m['motion_failed'] for m in good),
                    jump_success=sum(m['jump_success'] for m in good),
                    running_with_flight=sum(m['running_with_flight'] for m in good),
                    mean_moving_speed=mean('moving_speed_m_s'),mean_camera_yaw_rms_deg=mean('camera_heading_error_rms_deg'),
                    mean_camera_pitch_rms_deg=mean('camera_pitch_rms_deg'),
                    maximum_clear_flight_s=max([m['max_clear_flight_s'] for m in good],default=0),
                    settled=sum(m['settled_final_second'] for m in good),
                    max_joint_limit_excess=max([m['joint_limit_excess_rad'] for m in good],default=0))
                groups.append(row)
        for r in records:
            m=r.get('metrics',{})
            summaries.append(dict(label=label,name=r['name'],status=r['status'],error=r.get('error'),
                skill=r['skill'],speed=r['speed'],delay_ms=r['hardware']['command_ms'],seed=r['seed'],
                dt=r['hardware']['physics_dt'],motor_curve=r['hardware']['motor_curve'],voltage=r['hardware']['voltage'],
                moving_speed=m.get('moving_speed_m_s'),flight_s=m.get('max_clear_flight_s'),
                jump=m.get('jump_success'),running=m.get('running_with_flight'),motion_failed=m.get('motion_failed'),
                settled=m.get('settled_final_second'),peak_ground_force=m.get('peak_ground_force_n'),
                peak_torque=m.get('peak_joint_torque_nm'),limit_excess=m.get('joint_limit_excess_rad'),
                camera_yaw_rms=m.get('camera_heading_error_rms_deg'),camera_pitch_rms=m.get('camera_pitch_rms_deg')))
    (OUT/'training_audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps(dict(groups=groups,conditions=summaries),indent=2),encoding='utf-8')
    with (OUT/'conditions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)

    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for label in ['source','run']:
        g=[r for r in groups if r['label']==label and r['skill']=='run']
        axes[0,0].plot([r['command_speed'] for r in g],[r['mean_moving_speed'] for r in g],marker='o',label=label)
    axes[0,0].plot([.4,.8],[.4,.8],ls='--',color='gray',label='command')
    axes[0,0].set(xlabel='Command speed (m/s)',ylabel='Measured speed (m/s)',title='Running task: completed conditions only');axes[0,0].legend()
    for label in ['source','jump']:
        f=OUT/'evaluation'/label/'jump_v0.6_d10_s101_dt0.00125_curve0_V12.6.npz'
        a=np.load(f)['physics']
        axes[0,1].plot(a[:,0],np.min(a[:,14:16],axis=1)*1000,label=label)
        axes[1,0].plot(a[:,0],a[:,4],label=label)
        axes[1,1].plot(a[:,0],a[:,1]*1000,label=label)
    axes[0,1].axhline(2,color='gray',ls='--');axes[0,1].set(title='Jump: both-foot minimum mesh clearance',xlabel='Time (s)',ylabel='Clearance (mm)')
    axes[1,0].set(title='Jump: total robot-ground support force',xlabel='Time (s)',ylabel='Force (N)')
    axes[1,1].set(title='Jump: trunk height is not jump height',xlabel='Time (s)',ylabel='Root height (mm)')
    for ax in [axes[0,1],axes[1,0],axes[1,1]]:ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    original=json.loads((OUT/'physics_precheck.json').read_text())['production_sha_before']
    assert sha('D:/microduck_rl/microdinosaur_p2.onnx')==original
    print(json.dumps(dict(groups=groups,training={k:{n:v[n] for n in ['iterations','transitions','onnx_sha256']} for k,v in audits.items()},production_unchanged=True),indent=2))


if __name__=='__main__':main()
