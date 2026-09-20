"""Verify paired sources, normalizers, training counters and independent frozen inputs."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import onnxruntime as ort
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from evaluate_policy import sha
from mjlab_microduck.tasks.symmetry_microdinosaur import _JOINT_PERM,_JOINT_SIGN,_OBS_PERM,_OBS_SIGN

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_head_gaze_ablation'
RELEASE=Path('D:/microduck_rl/microdinosaur_p2.onnx')
RELEASE_HASH='0804114efd1e457ec2c297d709c0464fd0c2bfe251cbb36db7de74543857b0e3'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def audit_job(job,plan,bank,mirror):
    run=Path(job['run']);p=read(run/'run_provenance.json')
    assert p['status']=='COMPLETE' and p['finite_output_check']
    assert p['source_checkpoint_sha256']==plan['source_sha256']
    assert p['seed']==job['seed'] and p['envs']==1024 and p['iterations']==101
    assert p['delay_coverage']=='original' and p['delay_contract_before']==p['delay_contract_after']
    assert p['hardware_profile']=='s288-protocol' and p['action_filter_alpha_new']==.9
    assert p['same_81D_actor_contract'] and not p['head_IMU_actor_input']
    assert p['microdino_probe_environment']==''
    assert p['effective_head_gaze_weight']==(-.2 if job['arm']=='control' else 0.)
    assert p['head_world_gaze_weight']==1.5
    for src in p['source_files']:
        assert sha(src['snapshot'])==src['sha256']==plan['source_hashes'][src['path']]
    ckpt=Path(p['final_checkpoint']);assert sha(ckpt)==p['final_checkpoint_sha256']
    d=torch.load(ckpt,map_location='cpu',weights_only=False)
    assert d['iter']==15600 and d['infos']['env_state']['common_step_counter']==374568
    normalizer_keys=[k for k in d['actor_state_dict'] if 'normalizer' in k or 'normalization' in k or 'normaliz' in k]
    assert normalizer_keys, list(d['actor_state_dict'])
    def finite_tree(value):
        if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
        elif isinstance(value,dict):
            for v in value.values():finite_tree(v)
        elif isinstance(value,(list,tuple)):
            for v in value:finite_tree(v)
    finite_tree(d)
    cfg=yaml.load((run/'params/env.yaml').read_text(encoding='utf-8'),Loader=yaml.BaseLoader)
    assert 'lean_drift' not in cfg['rewards'] and 'contact_timing' not in cfg['rewards']
    assert float(cfg['rewards']['head_pose_tracking']['weight'])==.8
    assert float(cfg['rewards']['body_ang_vel']['weight'])==-.05
    events=EventAccumulator(str(run),size_guidance={'scalars':0});events.Reload();scalars={}
    for tag in events.Tags()['scalars']:
        values=np.array([e.value for e in events.Scalars(tag)])
        assert np.isfinite(values).all(),tag
        scalars[tag]={'count':len(values),'min':float(values.min()),'max':float(values.max()),
            'last':float(values[-1]),'last_10_mean':float(values[-10:].mean())}
        if tag.startswith('Episode_Reward/'):
            term=tag.split('/',1)[1]
            weight=float(cfg['rewards'].get(term,{}).get('weight',0))
            if weight<0:assert values.max()<=1e-7,tag
    assert scalars['Loss/value']['count']==101
    assert scalars['Episode_Termination/nan_state']['max']==0
    gaze=scalars.get('Episode_Reward/head_gaze')
    if job['arm']=='control':assert gaze is not None and gaze['min']<0
    else:assert gaze is None or gaze['max']==gaze['min']==0
    onnx=Path(p['onnx']);options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(onnx),sess_options=options,providers=['CPUExecutionProvider'])
    predict=lambda obs:np.concatenate([session.run(None,{'obs':x[None]})[0] for x in obs])
    actions=predict(bank);mirrored=predict(mirror)
    assert actions.shape==mirrored.shape==(120,19) and np.isfinite(actions).all() and np.isfinite(mirrored).all()
    residual=mirrored-actions[:,_JOINT_PERM]*np.array(_JOINT_SIGN)
    legs=[0,1,2,3,4,10,11,12,13,14]
    contract={'candidate_onnx':str(onnx),'onnx_sha256':sha(onnx),
        'training_seed':job['seed'],'head_gaze_weight':p['effective_head_gaze_weight'],
        'input_output':[81,19],'normalizer_in_official_export':True,
        'external_execution_required':{'profile':'s288-protocol','policy_period_s':.02,
            'host_action_filter_alpha':.9,'host_action_max_delta':.6,
            'nominal_kp':7.,'nominal_kd':.8,'gain_encoding':'S288 firmware rounding',
            'training_delay_ms':plan['training_delay_ms'],
            'nominal_plant':str(ROOT/'20260913_handoff/native_v07/nominal.mjb'),
            'nominal_plant_sha256':sha(ROOT/'20260913_handoff/native_v07/nominal.mjb'),
            'implementation':'research/hardware_sim.py + mjlab_microduck/s288_protocol.py'},
        'head_gyro_actor_input':False,'approved_for_real_hardware':False,'automatic_promotion':False,
        'reference_run_provenance':str(run/'run_provenance.json')}
    contract_path=OUT/'candidate_contracts'/f's{job["seed"]}_{job["arm"]}.json'
    contract_path.parent.mkdir(exist_ok=True);contract_path.write_text(json.dumps(contract,indent=2),encoding='utf-8')
    return {'seed':job['seed'],'arm':job['arm'],'run':str(run),'status':'PASS',
        'onnx_sha256':sha(onnx),'checkpoint_sha256':sha(ckpt),'checkpoint_iter':d['iter'],
        'common_step_counter':374568,'new_transitions':1024*101*24,'normalizer_state_keys':normalizer_keys,
        'all_checkpoint_tensors_finite':True,'all_logged_scalars_finite':True,
        'negative_weight_reward_logs_nonpositive':True,'scalars':scalars,
        'mirror_bank_sha256':sha(ROOT/'20260913_direction_bias/shared_observation_bank.npz'),
        'shared_observation_count':120,'all_action_mirror_residual_rms':float(np.sqrt(np.mean(residual**2))),
        'leg_action_mirror_residual_rms':float(np.sqrt(np.mean(residual[:,legs]**2))),
        'mirror_residual_is_policy_response_not_physical_symmetry':True,'execution_contract':str(contract_path)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    plan=read(OUT/'plan.json');status=read(OUT/'train_status.json')
    if not args.partial:assert status['status']=='COMPLETE' and len(status['jobs'])==6
    assert sha(plan['source_checkpoint'])==plan['source_sha256']
    assert sha(RELEASE)==RELEASE_HASH
    for path,expected in plan['source_hashes'].items():assert sha(path)==expected,path
    raw=np.load(ROOT/'20260913_direction_bias/shared_observation_bank.npz')
    bank,mirror=raw['original'],raw['mirrored']
    np.testing.assert_array_equal(bank[:,_OBS_PERM]*np.array(_OBS_SIGN),mirror)
    jobs=[audit_job(j,plan,bank,mirror) for j in status['jobs'] if j['status']=='COMPLETE']
    video=read(OUT/'video_seed42/contract.json');reader=imageio.get_reader(str(OUT/'video_seed42/forward_pair.mp4'))
    meta=reader.get_meta_data();frames=reader.count_frames();reader.close()
    assert frames==video['frame_count']==150 and meta['fps']==25 and meta['size']==(1280,720)
    report={'status':'PARTIAL_PASS' if args.partial else 'PASS','jobs':jobs,
        'new_training_transitions':sum(j['new_transitions'] for j in jobs),
        'production_onnx_unchanged_sha256':sha(RELEASE),'watched_training_sources_unchanged':True,
        'video':{'frame_count':frames,'fps':meta['fps'],'size':meta['size'],
            'final_state_identical_to_saved_evaluation':video['rendered_endpoint_max_error_vs_saved_trace']},
        'versions':{p:importlib.metadata.version(p) for p in ('torch','mjlab','mujoco','mujoco-warp','rsl-rl-lib','onnxruntime')}}
    path=OUT/('preview_audit.json' if args.partial else 'audit.json')
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'jobs':len(jobs),'new_transitions':report['new_training_transitions'],
        'mirror':[{k:j[k] for k in ('seed','arm','leg_action_mirror_residual_rms')} for j in jobs]},indent=2))


if __name__=='__main__':main()
