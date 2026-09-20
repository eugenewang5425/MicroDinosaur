"""Bounded confirmation matrix after developing on v7 / seed 0 only."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback
import numpy as np
from evaluate_policy import sha
from hardware_sim import HardwareCase
from heading_sim import HeadingExperiment,ImuTransportConfig,TRACE_COLUMNS,WARMUP_SECONDS
from imu_heading import HeadingConfig,EstimatorConfig

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_imu_heading'
POLICIES={
    'v7':ROOT/'20260913_handoff/v7_reference.onnx',
    's42_no_neck':Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_v07_calibration/20260914_gaze_train_s42_no_neck_cost_1024x101/candidate.onnx'),
}


def definitions():
    cases={'nominal':(HardwareCase(name='nominal',physics_dt=.00125),ImuTransportConfig()),
        'physics_0p625ms':(HardwareCase(name='physics_0p625ms',physics_dt=.000625),ImuTransportConfig()),
        'slower_sensing':(HardwareCase(name='slower_sensing',physics_dt=.00125,command_ms=15,position_ms=40),ImuTransportConfig(period_ms=40,latency_ms=60)),
        'residual_bias_plus':(HardwareCase(name='residual_bias_plus',physics_dt=.00125),ImuTransportConfig(residual_z_bias_deg_s=.5)),
        'imu_outage':(HardwareCase(name='imu_outage',physics_dt=.00125),ImuTransportConfig(outage_duration_s=.2)),
        'physics_5ms':(HardwareCase(name='physics_5ms',physics_dt=.005),ImuTransportConfig()),
        'long_bias_plus':(HardwareCase(name='long_bias_plus',physics_dt=.00125),ImuTransportConfig(residual_z_bias_deg_s=.5)),
        'long_bias_minus':(HardwareCase(name='long_bias_minus',physics_dt=.00125),ImuTransportConfig(residual_z_bias_deg_s=-.5))}
    jobs=[]
    def add(case,policies,seeds,tasks,modes,seconds=12):
        for policy in policies:
            for seed in seeds:
                for task in tasks:
                    for mode in modes:
                        jobs.append({'case':case,'policy':policy,'seed':seed,'scenario':task,'mode':mode,
                            'seconds':8 if task=='stand' else seconds})
    add('nominal',POLICIES,[1,2,3],['stand','straight','left_then_hold','right_then_hold'],['open','imu'])
    for case in ('physics_0p625ms','slower_sensing','residual_bias_plus','imu_outage'):
        add(case,POLICIES,[1,2],['straight'],['open','imu'])
    add('physics_5ms',POLICIES,[1],['straight'],['open','imu'])
    add('nominal',POLICIES,[1],['straight','left_then_hold','right_then_hold'],['oracle'])
    for case in ('long_bias_plus','long_bias_minus'):add(case,['v7'],[1],['straight'],['open','imu'],30)
    assert len(jobs)==94
    return cases,jobs


def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');args=p.parse_args()
    OUT.mkdir(exist_ok=True);(OUT/'traces').mkdir(exist_ok=True);(OUT/'trials').mkdir(exist_ok=True)
    cases,jobs=definitions();controller=HeadingConfig();estimator=EstimatorConfig()
    paths=[ROOT/name for name in ('imu_heading.py','heading_sim.py','run_heading_experiment.py',
        'evaluate_gaze_ablation.py','hardware_sim.py','evaluate_policy.py')]
    source_hashes={p.name:sha(p) for p in paths}
    plan={'status':'FROZEN','attempted_trials':len(jobs),'development_case':'v7 seed 0, nominal 12s straight',
        'confirmation_initial_states':[1,2,3],'policies':{k:{'path':str(v),'sha256':sha(v)} for k,v in POLICIES.items()},
        'controller':asdict(controller),'estimator':asdict(estimator),'warmup_seconds':WARMUP_SECONDS,
        'cases':{k:{'hardware':asdict(h),'outer_imu':asdict(i)} for k,(h,i) in cases.items()},'jobs':jobs,
        'source_hashes':source_hashes,'existing_actor_observations_unchanged':True,
        'outer_imu_mode_uses_truth':False,'oracle_mode_is_explicit_diagnostic':True,
        'head_imu_estimation_only_no_new_head_motor_loop':True,'new_rl_training':False,
        'sensor_noise_bias_and_latency_are_unidentified_sensitivity_settings':True}
    path=OUT/'plan.json'
    if path.exists():assert args.resume and json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    state={'status':'RUNNING','completed':0,'calibration_rejected':0,'total':94,'started_unix':time.time()}
    records=[];status_path=OUT/'status.json'
    def save():status_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    save()
    try:
        for number,job in enumerate(jobs):
            assert {p.name:sha(p) for p in paths}==source_hashes,'Source changed during confirmation'
            key='__'.join(str(job[k]) for k in ('case','policy','seed','scenario','mode'))
            target=OUT/'trials'/f'{key}.json';state['current']=key;save()
            if target.exists():
                assert args.resume;record=json.loads(target.read_text());assert record['source_hashes']==source_hashes
            else:
                hardware,transport=cases[job['case']]
                sim=HeadingExperiment(ROOT/'20260913_handoff/native_v07',POLICIES[job['policy']],hardware,transport,controller,estimator)
                started=time.time()
                try:
                    metrics,trace=sim.run(job['scenario'],job['mode'],job['seed'],job['seconds'],capture=job['seed']==1)
                    record={**job,'status':'COMPLETE','metrics':metrics,'source_hashes':source_hashes,'elapsed_seconds':time.time()-started}
                    if trace is not None:np.savez_compressed(OUT/'traces'/f'{key}.npz',trace=trace,columns=np.array(TRACE_COLUMNS))
                except ValueError as exc:
                    if 'Calibration' not in str(exc):raise
                    record={**job,'status':'CALIBRATION_REJECTED','reason':str(exc),'source_hashes':source_hashes}
                target.write_text(json.dumps(record,indent=2),encoding='utf-8')
            records.append(record);state['completed']=number+1
            state['calibration_rejected']+=record['status']=='CALIBRATION_REJECTED';save()
            print(json.dumps({'completed':number+1,'total':94,'key':key,'status':record['status'],
                'heading_rms':record.get('metrics',{}).get('heading_error_rms_deg'),'fell':record.get('metrics',{}).get('fell')},ensure_ascii=False),flush=True)
        (OUT/'matrix.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        state.update(status='COMPLETE',finished_unix=time.time());save()
    except Exception as exc:
        state.update(status='FAILED',error=str(exc),traceback=traceback.format_exc());save();raise


if __name__=='__main__':main()
