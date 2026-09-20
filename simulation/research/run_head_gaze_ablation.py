"""Sequential paired training; source/config gates precede and follow full runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/20260914_head_gaze_ablation'
RUNS=Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_v07_calibration')
SOURCE=Path('D:/microduck_rl/logs/rsl_rl/velocity_microdinosaur/2026-09-13_13-46-57_velocity_microdinosaur/model_15500.pt')
XML=Path('D:/microduck_rl/src/mjlab_microduck/robot/microdinosaur_v07/robot_microdinosaur_v07.xml')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return [d for key in sorted(set(a)|set(b)) for d in differences(a.get(key),b.get(key),path+'.'+key)]
    return [] if a==b else [{'path':path,'control':a,'treatment':b}]


def yaml_tags(node,path=''):
    result={path:node.tag}
    if isinstance(node,yaml.MappingNode):
        for k,v in node.value:result.update(yaml_tags(v,path+'.'+k.value))
    elif isinstance(node,yaml.SequenceNode):
        for i,v in enumerate(node.value):result.update(yaml_tags(v,path+f'[{i}]'))
    return result


def compare_pair(a,b):
    diffs=[]
    for name in ('env.yaml','agent.yaml'):
        ta=(a/'params'/name).read_text();tb=(b/'params'/name).read_text()
        diffs+=differences(yaml.load(ta,Loader=yaml.BaseLoader),yaml.load(tb,Loader=yaml.BaseLoader),name)
        assert yaml_tags(yaml.compose(ta))==yaml_tags(yaml.compose(tb)), 'YAML callable/type tags changed'
    assert {d['path'] for d in diffs}=={'agent.yaml.run_name','env.yaml.rewards.head_gaze.weight'},diffs
    pa=json.loads((a/'run_provenance.json').read_text());pb=json.loads((b/'run_provenance.json').read_text())
    for key in ('source_checkpoint_sha256','model_xml_sha256','seed','iterations','envs','delay_contract_after'):
        assert pa[key]==pb[key],key
    assert [r['sha256'] for r in pa['source_files']]==[r['sha256'] for r in pb['source_files']]
    assert pa['head_world_gaze_weight']==pb['head_world_gaze_weight']==1.5
    return {'status':'PASS','differences':diffs,'same_source_physics_seed_budget_and_code':True,
        'callable_and_type_tags_identical':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['smoke','train'],required=True)
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    watched=[ROOT/'research/train_candidate.py',XML,
        *[Path('D:/microduck_rl/src/mjlab_microduck')/name for name in (
            'tasks/mdp.py','tasks/__init__.py','tasks/microduck_velocity_env_cfg.py',
            'robot/microdinosaur_constants.py','action_filter.py','s288_profile.py','s288_protocol.py','tasks/symmetry_microdinosaur.py')]]
    hashes={str(p):sha(p) for p in watched}
    plan_path=OUT/'plan.json'
    if not plan_path.exists():
        plan={'source_checkpoint':str(SOURCE),'source_sha256':sha(SOURCE),'training_seeds':[42,43,44],
            'envs':1024,'updates_per_run':101,'paired_runs':6,'control_head_gaze':-.2,'treatment_head_gaze':0.,
            'head_world_target_unchanged':True,'body_ang_vel_unchanged':True,'algorithm':'PPO',
            'hardware_profile':'s288-protocol with effective friction randomization',
            'training_delay_ms':{'motor':[5,15],'position':[20,40],'velocity':[20,20]},
            'training_physics_dt':.005,'seed_replication_is_shared_checkpoint_not_from_scratch':True,
            'source_hashes':hashes,'promote_automatically':False}
        plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    assert json.loads(plan_path.read_text())['source_hashes']==hashes,'Source changed since experiment plan'
    if args.stage=='train':
        check=json.loads((OUT/'smoke_pair_check.json').read_text());assert check['status']=='PASS'
    seeds=[42] if args.stage=='smoke' else [42,43,44]
    envs,iterations=(64,5) if args.stage=='smoke' else (1024,101)
    state={'stage':args.stage,'status':'RUNNING','started_unix':time.time(),'jobs':[]}
    state_path=OUT/f'{args.stage}_status.json'
    def save():state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    save()
    try:
        for seed in seeds:
            pair=[]
            # Alternate order across seeds to avoid always assigning one arm
            # to the warmer/later GPU run. No policy/optimizer sharing between arms.
            arms=[('control',-.2),('no_neck_cost',0.)]
            if seed%2:arms.reverse()
            for label,weight in arms:
                assert {p:sha(p) for p in hashes}==hashes
                run=RUNS/f'20260914_gaze_{args.stage}_s{seed}_{label}_{envs}x{iterations}'
                pair.append((label,run))
                if run.exists():
                    provenance=json.loads((run/'run_provenance.json').read_text())
                    assert provenance['status']=='COMPLETE','Incomplete run requires inspection; do not overwrite'
                    state['jobs'].append({'seed':seed,'arm':label,'run':str(run),'status':'COMPLETE_REUSED'})
                    save();continue
                command=[sys.executable,'-u',str(ROOT/'research/train_candidate.py'),
                    '--xml',str(XML),'--checkpoint',str(SOURCE),'--out',str(run),
                    '--envs',str(envs),'--iterations',str(iterations),'--seed',str(seed),
                    '--hardware-profile','s288-protocol','--reward-recipe','v7','--head-gaze-weight',str(weight)]
                stdout=OUT/f'{args.stage}_s{seed}_{label}.log';stderr=stdout.with_suffix('.err.log')
                job={'seed':seed,'arm':label,'run':str(run),'status':'RUNNING','started_unix':time.time()}
                state['jobs'].append(job);save()
                child_env=os.environ.copy();child_env.pop('MICRODINO_PROBE',None)
                with stdout.open('w',encoding='utf-8') as log,stderr.open('w',encoding='utf-8') as err:
                    child=subprocess.Popen(command,cwd=ROOT,env=child_env,stdout=log,stderr=err,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    job['pid']=child.pid;save();code=child.wait()
                if code:raise RuntimeError(f'{run.name} exited {code}; inspect {stderr}')
                provenance=json.loads((run/'run_provenance.json').read_text())
                assert provenance['status']=='COMPLETE' and provenance['finite_output_check']
                job.update(status='COMPLETE',finished_unix=time.time());save()
                print(json.dumps(job),flush=True)
            mapped=dict(pair);check=compare_pair(mapped['control'],mapped['no_neck_cost'])
            (OUT/('smoke_pair_check.json' if args.stage=='smoke' else f'pair_s{seed}_check.json')).write_text(json.dumps(check,indent=2))
        state.update(status='COMPLETE',finished_unix=time.time());save()
    except Exception as exc:
        state.update(status='FAILED',error=str(exc));save();raise


if __name__=='__main__':main()
