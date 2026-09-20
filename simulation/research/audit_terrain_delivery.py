"""Archive confirmation counts, provenance and the repaired resume-LR check."""
import hashlib
import json
from pathlib import Path
import shutil
import torch
from terrain_skill_eval import OUT


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    smoke=Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_terrain_skills/20260914_resume_lr_smoke_64x5')
    p=json.loads((smoke/'run_provenance.json').read_text()); assert p['status']=='COMPLETE'
    d=torch.load(p['final_checkpoint'],map_location='cpu',weights_only=False)
    assert all(g['lr']==p['optimizer_learning_rate'] for g in d['optimizer_state_dict']['param_groups'])
    counts={}; total=0; falls=0; distances=[]
    for label in ('v7_reference','terrain_15650','crouch_15650'):
        n=0; f=0
        for path in sorted((OUT/'evaluation'/label).glob('*/matrix.json')):
            for r in json.loads(path.read_text()):
                assert r['status']=='COMPLETE'
                m=r['metrics']; n+=1; f+=int(m['fell'])
                # Descriptive route milestones added after the main tests;
                # report alongside falls, not as a preregistered success gate.
                kind=m['terrain']; required=(.81 if kind.startswith('steps') else
                    (.8 if kind.startswith('slope') else (2.3 if kind=='roughgrid_8mm' else None)))
                if required is not None:
                    trial=path.parent/f"{kind}__{m['posture']}__{m['scenario']}__s{m['seed']}.npz"
                    import numpy as np
                    end_x=float(np.load(trial)['gaze'][-1,1])
                    distances.append(dict(policy=label,terrain=kind,seed=m['seed'],
                        fell=m['fell'],end_x_m=end_x,descriptive_milestone_x_m=required,
                        reached_without_fall=not m['fell'] and end_x>=required))
        for r in json.loads((OUT/'turns'/label/'matrix.json').read_text()):
            n+=1;f+=int(r['metrics']['fell'])
        counts[label]=dict(complete=n,falls=f);total+=n;falls+=f
    assert total==150, counts
    release=Path('D:/microduck_rl/microdinosaur_p2.onnx')
    assert digest(release)=='0804114efd1e457ec2c297d709c0464fd0c2bfe251cbb36db7de74543857b0e3'
    scripts=['terrain_skill_cfg.py','train_terrain_skill.py','terrain_skill_eval.py',
        'test_terrain_skills.py','probe_crouch_feasibility.py','probe_planned_motion_contacts.py',
        'probe_jump_feasibility.py','export_skill_checkpoint.py','evaluate_skill_turns.py',
        'render_terrain_skill.py','summarize_terrain_skills.py','plot_terrain_skills.py',
        'audit_terrain_skill_runs.py','audit_terrain_delivery.py','head_attitude.py',
        'head_attitude_sim.py','head_attitude_runtime.py','heading_sim.py','imu_heading.py',
        'hardware_sim.py','evaluate_policy.py','evaluate_gaze_ablation.py','run_heading_stable_start.py']
    snap=OUT/'source_snapshot';snap.mkdir(exist_ok=True)
    sources={}
    for name in scripts:
        path=Path(__file__).parent/name;shutil.copy2(path,snap/name);sources[name]=digest(path)
    result=dict(status='PASS',confirmation=counts,total_confirmation=total,falls=falls,
        additional_exploratory_trials=7,archived_superseded_stripe_trials=3,
        repaired_learning_rate_entrypoint=dict(status='PASS',smoke_envs=64,smoke_updates=5,
            actual_learning_rate=p['optimizer_learning_rate'],provenance=str(smoke/'run_provenance.json')),
        source_sha256=sources,release_sha256=digest(release),
        training_candidate_promoted=False,hardware_deployed=False,erpo=False)
    (OUT/'delivery_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (OUT/'route_milestones.json').write_text(json.dumps(distances,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',complete=total,falls=falls,release_unchanged=True)))


if __name__=='__main__':main()
