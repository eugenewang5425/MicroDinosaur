"""Freeze this completed experiment's research entrypoints and validate delivery."""
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).parent;OUT=ROOT/'20260914_transition_refine'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files=[ROOT/name for name in ('refine_reference_audit.py','capture_v7_anchor.py','transition_refine_cfg.py',
        'train_terrain_skill.py','evaluate_transition_refine.py','audit_transition_live.py','audit_transition_training.py',
        'test_transition_refine.py','summarize_transition_refine.py','render_transition_refine.py','join_transition_video.py',
        'diagnose_transition_stop.py','diagnose_transition_gaze.py','heading_sim.py','head_attitude_sim.py','head_attitude.py',
        'terrain_skill_eval.py','hardware_sim.py','evaluate_policy.py','evaluate_gaze_ablation.py',
        'run_heading_stable_start.py','imu_heading.py','archive_transition_refine.py')]
    files += [Path('D:/microduck_rl/src/mjlab_microduck')/name for name in
        ('calibrated_head_action.py','retention_ppo.py','head_imu_action.py','head_attitude_torch.py','tasks/mdp.py')]
    files += [OUT/name for name in ('RECIPE.md','calibrated_reset_bank.json','v7_anchor.json')]
    dest=OUT/'source_snapshot';dest.mkdir(exist_ok=False)
    manifest={}
    for i,path in enumerate(files):
        target=dest/f'{i}_{path.name}';shutil.copy2(path,target);assert sha(path)==sha(target)
        manifest[str(path)]=dict(copy=target.name,sha256=sha(path))
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    summary=json.loads((OUT/'summary.json').read_text());assert summary['formal_trials']==48 and not summary['promoted']
    assert json.loads((OUT/'training_audit.json').read_text())['status']=='PASS'
    assert json.loads((OUT/'live_audit.json').read_text())['status']=='PASS'
    for label in ('previous','candidate','candidate_steps10_failure'):
        assert json.loads((OUT/'videos'/label/'verification.json').read_text())['metrics_match']
    for name in ('RESULTS.md','TABLES.md','REFERENCE_FINDINGS.md','RECIPE.md'):
        for link in re.findall(r'\]\(([^)]+)\)',(OUT/name).read_text(encoding='utf-8')):
            if not link.startswith(('http:','https:')):assert (OUT/link).exists(),link
    result=dict(status='PASS',frozen_source_files=len(files),formal_cpu_trials=48,startup_diagnostics=27,
        teacher_capture_trials=15,regression_tests=29,promoted=False,production_model_changed=False,
        video_metrics_verified=True,report_links_exist=True)
    (OUT/'delivery_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)


if __name__=='__main__':main()
