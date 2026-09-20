"""Aggregate all predeclared confirmations, audit checkpoints, draw traces."""
import json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_policy import sha
from evaluate_deep_crouch import OUT

RUN = Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_deep_crouch/20260914_train_512x301')


def main():
    groups = {}
    for label in ('v7', 'shallow', 'deep'):
        rows = json.loads((OUT/'evaluation'/label/'matrix.json').read_text())
        assert all(r['status']=='COMPLETE' for r in rows), 'Report failed evaluations explicitly before aggregating'
        by_case = {}
        for row in rows:
            key = f"d{row['depth_mm']}_{row['scenario']}_lag{row['motor_delay_ms']}"
            by_case.setdefault(key, []).append(row['metrics'])
        groups[label] = {}
        for key, metrics in by_case.items():
            entry = dict(n=len(metrics), falls=sum(m['fell'] for m in metrics), successes=sum(m['cycle_success'] for m in metrics))
            for metric in ('actual_drop_mm', 'height_plateau_mean_m', 'return_height_error_mm',
                           'returned_drift_m_s', 'body_vx_mean_m_s', 'camera_heading_error_rms_deg',
                           'camera_yaw_rate_error_rms_deg_s', 'camera_pitch_rms_deg', 'camera_roll_rms_deg',
                           'heading_error_final_deg'):
                values = [m[metric] for m in metrics]
                entry[metric] = dict(mean=float(np.mean(values)), min=float(min(values)), max=float(max(values)))
            groups[label][key] = entry
    mean = lambda label, key, metric: groups[label][key][metric]['mean']
    covered = [v for k, v in groups['deep'].items() if k.startswith(('d30_stand', 'd40_stand'))]
    gates = dict(deep_delay_cycles=sum(v['successes'] for v in covered)==18,
        stand_drift_halved=mean('deep', 'd0_stand_lag10', 'returned_drift_m_s') <= .5*mean('shallow', 'd0_stand_lag10', 'returned_drift_m_s'),
        head_yaw_rate_preserved=mean('deep', 'd0_stand_lag10', 'camera_yaw_rate_error_rms_deg_s') <= mean('shallow', 'd0_stand_lag10', 'camera_yaw_rate_error_rms_deg_s'),
        head_pitch_preserved=mean('deep', 'd0_stand_lag10', 'camera_pitch_rms_deg') <= mean('shallow', 'd0_stand_lag10', 'camera_pitch_rms_deg'),
        walking_speed_preserved=mean('deep', 'd0_straight_lag10', 'body_vx_mean_m_s') >= .95*mean('shallow', 'd0_straight_lag10', 'body_vx_mean_m_s'))
    (OUT/'summary.json').write_text(json.dumps(dict(groups=groups, primary_gates=gates), indent=2), encoding='utf-8')
    lines = ['# 深蹲确认汇总', '', '均值；每格三个评估初态，不是三个训练seed。所有跌倒单列。', '',
        '|模型/条件|蹲幅mm|低姿高度mm|回升误差mm|末段平移mm/s|相机朝向RMS°|偏航角速度RMS°/s|前向cm/s|蹲起合格/跌倒|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label, cases in groups.items():
        for key, e in cases.items():
            value = lambda m, s=1.: e[m]['mean']*s
            lines.append(f"|{label}/{key}|{value('actual_drop_mm'):.2f}|{value('height_plateau_mean_m',1000):.2f}|{value('return_height_error_mm'):.2f}|{value('returned_drift_m_s',1000):.2f}|{value('camera_heading_error_rms_deg'):.2f}|{value('camera_yaw_rate_error_rms_deg_s'):.2f}|{value('body_vx_mean_m_s',100):.2f}|{e['successes']}/{e['n']}; {e['falls']}倒|")
    (OUT/'TABLES.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    p = json.loads((RUN/'run_provenance.json').read_text())
    assert p['status'] == 'COMPLETE'
    source = torch.load(p['source_checkpoint'], map_location='cpu', weights_only=False)
    final = torch.load(p['final_checkpoint'], map_location='cpu', weights_only=False)
    checkpoints = []
    for path in sorted(RUN.glob('model_*.pt')):
        d = torch.load(path, map_location='cpu', weights_only=False)
        rates = [g['lr'] for g in d['optimizer_state_dict']['param_groups']]
        assert all(lr == .00003 for lr in rates)
        for group in ('actor_state_dict', 'critic_state_dict'):
            assert all(torch.isfinite(v).all() for v in d[group].values())
        for state in d['optimizer_state_dict']['state'].values():
            assert all(not torch.is_tensor(v) or torch.isfinite(v).all() for v in state.values())
        checkpoints.append(dict(name=path.name, iteration=d['iter'], infos=d['infos'], learning_rates=rates))
    count = final['infos']['env_state']['common_step_counter']-source['infos']['env_state']['common_step_counter']
    assert count == 301*24
    assert any('normalizer' in k for k in final['actor_state_dict'])
    session = ort.InferenceSession(p['onnx'], providers=['CPUExecutionProvider'])
    y = session.run(None, {'obs': np.zeros((1, 81), np.float32)})[0]
    assert y.shape == (1, 19) and np.isfinite(y).all()
    for key, field in (('final_checkpoint', 'final_sha256'), ('onnx', 'onnx_sha256')):
        assert sha(Path(p[key])) == p[field]
    publication = Path('D:/microduck_rl/microdinosaur_p2.onnx')
    assert sha(publication) == '0804114efd1e457ec2c297d709c0464fd0c2bfe251cbb36db7de74543857b0e3'
    audit = dict(status='PASS', updates=301, environments=512, new_transitions=count*512,
        checkpoints=checkpoints, exported_onnx_sha256=p['onnx_sha256'], head_loop_in_gpu_training=True,
        training_seconds=p['finished_unix']-p['started_unix'], published_model_unchanged=True,
        original_source_unchanged=sha(Path(p['source_checkpoint']))==p['source_sha256'],
        source_snapshot_matches_running_files=all(sha(Path(path))==digest for path,digest in p['source_hashes'].items()))
    (OUT/'training_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for label, color in (('shallow', '#a77236'), ('deep', '#247b9a')):
        z = np.load(OUT/'evaluation'/label/'d40_stand_lag10_s1.npz')
        f = z['gaze']; f = f[f[:,0]>=6.]; t = f[:,0]-6.
        axes[0].plot(t, f[:,3]*1000, color=color, label=label)
        axes[1].plot(t, np.rad2deg(np.unwrap(f[:,5])-f[0,4]), color=color, label=label)
        axes[2].plot(t, np.rad2deg(f[:,7]), color=color, label=label)
        if label=='deep': axes[0].plot(z['commands'][:,0], 117.182+z['commands'][:,1]*1000, '--', color='#666666', label='nominal height command')
    for ax, title in zip(axes, ('Trunk height (mm)', 'Camera forward yaw error (deg)', 'Camera yaw rate (deg/s)')):
        ax.set_ylabel(title); ax.grid(alpha=.2); ax.axvspan(2, 7, alpha=.06, color='#247b9a')
    axes[0].legend(loc='best'); axes[-1].set_xlabel('Time after stationary calibration (s)')
    fig.suptitle('40 mm crouch and return | preselected seed 1 | positive hardware delays')
    fig.tight_layout(); fig.savefig(OUT/'confirmation_traces.png', dpi=160); plt.close(fig)
    print(json.dumps(dict(gates=gates, audit=audit['status'], groups=groups), indent=2))


if __name__ == '__main__': main()
