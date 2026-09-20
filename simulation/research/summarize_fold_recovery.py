"""Audit the feasibility battery and summarize measured fold depths."""
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fold_recovery_probe import OUT,ROOT,XML,Probe
from evaluate_policy import sha


def main():
    records=[];folds=[]
    for p in sorted((OUT/'trials').glob('*.json')):
        r=json.loads(p.read_text());data=np.load(p.with_suffix('.npz'));a=data['trace'];q=data['qpos']
        assert a.shape[0]==q.shape[0]==round(r['seconds']*50)
        assert np.isfinite(a).all() and np.isfinite(q).all()
        assert np.allclose(np.diff(a[:,0]),.02,atol=1e-8)
        assert r['command_delay_ms'] in (5,10,15)
        assert r['peak_torque_nm']<=.600001
        records.append(r)
        if r['label'].startswith('fold_') and r['label'].endswith('_mesh'):
            hold=a[(a[:,0]>=3.5)&(a[:,0]<4.8)]
            folds.append(dict(depth_command_mm=int(r['label'].split('_')[1][:-2]),
                hold_height_mm=float(hold[:,1].mean()*1000),hold_tilt_max_deg=float(hold[:,2].max()),
                body_contact_peak_n=float(hold[:,7].max()),return_stable=r['raw_recovery_criterion'],
                sampled_peak_torque_nm=r['peak_torque_nm'],new_convex_overlap_flags=len(r['collision_suspects'])))
    assert len(records)==78,len(records)
    folds.sort(key=lambda r:r['depth_command_mm'])
    zero=folds[0]['hold_height_mm']
    for r in folds:r['actual_drop_mm']=zero-r['hold_height_mm']
    geometry=json.loads((OUT/'fold_geometry.json').read_text());e=Probe()
    assert abs(e.model.body_mass.sum()-1.0982148)<1e-8
    assert np.all(e.model.actuator_gainprm[e.aids,0]>0)
    assert np.all(e.model.actuator_biasprm[e.aids,2]<0)
    assert sha(XML)=='e33fa22db6eb35c9f6d9d076267ae46e4bd7916cc83a63cf2b82ee63ed2e721c'
    release=Path('D:/microduck_rl/microdinosaur_p2.onnx')
    assert sha(release)=='0804114efd1e457ec2c297d709c0464fd0c2bfe251cbb36db7de74543857b0e3'
    recovery=json.loads((OUT/'recovery_trials.json').read_text())+json.loads((OUT/'bridge_search.json').read_text())
    confirm=json.loads((OUT/'fold_confirmation.json').read_text())
    result=dict(status='PASS',probe_count=len(records),fold_geometry_samples=len(geometry),
        fold_summary=folds,down_pose_recovery_trials=len(recovery),
        down_pose_raw_successes=sum(r['raw_recovery_criterion'] for r in recovery),
        positive_delay_fold_confirmations=len(confirm),positive_delay_fold_successes=sum(r['raw_recovery_criterion'] for r in confirm),
        physics_dt_s=.00125,controller_dt_s=.02,records_dt_s=.02,actuator_delay_ms=[5,10,15],
        nominal_kp=7,nominal_kd=.8,actual_kp=e.model.actuator_gainprm[e.aids,0].tolist(),
        actual_kd=(-e.model.actuator_biasprm[e.aids,2]).tolist(),
        nominal_mass_kg=float(e.model.body_mass.sum()),
        source_model_limits_verified_on_hardware=False,full_self_collision_verified=False,
        separate_head_imu_loop_used=False,scripted_feasibility_not_RL=True,release_unchanged=True)
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    files=[ROOT/n for n in ('fold_recovery_probe.py','search_recovery_bridge.py','confirm_fold_frontier.py',
        'render_fold_recovery.py','summarize_fold_recovery.py','terrain_skill_cfg.py')]
    files += [Path('D:/项目/miro_dinosaur/microdinosaur/blend2mjcf.py'),XML]
    result['sources']={}
    for p in files:shutil.copy2(p,snapshot/p.name);result['sources'][str(p)]=sha(p)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    plt.rcParams['font.family']='Microsoft YaHei';plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    xs=[r['depth_command_mm'] for r in folds]
    axes[0].plot(xs,[r['actual_drop_mm'] for r in folds],'-o',label='实际下降（失稳点不代表有效蹲深）')
    axes[0].plot(xs,xs,'--',color='gray',label='名义深度')
    axes[0].scatter([70],[folds[-1]['actual_drop_mm']],marker='x',s=120,color='red',label='失稳，回升失败',zorder=4)
    axes[0].set_xlabel('名义蹲深 (mm)');axes[0].set_ylabel('相对稳定站姿的实际下降 (mm)');axes[0].legend(fontsize=8)
    axes[1].plot(xs,[max(r['foot_position_error_mm']) for r in geometry],'-o',color='#cf752d')
    axes[1].axvspan(50,70,color='red',alpha=.08,label='膝关节接近当前限位')
    axes[1].set_xlabel('名义蹲深 (mm)');axes[1].set_ylabel('固定双脚IK位置残差 (mm)');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('v07折腿边界探针：3–4cm有余量，极深低姿需要另一种接触策略')
    fig.savefig(OUT/'fold_frontier.png',dpi=150);fig.savefig(OUT/'fold_frontier.svg');plt.close(fig)
    print(json.dumps({k:result[k] for k in ('status','probe_count','down_pose_recovery_trials','down_pose_raw_successes','positive_delay_fold_successes','release_unchanged')}))


if __name__=='__main__':main()
