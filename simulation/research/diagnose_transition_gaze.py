"""Separate camera direction bias from yaw-rate jitter and teacher fit."""
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from evaluate_transition_refine import OUT


def main():
    records=[]
    for label in ('previous','candidate'):
        for seed in (1,2,3):
            p=OUT/f'evaluation/candidate/confirmation/legacy_flat_straight_d0_lag10_s{seed}.npz' if label=='candidate' else OUT.parent/f'20260914_deep_crouch/evaluation/deep/d0_straight_lag10_s{seed}.npz'
            z=np.load(p);f=z['gaze'];zero=f[np.argmin(abs(f[:,0]-6)),4];f=f[(f[:,0]>=8)&(f[:,0]<18)]
            yaw=np.unwrap(f[:,5])-zero;yaw-=2*np.pi*round(yaw[0]/(2*np.pi))
            h=z['head_controls'];h=h[(h[:,0]>=2)&(h[:,0]<12)]
            records.append(dict(label=label,seed=seed,camera_yaw_mean_deg=float(np.rad2deg(yaw.mean())),
                camera_yaw_std_deg=float(np.rad2deg(yaw.std())),camera_pitch_mean_deg=float(np.rad2deg(f[:,6].mean())),
                estimated_yaw_error_mean_deg=float(np.rad2deg(h[:,3].mean())),
                nominal_policy_head_yaw_mean_deg=float(np.rad2deg(h[:,14].mean())),
                controller_head_yaw_correction_mean_deg=float(np.rad2deg(h[:,5].mean()))))
    bank=np.load(OUT/'v7_anchor.npz');obs=bank['obs'][::10];wanted=bank['actions'][::10]
    paths={'previous':Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_deep_crouch/20260914_train_512x301/candidate.onnx'),
        'candidate':Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_transition_refine/20260914_train_512x201/candidate.onnx')}
    teacher_fit={}
    for label,path in paths.items():
        session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
        y=np.concatenate([session.run(None,{'obs':r[None]})[0] for r in obs])
        err=(y-wanted)**2
        teacher_fit[label]=dict(all_observations_action_rmse_rad=float(np.sqrt(err.mean())),
            steps_observations_action_rmse_rad=float(np.sqrt(err[720:].mean())))
    result=dict(records=records,teacher_fit=teacher_fit,
        note='Post-hoc diagnostic on recorded teacher observations; lower action error is not terrain success or a causal ablation.')
    (OUT/'gaze_teacher_diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
