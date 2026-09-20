"""Post-hoc diagnosis: distinguish delayed settling from a persistent gait cycle."""
import json
from pathlib import Path
import numpy as np
from terrain_skill_eval import TerrainSkillExperiment
from evaluate_deep_crouch import OUT
from evaluate_policy import sha


def main():
    policy = Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_deep_crouch/20260914_train_512x301/candidate.onnx')
    dest = OUT/'return_history'; dest.mkdir(exist_ok=False)
    records = []
    for posture in ('none', 'crouch30', 'crouch40'):
        e = TerrainSkillExperiment(policy, posture=posture)
        metrics, trace = e.run('stand', 'imu', 1, 30., True)
        full = np.asarray(e.sim.gaze_trace); t = full[:, 0]-6.
        windows = []
        for lo, hi in ((9.5, 12.), (15., 20.), (25., 30.)):
            f = full[(t>=lo)&(t<hi)]
            v = np.diff(f[:,1:3], axis=0)/np.diff(f[:,0])[:,None]
            controls = np.asarray(e.head_rows); controls = controls[(controls[:,0]>=lo)&(controls[:,0]<hi)]
            windows.append(dict(time_s=[lo,hi], planar_speed_mean_mm_s=float(np.linalg.norm(v,axis=1).mean()*1000),
                net_drift_mm_s=float(np.linalg.norm(f[-1,1:3]-f[0,1:3])/(f[-1,0]-f[0,0])*1000),
                camera_pitch_mean_deg=float(np.rad2deg(f[:,6].mean())),
                camera_pitch_std_deg=float(np.rad2deg(f[:,6].std())),
                camera_yaw_rate_rms_deg_s=float(np.rad2deg(np.sqrt(np.mean(f[:,7]**2)))),
                head_correction_mean_rad=controls[:,4:7].mean(0).tolist(),
                nominal_head_target_mean_rad=controls[:,13:16].mean(0).tolist()))
        record = dict(posture=posture, seed=1, seconds=30, fell=metrics['fell'], windows=windows,
            policy_sha256=sha(policy), purpose='Post-hoc diagnostic, not a new promotion threshold')
        records.append(record); np.savez_compressed(dest/(posture+'.npz'), gaze=full, trace=trace, head=np.asarray(e.head_rows))
        (dest/(posture+'.json')).write_text(json.dumps(record,indent=2), encoding='utf-8')
        print(json.dumps(record), flush=True)
    (dest/'matrix.json').write_text(json.dumps(records,indent=2), encoding='utf-8')


if __name__ == '__main__': main()
