"""Post-hoc split of braking and sustained motion; does not change frozen gates."""
import json
from pathlib import Path
import numpy as np
from evaluate_transition_refine import OUT,phase_stats


def main():
    records=[];groups={}
    for label in ('previous','candidate'):
        for lag in (5,10,15):
            rows=[]
            for seed in (1,2,3):
                p=OUT/f'evaluation/{label}/transitions/resume_flat_stand_d40_lag{lag}_s{seed}.npz'
                z=np.load(p);f=z['gaze'];user=z['user_commands']
                zero=user[(user[:,0]>=17)&(abs(user[:,1])<1e-6)][0,0]
                phases={name:phase_stats(f,begin,end) for name,begin,end in
                    [('braking',17,18),('after1s',18,20),('after2s',19,20)]}
                record=dict(label=label,lag=lag,seed=seed,user_command_zero_at_s=float(zero),phases=phases)
                records.append(record);rows.append(record)
            groups[f'{label}_lag{lag}']={phase:{k:float(np.mean([r['phases'][phase][k] for r in rows]))
                for k in ('net_drift_mm_s','planar_speed_mean_mm_s','camera_pitch_mean_deg','camera_pitch_std_deg')}
                for phase in ('braking','after1s','after2s')}
    (OUT/'stop_diagnostics.json').write_text(json.dumps(dict(records=records,groups=groups),indent=2),encoding='utf-8')
    print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
