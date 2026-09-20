"""Summarize existing trajectories and sampled CAD; do not rerun training."""
from pathlib import Path
import json
import numpy as np
import imageio.v2 as imageio

OUT=Path(__file__).parent/'20260914_contact_motion'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2),encoding='utf-8')

gates={}
for folder in ('preferred_squat_cad','run_final_cad','run_sole_final_cad','getup_dense_final_cad'):
    frames=read(OUT/folder/'solid_intersections.json')
    home={tuple(sorted(r['parts'])):r['intersection_mm3'] for r in frames[0]['pairs']}
    rows=[dict(frame=f['frame'],parts=r['parts'],intersection_mm3=r['intersection_mm3'],
        increase_over_home_mm3=max(0,r['intersection_mm3']-home[tuple(sorted(r['parts']))]))
        for f in frames[1:] for r in f['pairs']]
    rows.sort(key=lambda r:r['increase_over_home_mm3'],reverse=True)
    failed=[r for r in rows if r['increase_over_home_mm3']>.001]
    gates[folder]=dict(sampled_motion_poses=len(frames)-1,home_poses=1,tolerance_mm3=.001,
        passed=not failed,failed_pairs_at_poses=len(failed),
        failed_pose_count=len({r['frame'] for r in failed}),max_increase_mm3=rows[0]['increase_over_home_mm3'],
        largest_intersections=rows[:10],failed_entries=failed,
        limitation='Sampled solid intersections against HOME; not continuous full-range self-collision validation')
write('final_geometry_gates.json',gates)
print('CAD',json.dumps({k:{x:v[x] for x in ('sampled_motion_poses','passed','failed_pose_count','max_increase_mm3')} for k,v in gates.items()}))

recovery=[]
for folder in ('getup_dense_final','getup_preferred_contact'):
    for r in read(OUT/folder/'summary.json'):
        if r['status']!='COMPLETE':continue
        x=np.load(OUT/folder/(r['key']+'.npz'))['physics'];tail=x[x[:,0]>=10]
        checks={'height':(tail[:,1]>.1035)&(tail[:,1]<.1255),'tilt':tail[:,8]<10,
            'speed':np.linalg.norm(tail[:,9:11],axis=1)<.03,'angular_speed':tail[:,17]<.3,
            'both_feet':(tail[:,5]>.5)&(tail[:,6]>.5),'body_clear':tail[:,7]<.2}
        recovery.append(dict(folder=folder,key=r['key'],start_class=r['start_class'],
            passed=r['passed'],final_tilt_deg=r['final_tilt_deg'],
            final_2s=dict(height_range_mm=(tail[:,1]*1000)[[np.argmin(tail[:,1]),np.argmax(tail[:,1])]].tolist(),
                speed_mean_m_s=float(np.linalg.norm(tail[:,9:11],axis=1).mean()),
                omega_mean_rad_s=float(tail[:,17].mean()),foot_force_min_N=tail[:,5:7].min(0).tolist(),
                nonfoot_force_max_N=float(tail[:,7].max())),
            individual_gate_fraction={k:float(v.mean()) for k,v in checks.items()}))
write('getup_failure_analysis.json',recovery)
print('FRONT',json.dumps([r for r in recovery if r['key'].startswith('front')]))

media=[]
for relative,seconds in [('squat_contact_optimized.mp4',5),('getup_dense_final/front_c10_s941.mp4',11),('run_sole_final/v0.75_c10_s941.mp4',5)]:
    path=OUT/relative;reader=imageio.get_reader(path);meta=reader.get_meta_data()
    frame=reader.get_data(int(seconds*meta['fps']));dest=path.with_name(path.stem+'_review.png')
    imageio.imwrite(dest,frame);reader.close()
    media.append(dict(path=str(path),thumbnail=str(dest),fps=meta['fps'],duration_s=meta.get('duration'),
        reviewed_frame_s=seconds,visual_review=False))
write('video_audit.json',media)
