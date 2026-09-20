"""Reproducible static-ray feasibility study; no RL, dynamics or robot writes."""
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from tof_skill_contract import (TofFrame, SkillRequestGate, evidence, rays,
                                pitch_rotation, height_envelope)

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'20260914_tof_feasibility'
PLANTS = ROOT/'20260914_terrain_skills/plants'
POLICY = Path('D:/microduck_rl/microdinosaur_p2.onnx')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_model(kind):
    path = PLANTS/kind/'nominal.mjb'
    m = mujoco.MjModel.from_binary_path('nominal.mjb', assets={'nominal.mjb': path.read_bytes()})
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d, path


def query_model(kind):
    """Copy actual archived group-0 geometry into a terrain-only query model.

    20/30 mm steps scale only copied stair heights; these are synthetic probes,
    not claims that those terrains have been trained. Robot self-occlusion is
    excluded: no physical sensor mount has yet been chosen.
    """
    base = ('downsteps_10' if kind.startswith('downsteps') else
            'steps_10' if kind.startswith('steps') else 'flat')
    m, d, path = source_model(base)
    s = mujoco.MjSpec()
    factor = int(kind.split('_')[-1])/10 if 'steps_' in kind else 1.
    geometry = []
    for i in range(m.ngeom):
        if m.geom_group[i] != 0:
            continue
        pos, size = d.geom_xpos[i].copy(), m.geom_size[i].copy()
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, d.geom_xmat[i])
        if factor != 1.:
            pos[2] *= factor
            if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
                size[2] *= factor
        s.worldbody.add_geom(name=m.geom(i).name, type=mujoco.mjtGeom(int(m.geom_type[i])),
                             pos=pos, size=size, quat=quat, group=0)
        geometry.append(dict(name=m.geom(i).name, pos=pos.tolist(), size=size.tolist()))
    if kind == 'box_100':
        # Broad obstacle probe: 100 mm high, 300 mm deep, 400 mm wide.
        s.worldbody.add_geom(name='probe_box', type=mujoco.mjtGeom.mjGEOM_BOX,
                             pos=[.45,0.,.05], size=[.15,.2,.05], group=0)
    qm = s.compile()
    qd = mujoco.MjData(qm)
    mujoco.mj_forward(qm, qd)
    return qm, qd, dict(source=str(path), source_sha256=sha(path), height_scale=factor,
                       copied_geometry=geometry, probe_box=kind == 'box_100')


SUBRAYS = rays(3)


def scan(model, data, origin, rotation):
    directions = SUBRAYS @ rotation.T
    ranges = np.full((8,8,9), np.nan)
    geom = np.empty(1, np.int32)
    for i in range(8):
        for j in range(8):
            for k in range(9):
                r = mujoco.mj_ray(model, data, origin, directions[i,j,k], None, 1, -1, geom)
                if .02 <= r <= 4.:
                    ranges[i,j,k] = r
    points = origin + ranges[:, :, :, None]*directions
    return ranges, points


def make_frame(seq, t, ranges, origin, rotation, **kw):
    return TofFrame('synthetic_replay', seq, t, t+.02, t,
                    ranges.copy(), np.isfinite(ranges), origin.copy(), rotation.copy(),
                    calibrated=True, **kw)


def main():
    OUT.mkdir(exist_ok=True)
    before = sha(POLICY)
    m, d, p = source_model('steps_10')
    head_origin = d.site_xpos[m.site('robot/head_imu').id].copy()
    nominal_height = float(head_origin[2])
    started = time.perf_counter()
    models, provenance = {}, {}
    kinds = ['flat','steps_10','steps_20','steps_30','downsteps_10',
             'downsteps_20','downsteps_30','box_100']
    for kind in kinds:
        qm, qd, info = query_model(kind)
        models[kind] = (qm, qd)
        provenance[kind] = info
    rows = []
    snapshots = {}
    for height in [nominal_height-.04, nominal_height, nominal_height+.04]:
        for pitch in [0,20,35,45]:
            rotation = pitch_rotation(pitch)
            for kind in kinds:
                print(f'height={height:.4f} pitch={pitch} scene={kind}', flush=True)
                for distance in np.round(np.arange(.1,.801,.025), 3):
                    origin = np.array([.3-distance,0.,height])
                    rr, xyz = scan(*models[kind], origin, rotation)
                    center = rr[:, :, 4]
                    z = xyz[:, :, 4, 2]
                    pure = np.zeros((8,8), bool)
                    first = np.zeros((8,8), bool)
                    if 'steps_' in kind:
                        top = int(kind.split('_')[-1])/1000 * (-1 if kind.startswith('down') else 1)
                        first_sub = (np.abs(xyz[:,:,:,2]-top) < 1e-6) & (xyz[:,:,:,0] > .300001) & (xyz[:,:,:,0] < .479999)
                        first = first_sub[:,:,4]
                        pure = np.all(first_sub, axis=2)
                    else:
                        top = 0.
                    profiles = {}
                    for name, pose_err, zone_extent, range_err in [
                        ('center_range_only',0.,False,.05),
                        ('zone_pose_1deg_dark',1.,True,.05),
                        ('zone_pose_1deg_ambient',1.,True,.11)]:
                        radius = height_envelope(center, rotation, pose_err,
                                                 .003 if pose_err else 0., range_err, zone_extent)
                        different = z-radius > .005 if top >= 0 else z+radius < -.005
                        profiles[name] = int(np.count_nonzero(first & different))
                    frame = make_frame(0,0.,center,origin,rotation)
                    decision = evidence(frame,.03)
                    finite_z = np.where(np.isfinite(xyz[:,:,:,2]), xyz[:,:,:,2], np.nan)
                    # Mixed sampled surface height is geometric diagnostics only;
                    # no claim that a real driver knows this ground-truth mask.
                    spread = np.max(np.where(np.isfinite(finite_z),finite_z,-np.inf),axis=2)-np.min(np.where(np.isfinite(finite_z),finite_z,np.inf),axis=2)
                    row = dict(scene=kind,height_m=height,pitch_deg=pitch,distance_to_first_edge_m=float(distance),
                               valid_center_zones=int(np.isfinite(center).sum()),
                               first_tread_center_zones=int(first.sum()), first_tread_pure_sampled_zones=int(pure.sum()),
                               mixed_height_sampled_zones=int(np.count_nonzero(spread > .005)),
                               evidence=decision.state, supporting_zones=decision.supporting_zones, **profiles)
                    rows.append(row)
                    if height == nominal_height and distance == .3 and pitch in [0,35] and kind in ['flat','steps_10','steps_30','box_100']:
                        snapshots[f'{kind}_{pitch}'] = (center,xyz,origin,rotation)
    with (OUT/'geometry_sweep.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    # Moving approaches are sensor-only kinematic replays, not robot dynamics.
    replay = []
    for kind in ['flat','steps_10','steps_30','box_100']:
        for speed in [.2,.35,.55]:
            gate = SkillRequestGate()
            for seq,t in enumerate(np.arange(0., .7/speed, 1/15)):
                distance = .8-speed*t
                origin = np.array([.3-distance,0.,nominal_height])
                rotation = pitch_rotation(35)
                rr, _ = scan(*models[kind], origin, rotation)
                frame = make_frame(seq,float(t),rr[:,:,4],origin,rotation)
                # Repeat at a faster scheduler rate: repeated sensor frames must
                # never manufacture the three independent confirmations.
                for offset in [.03,.05,.06]:
                    decision = gate.update(frame,float(t+offset))
                    replay.append(dict(scene=kind,speed_m_s=speed,sequence=seq,t_s=float(t+offset),
                                       distance_at_capture_m=float(distance),
                                       distance_at_request_m=float(distance-speed*offset),**asdict(decision)))
    (OUT/'request_replay.json').write_text(json.dumps(replay,indent=2),encoding='utf-8')
    summary = []
    for kind in kinds:
        for pitch in [0,20,35,45]:
            part=[r for r in rows if r['scene']==kind and r['pitch_deg']==pitch and r['height_m']==nominal_height]
            summary.append(dict(scene=kind,pitch_deg=pitch,
                positions_with_4_first_tread_centers=sum(r['first_tread_center_zones']>=4 for r in part),
                positions_with_4_pure_sampled_zones=sum(r['first_tread_pure_sampled_zones']>=4 for r in part),
                positions_with_4_height_separable_first_zones=sum(r['zone_pose_1deg_dark']>=4 for r in part),
                total_positions=len(part)))
    latency=[]
    # Assumed upper budget: phase wait + 3-frame confirmation, 20 ms delivery,
    # 10 ms processing, 20 ms scheduler, existing 15 ms actuation delay.
    for speed in [.2,.35,.55]:
        for transition_s in [0.,.25,.5]:
            latency.append(dict(speed_m_s=speed,perception_to_motor_budget_s=3/15+.02+.01+.02+.015,
                                hypothetical_transition_s=transition_s,
                                travel_m=speed*(3/15+.02+.01+.02+.015+transition_s)))
    mount=[]
    for pitch in [0,20,35,45]:
        mount.append(dict(pitch_deg=pitch,
                          nearest_flat_ground_fov_edge_m=nominal_height/np.tan(np.deg2rad(pitch+22.5))))
    sensitivity=[]
    for range_m in [.3,.5,.8]:
        for pose_deg in [.5,1.,2.]:
            for height_error_m in [.003,.01,.02]:
                halfwidth=height_envelope(np.full((8,8),range_m),pitch_rotation(35),pose_deg,height_error_m)
                sensitivity.append(dict(range_m=range_m,pose_error_deg=pose_deg,
                    assumed_sensor_height_error_m=height_error_m,
                    center_zone_height_halfwidth_m=float(halfwidth[3,3])))
    first_requests=[]
    for kind in ['flat','steps_10','steps_30','box_100']:
        for speed in [.2,.35,.55]:
            events=[x for x in replay if x['scene']==kind and x['speed_m_s']==speed and x['state']=='REQUEST_ONLY']
            first_requests.append(dict(scene=kind,speed_m_s=speed,first=events[0] if events else None))
    result=dict(kind='static_geometry_and_sensor_only_kinematic_study', frames=len(rows),
        rays_per_frame=576, head_site_qpos0_m=head_origin.tolist(),
        nominal_height_m=nominal_height, camera_orientation_changed=False,
        physical_sensor_mount_selected=False, robot_self_occlusion_simulated=False,
        tof_radiometry_simulated=False, pi_or_f411_benchmark=False,
        dynamic_policy_handoff_validated=False, production_sha_before=before,
        production_sha_after=sha(POLICY), geometry_summary=summary, source_models=provenance,
        mounting_geometry=mount, latency_assumptions=latency, first_request_replays=first_requests,
        height_error_sensitivity=sensitivity,
        lightweight_payload=dict(range_uint16_bytes=128,status_uint8_bytes=64,
                                 timestamp_sequence_header_bytes=24,bytes_per_frame=216,
                                 bytes_per_second_at_15hz=3240,
                                 includes_raw_i2c_or_usb_overhead=False),
        elapsed_desktop_s=time.perf_counter()-started)
    assert result['production_sha_after']==before
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    ax=axes[0,0]
    for pitch,color in zip([0,20,35,45],['tab:blue','tab:orange','tab:green','tab:red']):
        for i in [0,7]:
            direction=(rays()@pitch_rotation(pitch).T)[i,3,0]
            if direction[2]<0:
                end=direction*(nominal_height/-direction[2])
                ax.plot([0,end[0]],[nominal_height,0],color=color,alpha=1 if i==7 else .4,
                        label=f'{pitch} deg' if i==7 else None)
    ax.axhline(0,color='black');ax.set(xlim=(0,1),ylim=(-.01,.32),xlabel='Ahead of sensor (m)',ylabel='Height (m)',title='Center-ray ground coverage; virtual sensor pitch')
    ax.legend(ncol=2)
    ax=axes[0,1]
    for kind in ['steps_10','steps_20','steps_30']:
        part=[r for r in rows if r['scene']==kind and r['pitch_deg']==35 and r['height_m']==nominal_height]
        x=[r['distance_to_first_edge_m'] for r in part]
        ax.plot(x,[r['first_tread_center_zones'] for r in part],label=kind+' ideal',alpha=.6)
        ax.plot(x,[r['zone_pose_1deg_dark'] for r in part],ls='--',label=kind+' separable')
    ax.set(xlabel='Sensor to first edge (m)',ylabel='First-tread zones',title='35 deg pitch: visible does not imply distinguishable')
    ax.legend(fontsize=8,ncol=2)
    for ax,key in [(axes[1,0],'steps_10_35'),(axes[1,1],'box_100_35')]:
        r,xyz,origin,rot=snapshots[key]
        heights=xyz[:,:,4,2]*1000
        im=ax.imshow(heights,vmin=0,vmax=100,cmap='viridis')
        for i in range(8):
            for j in range(8):
                if np.isfinite(heights[i,j]):
                    ax.text(j,i,f'{heights[i,j]:.0f}',ha='center',va='center',fontsize=8,color='white' if heights[i,j]<60 else 'black')
        ax.set(title=key+' at 0.30 m; IDEAL height mm',xlabel='Zone column',ylabel='Zone row')
        fig.colorbar(im,ax=ax,label='Ideal endpoint height (mm)')
    fig.savefig(OUT/'geometry_and_limits.png',dpi=160)
    plt.close(fig)
    print(json.dumps(dict(frames=len(rows),output=str(OUT),elapsed_s=result['elapsed_desktop_s']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
