"""Frozen-policy yaw/strafe command response with paired initial states."""
import argparse
import json
from pathlib import Path

import numpy as np
from evaluate_policy import Sim, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', action='append', required=True)
    p.add_argument('--plant', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    commands = [(f'forward_yaw_{w:+.1f}', [.55,0,w]) for w in (-.9,-.3,0,.3,.9)]
    commands += [('strafe_left',[0,.25,0]), ('strafe_right',[0,-.25,0])]
    for policy in args.policy:
        label, onnx = policy.split('=',1)
        dest = out / f'{label}.json'
        if dest.exists():
            raise FileExistsError(dest)
        sim = Sim(args.plant, onnx, command_lag=2, obs_lag=1, velocity_lag=1)
        results = []
        for name, twist in commands:
            for seed in range(3):
                cmd = np.zeros(18); cmd[:3] = twist
                sim.reset(seed, seed > 0)
                for _ in range(100):
                    sim.step(cmd)
                points = [sim.sample()]
                for _ in range(300):
                    sim.step(cmd); points.append(sim.sample())
                a = np.asarray(points)
                heading = a[0,3]
                rotation = np.array([[np.cos(heading),np.sin(heading)],[-np.sin(heading),np.cos(heading)]])
                xy = (rotation @ (a[:,:2]-a[0,:2]).T).T
                yaw = np.unwrap(a[:,3])
                results.append({'name':name, 'command':twist, 'seed':seed,
                    'vx_body_m_s':float(a[:,4].mean()), 'vy_body_m_s':float(a[:,5].mean()),
                    'yaw_drift_deg':float(np.rad2deg(yaw[-1]-yaw[0])),
                    'yaw_rate_rad_s':float((yaw[-1]-yaw[0])/6),
                    'lateral_mm':float(xy[-1,1]*1000),
                    'trajectory_xy_m':xy[::5].tolist(),
                    'fell':bool(np.any(a[:,2]<.06) or np.any(a[:,6]>60))})
        report = {'policy':label, 'onnx_sha256':sha(onnx),
                  'plant_sha256':sha(Path(args.plant)/'nominal.mjb'),
                  'command_ms':10,'position_ms':20,'velocity_ms':20,'imu_ms':0,
                  'action_alpha':sim.contract['action_filter']['alpha_new'],
                  'warmup_seconds':2,'measurement_seconds':6,'trials':results}
        dest.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({'policy':label,'trials':len(results),'falls':sum(r['fell'] for r in results)}),flush=True)


if __name__ == '__main__':
    main()
