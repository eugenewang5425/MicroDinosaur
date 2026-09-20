"""Frozen-policy delay ablations; no learning or dynamics randomization."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from evaluate_policy import Sim, run_trial


class TracedSim(Sim):
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.trace = []

    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)
        head_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                self.head, head_velocity, 0)
        self.trace.append(np.r_[head_velocity[:3], self.raw.copy(), self.applied.copy()])


def trial(sim, command, seed, seconds):
    result = run_trial(sim, command, seed=seed, perturb=seed > 0, seconds=seconds)
    trace = np.asarray(sim.trace)[-round(seconds / sim.dt):]
    frequencies = np.fft.rfftfreq(len(trace), sim.dt)
    spectrum = np.fft.rfft(trace[:, :3], axis=0)
    spectrum[frequencies < 5] = 0
    high = np.fft.irfft(spectrum, n=len(trace), axis=0)
    result['head_5_25hz_rms_deg_s'] = float(np.rad2deg(np.sqrt(np.mean(np.sum(high**2, axis=1)))))
    result['raw_action_step_rms_rad'] = float(np.sqrt(np.mean(np.diff(trace[:, 3:22], axis=0)**2)))
    result['target_step_rms_rad'] = float(np.sqrt(np.mean(np.diff(trace[:, 22:], axis=0)**2)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--policy', action='append', required=True, help='label=onnx_path')
    parser.add_argument('--plant', required=True)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--native-extra-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = [(c, o, o) for c in range(4) for o in range(3)]
    # Separate position-feedback from velocity-feedback at 0 and 10 ms actuation lag.
    settings += [(c, p, v) for c in (0, 2) for p, v in ((0, 1), (1, 0), (2, 1))]
    native_extra = [(1, 2, 1), (3, 2, 1)]
    settings = native_extra if args.native_extra_only else settings + native_extra
    started = time.time()
    all_results = []
    for policy in args.policy:
        label, onnx = policy.split('=', 1)
        for command_lag, position_lag, velocity_lag in settings:
            path = out / f'{label}_c{command_lag}_p{position_lag}_v{velocity_lag}.json'
            if path.exists():
                if not args.resume:
                    raise FileExistsError(f'Refusing to overwrite {path}')
                existing = json.loads(path.read_text())
                assert existing['onnx_sha256'] == hashlib.sha256(Path(onnx).read_bytes()).hexdigest()
                assert existing['plant_sha256'] == hashlib.sha256((Path(args.plant)/'nominal.mjb').read_bytes()).hexdigest()
                assert (existing['command_ms'], existing['position_ms'], existing['velocity_ms']) == (command_lag*5, position_lag*20, velocity_lag*20)
                assert all(len(ts) == args.seeds for ts in existing['trials'].values())
                contract = json.loads((Path(args.plant)/'contract.json').read_text())
                assert existing.get('action_filter_alpha_new', .9) == contract['action_filter']['alpha_new']
                all_results.append(existing)
                continue
            sim = TracedSim(args.plant, onnx, command_lag, position_lag, velocity_lag)
            result = {'policy': label, 'onnx': onnx,
                      'onnx_sha256': hashlib.sha256(Path(onnx).read_bytes()).hexdigest(),
                      'plant_sha256': hashlib.sha256((Path(args.plant)/'nominal.mjb').read_bytes()).hexdigest(),
                      'command_ms': command_lag*5, 'position_ms': position_lag*20,
                      'velocity_ms': velocity_lag*20, 'imu_ms': 0,
                      'action_filter_alpha_new': sim.contract['action_filter']['alpha_new'],
                      'noise': False, 'random_dynamics': False, 'trials': {}}
            for name, speed, seconds in [('stand', 0, 5), ('forward', .55, 6)]:
                cmd = np.zeros(18); cmd[0] = speed
                result['trials'][name] = [trial(sim, cmd, seed, seconds) for seed in range(args.seeds)]
            path.write_text(json.dumps(result, indent=2), encoding='utf-8')
            all_results.append(result)
            print(json.dumps({'done': path.name, 'elapsed_s': round(time.time()-started),
                  'stand_head_rms': round(float(np.mean([t['head_angular_speed_rms_deg_s'] for t in result['trials']['stand']])), 3),
                  'forward_lateral_mm': round(float(np.mean([abs(t['lateral_displacement_mm']) for t in result['trials']['forward']])), 1)}), flush=True)
    (out/'matrix.json').write_text(json.dumps(all_results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
