"""200 Hz control/joint traces of one frozen policy; no reward computation."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from evaluate_policy import Sim

parser = argparse.ArgumentParser()
parser.add_argument('--plant', required=True)
parser.add_argument('--onnx', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
metrics = []
for i, lag in enumerate((0, 2)):
    sim = Sim(args.plant, args.onnx, command_lag=lag)
    records = []
    def record():
        velocity = np.zeros(6)
        # Derived cvel is from the start of the just-integrated 5 ms substep.
        # Do not add forward/solver calls that could affect the rollout.
        mujoco.mj_objectVelocity(sim.model, sim.data, mujoco.mjtObj.mjOBJ_BODY, sim.head, velocity, 0)
        records.append(np.r_[sim.data.time, velocity[:3], sim.data.qvel[sim.vadr],
                             sim.data.qpos[sim.jadr], sim.data.ctrl[sim.aids]])
    sim.substep_callback = record
    for _ in range(750):
        sim.step(np.zeros(18))
    trace = np.asarray(records)
    np.savez_compressed(out/f'command_{lag*5}ms.npz', trace=trace, joint_names=sim.names)
    measured = trace[trace[:, 0] >= 5.]
    head = measured[:, 1:4] - measured[:, 1:4].mean(axis=0)
    freq = np.fft.rfftfreq(len(head), sim.model.opt.timestep)
    window = np.hanning(len(head))
    spectrum = (np.abs(np.fft.rfft(head*window[:, None], axis=0))**2).sum(axis=1)
    selected = (freq >= 5) & (freq <= 100)
    peak = float(freq[selected][np.argmax(spectrum[selected])])
    rms = float(np.rad2deg(np.sqrt(np.mean(np.sum(measured[:, 1:4]**2, axis=1)))))
    metrics.append({'command_ms': lag*5, 'position_ms': 0, 'velocity_ms': 0,
                    'head_rms_deg_s': rms, 'largest_5_100hz_peak_hz': peak,
                    'measurement_seconds': 10, 'warmup_seconds': 5,
                    'derived_head_velocity_latency_ms': 5})
    # Show 0.5 s at the same scale, and a normalized spectral comparison.
    segment = measured[(measured[:, 0] >= 8) & (measured[:, 0] < 8.5)]
    for j, label in enumerate(('world X', 'world Y', 'world Z')):
        axes[i, 0].plot(segment[:, 0], np.rad2deg(segment[:, j+1]), label=label, linewidth=1)
    axes[i, 0].set_title(f'Motor target delay {lag*5} ms | head speed RMS {rms:.2f} deg/s')
    axes[i, 0].set_xlabel('Simulation time (s)'); axes[i, 0].set_ylabel('Head angular velocity (deg/s)')
    axes[i, 0].set_ylim(-65, 65); axes[i, 0].legend(fontsize=8)
    axes[i, 1].semilogy(freq[1:], spectrum[1:]/(window**2).sum()+1e-14)
    axes[i, 1].set_title(f'Largest peak above 5 Hz: {peak:.1f} Hz')
    axes[i, 1].set_xlabel('Frequency (Hz)'); axes[i, 1].set_ylabel('Windowed spectral power (common units)')
    axes[i, 1].set_xlim(0, 100); axes[i, 1].set_ylim(1e-12, 100)
    axes[i, 1].axvline(25, color='gray', linestyle='--', label='Policy Nyquist')
fig.suptitle('Same frozen candidate | fresh position/velocity feedback | 200 Hz physics sampling')
fig.tight_layout(rect=(0, 0, 1, .95))
fig.savefig(out/'standing_spectrum.png', dpi=170)
(out/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
print(json.dumps(metrics, indent=2))
