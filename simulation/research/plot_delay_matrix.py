"""Comparable heatmaps for frozen-policy delay probes (equal feedback delays)."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--matrix', action='append', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
rows = [r for p in args.matrix for r in json.loads(Path(p).read_text())]
policies = list(dict.fromkeys(r['policy'] for r in rows))
metrics = [('stand', 'head_angular_speed_rms_deg_s', 'Standing head speed RMS (deg/s)', False),
           ('stand', 'yaw_drift_deg', 'Standing yaw drift (deg / 5 s)', True),
           ('forward', 'lateral_displacement_mm', 'Forward lateral displacement (mm / 6 s)', True)]
fig, axes = plt.subplots(len(policies), len(metrics), figsize=(13, 3.4*len(policies)), squeeze=False)
for col, (task, metric, title, absolute) in enumerate(metrics):
    grids = []
    for label in policies:
        grid = np.full((4, 3), np.nan)
        for r in rows:
            if r['policy'] == label and r['position_ms'] == r['velocity_ms']:
                values = [t[metric] for t in r['trials'][task]]
                grid[r['command_ms']//5, r['position_ms']//20] = np.mean(np.abs(values) if absolute else values)
        grids.append(grid)
    vmax = max(float(np.nanmax(g)) for g in grids)
    for i, (label, grid) in enumerate(zip(policies, grids)):
        ax = axes[i, col]
        im = ax.imshow(grid, cmap='YlOrRd', vmin=0, vmax=vmax, aspect='auto')
        for (y, x), v in np.ndenumerate(grid):
            ax.text(x, y, f'{v:.1f}', ha='center', va='center', color='white' if v > vmax*.65 else 'black')
        ax.set_xticks(range(3), [0, 20, 40]); ax.set_yticks(range(4), [0, 5, 10, 15])
        ax.set_xlabel('Position + velocity feedback delay (ms)')
        ax.set_ylabel('Motor target delay (ms)')
        ax.set_title(f'{label} | {title}', fontsize=10)
        fig.colorbar(im, ax=ax, fraction=.035)
fig.suptitle('Same v07 nominal plant | 3 seeds | no sensor noise | fresh IMU | lower is better', fontsize=13)
fig.tight_layout(rect=(0, 0, 1, .96))
fig.savefig(args.out, dpi=170)
