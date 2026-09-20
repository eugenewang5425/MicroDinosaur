"""Plot measured training telemetry; reward is not a behavior acceptance gate."""
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).parent / '20260913_handoff'
log = (root / 'calibration_v07.log').read_text(encoding='utf-8')
blocks = re.split(r'Learning iteration\s+(\d+)/\d+', log)
rows = []
for i in range(1, len(blocks), 2):
    block = blocks[i + 1]
    reward = re.search(r'Mean reward:\s+([-\d.]+)', block)
    length = re.search(r'Mean episode length:\s+([-\d.]+)', block)
    if reward and length:
        rows.append((int(blocks[i]), float(reward[1]), float(length[1]) * .02))
a = np.array(rows)
assert np.isfinite(a).all() and len(a) > 0
fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, constrained_layout=True)
axes[0].plot(a[:, 0] - 15500, a[:, 1], color='#2979A3', linewidth=1.3)
axes[0].set_ylabel('Mean reward')
axes[0].set_title('MicroDinosaur v07 | bounded calibration from v7 checkpoint')
axes[1].plot(a[:, 0] - 15500, a[:, 2], color='#2C8C65', linewidth=1.3)
axes[1].axhline(20, color='gray', linestyle='--', linewidth=1, label='Episode limit: 20 s')
axes[1].set_ylabel('Mean episode length (s)')
axes[1].set_xlabel('Additional training iteration (zero-based)')
axes[1].legend(loc='lower right')
for ax in axes:
    ax.grid(alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
fig.savefig(root / 'training_curve.png', dpi=180)
print(f'Plotted {len(a)} logged iterations; last reward {a[-1,1]:.2f}, episode {a[-1,2]:.2f}s')
