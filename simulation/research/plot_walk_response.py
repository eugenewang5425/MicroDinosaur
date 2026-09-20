"""Command response and straight-command trajectories; ranges are initial states."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--input', required=True)
p.add_argument('--out', required=True)
args = p.parse_args()
fig, axes = plt.subplots(1,3,figsize=(14,4.4))
for index, path in enumerate(sorted(Path(args.input).glob('*.json'))):
    d = json.loads(path.read_text())
    label, rows = d['policy'], d['trials']
    yaw = [-.9,-.3,0,.3,.9]
    values = np.array([[r['yaw_rate_rad_s'] for r in rows
                       if r['command']==[.55,0,w]] for w in yaw])
    mean = values.mean(axis=1)
    axes[0].errorbar(yaw,mean,yerr=[mean-values.min(axis=1),values.max(axis=1)-mean],
                     marker='o',capsize=3,label=label)
    straight = [r for r in rows if r['command']==[.55,0,0]]
    for i,r in enumerate(straight):
        xy = np.array(r['trajectory_xy_m'])
        axes[1].plot(xy[:,0],xy[:,1],color=f'C{index}',alpha=.7,
                     label=label if i==0 else None)
    lateral = [[r['vy_body_m_s'] for r in rows if r['name']==n]
               for n in ('strafe_left','strafe_right')]
    axes[2].bar(np.arange(2)+(index-1)*.23,np.mean(lateral,axis=1),width=.22,label=label)
axes[0].plot([-.9,.9],[-.9,.9],'k--',lw=1,label='ideal')
axes[0].set(xlabel='Command yaw rate (rad/s), vx=0.55',ylabel='Measured yaw rate (rad/s)',title='Yaw sign and tracking bias')
axes[1].axhline(0,color='k',linestyle='--',lw=1)
axes[1].set(xlabel='Forward displacement (m)',ylabel='Lateral displacement (m)',title='Zero-yaw command: 6-second paths')
axes[1].set_aspect('equal',adjustable='datalim')
axes[2].axhline(.25,color='k',linestyle='--',lw=1)
axes[2].axhline(-.25,color='k',linestyle='--',lw=1)
axes[2].set(xticks=[0,1],xticklabels=['Left +0.25','Right -0.25'],ylabel='Body lateral speed (m/s)',title='Pure strafe command response')
for ax in axes:
    ax.grid(alpha=.2); ax.legend(fontsize=8)
fig.suptitle('Nominal v07 | motor/position/velocity = 10/20/20 ms | 3 initial states; ranges are not confidence intervals',fontsize=11)
fig.tight_layout()
fig.savefig(args.out,dpi=170)
