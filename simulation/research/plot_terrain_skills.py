"""Static research figures from the saved confirmation traces."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from terrain_skill_eval import OUT, ground_height


def main():
    labels=['v7_reference','terrain_15650','crouch_15650']
    colors=['#65758b','#287d67','#cf752d']
    fig,axes=plt.subplots(2,3,figsize=(14,7),sharex='col',constrained_layout=True)
    cases=[('crouch','flat','crouch20','stand'),('crouch','flat','crouch20','straight'),('terrain','slope_5','none','straight')]
    for col,(suite,terrain,posture,scenario) in enumerate(cases):
        for label,color in zip(labels,colors):
            p=OUT/'evaluation'/label/suite/f'{terrain}__{posture}__{scenario}__s1.npz'
            if not p.exists():continue
            data=np.load(p);g=data['gaze'];g=g[g[:,0]>=6];t=g[:,0]-6
            height=(g[:,3]-ground_height(terrain,g[:,1],g[:,2]))*1000
            axes[0,col].plot(t[::16],height[::16],color=color,label=label,lw=1.3)
            trace=data['trace'];axes[1,col].plot(trace[:,0],trace[:,14],color=color,lw=1,alpha=.85)
        cmd=np.load(OUT/'evaluation/v7_reference'/suite/f'{terrain}__{posture}__{scenario}__s1.npz')['commands']
        axes[0,col].plot(cmd[:,0],117.182+1000*cmd[:,1],ls='--',color='black',label='height command',lw=1)
        axes[0,col].set_title(f'{terrain} / {posture} / {scenario} (seed 1)')
        axes[0,col].legend(loc='lower left',fontsize=8)
        axes[1,col].set_xlabel('Time after calibration (s)')
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.set_xlim(0,12)
    axes[0,0].set_ylabel('Body clearance above local ground (mm)')
    axes[1,0].set_ylabel('Body forward speed (m/s)')
    fig.suptitle('Current v07 + S288 positive delays + body/head IMU control\nSaved trajectories; terrain candidate and crouch candidate are separate continuations')
    fig.savefig(OUT/'confirmation_traces.png',dpi=150);fig.savefig(OUT/'confirmation_traces.svg');plt.close(fig)


if __name__=='__main__':main()
