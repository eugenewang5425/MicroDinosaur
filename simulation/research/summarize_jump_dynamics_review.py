"""Shared-window work comparison and diagnostic figures from exact archived replays."""
import json
from pathlib import Path
import mujoco
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'20260914_jump_dynamics_review'


def main():
    audit=json.loads((OUT/'audit.json').read_text());result={}
    plant=ROOT/'20260914_run_jump/plant/nominal.mjb'
    model=mujoco.MjModel.from_binary_path('x.mjb',assets={'x.mjb':plant.read_bytes()})
    proxies=[g for g in range(model.ngeom) if model.geom(g).name.endswith('_floor_proxy')]
    data=mujoco.MjData(model)
    fig,axes=plt.subplots(4,2,figsize=(11,10),layout='constrained')
    for col,depth in enumerate([30,50]):
        end=audit[f'previous_c{depth}']['metrics']['flight']['takeoff_s']
        for label,color in [('previous','#687789'),('candidate','#117d6f')]:
            key=f'{label}_c{depth}';r=audit[key];d=np.load(OUT/(key+'.npz'));a=d['physics'];t=a[:,0]
            names=[x['name'] for x in r['model']['actuators']]
            ids=[i for i,n in enumerate(names) if n.startswith(('left_','right_')) and n.endswith(('hip_pitch','knee','ankle'))]
            mask=(t>=1.8)&(t<end);tau=d['joint_actuator_torque'][:,ids];v=d['qvel_pre'][:,ids]
            power=tau*v;b=d['bounds'][:,ids];dt=r['model']['physics_dt']
            sat=(tau<=b[:,:,0]+1e-6)|(tau>=b[:,:,1]-1e-6)
            poses=np.load(Path(r['source']).with_suffix('.npz'))['qpos'];lowest=[]
            for q in poses:
                data.qpos[:]=q;mujoco.mj_forward(model,data)
                R=data.geom_xmat[proxies].reshape(-1,3,3)
                lowest.append(data.geom_xpos[proxies,2]-(abs(R[:,2,:])*model.geom_size[proxies]).sum(-1))
            minimum=np.asarray(lowest).min(axis=0);gid=proxies[int(minimum.argmin())]
            result[key]=dict(shared_window_s=[1.8,end],same_window_as_previous_pre_takeoff=True,
                major_joints=[names[i] for i in ids],
                positive_work_j=float(np.maximum(power[mask],0).sum()*dt),
                negative_work_j=float(-np.minimum(power[mask],0).sum()*dt),
                dynamic_saturation_fraction=float(sat[mask].mean()),
                negative_net_power_time_fraction=float((power[mask].sum(-1)<0).mean()),
                passive_work_j=float((d['passive_force'][mask][:,ids]*v[mask]).sum()*dt),
                sampled_nonfoot_proxy_min_gap_mm=float(minimum.min()*1000),lowest_proxy=model.geom(gid).name,
                sampled_pose_count=len(poses),geometry_check_is_discrete_only=True)
            view=(t>=1.75)&(t<2.18)
            values=[d['com_velocity'][:,2],a[:,4],a[:,14:16].min(-1)*1000,power.sum(-1)]
            for row,series in enumerate(values):
                axes[row,col].plot(t[view],series[view],label=label,color=color)
        axes[0,col].set_title(f'{depth} mm crouch / seed 601 / 10 ms / 12 V')
        axes[0,col].axhline(0,color='#777777',lw=.6)
        axes[1,col].axhline(1.0982148*9.81,color='#777777',ls=':',label='Weight')
        axes[2,col].axhline(5,color='#a05d36',ls='--',label='5 mm clearance')
        axes[3,col].axhline(0,color='#777777',lw=.6)
        for row in range(4):
            axes[row,col].set_ylabel(['CoM vertical velocity [m/s]','Ground support [N]',
                'Minimum of both whole feet [mm]','Six leg actuators net power [W]'][row])
            axes[row,col].set_xlabel('Simulation time [s]');axes[row,col].grid(alpha=.2);axes[row,col].legend(fontsize=8)
    fig.savefig(OUT/'power_diagnostic.png',dpi=150);plt.close(fig)
    (OUT/'shared_window.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
