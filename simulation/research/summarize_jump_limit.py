"""Flight-height/energy audit of the frozen-policy ceiling study."""
import csv
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evaluate_policy import sha
from jump_limit_probe import ROOT,OUT,JUMP


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    scan=json.loads((OUT/'scan/matrix.json').read_text())
    confirmation=json.loads((OUT/'confirm/matrix.json').read_text())
    assert len(scan)==34 and len(confirmation)==14
    assert all(r['status']=='COMPLETE' for r in scan+confirmation)
    best=max((r for r in scan if r['metrics']['jump_success']),key=lambda r:r['metrics']['max_qualified_com_height_m'])
    m=best['metrics'];energy=m['energetics'];mass=m['mass_kg'];g=9.81
    assert sha(JUMP)==best['policy_sha256']
    previous=ROOT/'20260914_run_jump/evaluation/jump_return/jump_v0.6_d10_s101_dt0.00125_curve0_V12.6.npz'
    baseline=next(r for r in scan if r['recipe']==dict(crouch=.03,extension=.02,preparation_s=1.5,extension_s=.7))
    old=np.load(previous)['physics'];now=np.load(OUT/'scan'/(baseline['name']+'.npz'))['physics']
    difference=float(abs(old-now).max());assert difference==0
    a=np.load(OUT/'scan'/(best['name']+'.npz'))['physics']
    dt=best['hardware']['physics_dt'];i=np.argmin(abs(a[:,0]-best['recipe']['preparation_s']));j=np.argmin(abs(a[:,0]-energy['takeoff_s']))
    vz=np.gradient(a[:,3],dt)
    impulse=float(((a[i:j,4]-mass*g)*dt).sum());momentum=float(mass*(vz[j]-vz[i]))
    force=dict(mean_push_support_n=float(a[i:j,4].mean()),peak_push_support_n=float(a[i:j,4].max()),
               net_contact_impulse_ns=impulse,com_momentum_change_ns=momentum,
               relative_impulse_balance_error=abs(impulse-momentum)/max(abs(momentum),1e-12))
    assert force['relative_impulse_balance_error']<.02
    assert abs(energy['fitted_airborne_acceleration_m_s2']+g)<.05
    assert abs(energy['ballistic_height_m']-energy['measured_com_height_m'])<.00005
    required=[dict(height_cm=h*100,takeoff_velocity_m_s=float(np.sqrt(2*g*h)),
                   vertical_energy_j=mass*g*h,net_vertical_impulse_ns=float(mass*np.sqrt(2*g*h)),
                   energy_multiple_of_best=mass*g*h/energy['vertical_takeoff_energy_j'])
              for h in [.01,.02,.03,.05,.10]]
    delay_groups=[]
    for delay in [5,10,15]:
        group=[r for r in confirmation if r['seed'] in [301,302,303] and r['hardware']['command_ms']==delay]
        heights=[r['metrics']['max_qualified_com_height_m']*1000 for r in group]
        delay_groups.append(dict(delay_ms=delay,planned=len(group),success=sum(r['metrics']['jump_success'] for r in group),
                                 min_height_mm=min(heights),max_height_mm=max(heights),mean_height_mm=float(np.mean(heights))))
    conditions=[]
    for stage,records in [('scan',scan),('confirm',confirmation)]:
        for r in records:
            q=r['metrics']
            conditions.append(dict(stage=stage,name=r['name'],**r['recipe'],seed=r['seed'],delay_ms=r['hardware']['command_ms'],
                hardware_name=r['hardware']['name'],qualified_height_mm=q['max_qualified_com_height_m']*1000,
                measured_height_in_any_clear_segment_mm=1000*max([s['com_apex_above_takeoff_m'] for s in q['flight_segments']],default=0),
                foot_clearance_mm=q['maximum_both_foot_clearance_m']*1000,
                flight_ms=q['max_clear_flight_s']*1000,full_success=q['jump_success'],
                limit_excess_rad=q['joint_limit_excess_rad'],penetration_mm=q['maximum_mesh_floor_penetration_m']*1000))
    data=dict(best_development=best,development_count=len(scan),development_passed=sum(r['metrics']['jump_success'] for r in scan),
        confirmation_count=len(confirmation),delay_groups=delay_groups,force_impulse_audit=force,
        height_requirements=required,instrumentation_baseline_max_difference=difference,
        instrumentation_baseline_input_sha256=sha(previous),
        total_kinetic_as_vertical_equivalent_height_m=energy['kinetic_at_takeoff_j']/(mass*g),
        note='Equivalent energy heights are bookkeeping, not achievable or global jump limits; airborne internal motion cannot increase COM momentum.',
        conditions=conditions)
    (OUT/'summary.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    with (OUT/'conditions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(conditions[0]));writer.writeheader();writer.writerows(conditions)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    grid=np.array([r['metrics']['max_qualified_com_height_m']*1000 for r in scan[:25]]).reshape(5,5)
    im=axes[0].imshow(grid,origin='lower',vmin=0,vmax=10,cmap='viridis')
    for k,r in enumerate(scan[:25]):
        y,x=divmod(k,5);label=f'{grid[y,x]:.1f}'+(' x' if not r['metrics']['jump_success'] else '')
        axes[0].text(x,y,label,ha='center',va='center',color='white' if grid[y,x]<5 else 'black')
    axes[0].set(xticks=range(5),xticklabels=[0,10,20,30,40],yticks=range(5),yticklabels=[20,25,30,35,40],
                xlabel='Extension request (mm)',ylabel='Crouch request (mm)',title='COM jump height (mm); x = full trial failed')
    fig.colorbar(im,ax=axes[0],label='Qualified COM rise after takeoff (mm)')
    for r in confirmation[:9]:
        q=r['metrics'];axes[1].scatter(r['hardware']['command_ms'],1000*q['max_qualified_com_height_m'],
            color='tab:blue' if q['jump_success'] else 'tab:red',marker='o' if q['jump_success'] else 'x',s=55)
    axes[1].axhline(1000*m['max_qualified_com_height_m'],ls='--',color='gray',label='Best development sample')
    axes[1].set(xticks=[5,10,15],xlabel='Motor command delay (ms)',ylabel='COM rise after takeoff (mm)',
                ylim=(0,10),title='Frozen best recipe: new seeds 301-303')
    axes[1].grid(alpha=.2);axes[1].legend()
    fig.savefig(OUT/'jump_limit.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(best_height_mm=1000*m['max_qualified_com_height_m'],best_recipe=best['recipe'],
        best_takeoff_velocity=energy['com_takeoff_velocity_m_s'],best_vertical_energy=energy['vertical_takeoff_energy_j'],
        delay_groups=delay_groups,force_audit=force,requirements=required),indent=2))


if __name__=='__main__':main()
