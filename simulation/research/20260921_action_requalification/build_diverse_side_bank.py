"""Side-weighted collision-aware reset bank for action requalification (seeds 964-968).

Follows build_reset_bank_reference.py: every state is produced by a real physical
fall in the collision-enabled r7 plant (no random network rebuild, no state copying).
Standing and valid lean states are kept for every (direction, seed) pair. Full-fall
states are included on a seed subset so that left+right account for about 2/3 of
full-fall samples while front and back each still draw from at least 5 distinct
seeds overall. The audit_record check from the original bank is reused verbatim.
"""
from pathlib import Path
import sys,json,hashlib,copy
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'research'))
sys.path.insert(0,str(ROOT/'research/20260921_contact_reset_fix'))
from build_contact_motion_bank import prepare_fall,snapshot,constrain_recovery_head,FALL_DIRECTIONS
from hardware_sim import HardwareCase
from head_attitude_sim import HeadExperiment
from evaluate_run_jump import MotionExperiment
from head_attitude import HeadConfig
from imu_owned_head import configure_owned
from run_heading_stable_start import CALIBRATOR
from audit_reset_bank import audit_record

SEEDS=(964,965,966,967,968)
# Full-fall samples per direction use only these seeds; sides use all five so that
# left+right = 10 of 15 full-fall samples (2/3). Standing/lean states are kept for
# every seed, so front and back still span all five distinct seeds in the bank.
# Ten side falls and five sagittal falls make the promised 2/3 side emphasis.
# Front/back still retain standing and lean states from all five physical seeds.
FULL_FALL_SEEDS={'front':(964,965,966),'back':(967,968),'left':SEEDS,'right':SEEDS}
DEST_NAME='side_generation_r7s';BANK_NAME='side_getup_reset_bank_v071_r7s.json'


def main():
	plant=ROOT/'research/20260921_collision_v071/cadcol_v071_range_r7'
	dest=OUT/DEST_NAME;dest.mkdir(exist_ok=False)  # refuse to overwrite existing outputs
	records=[];audit=[];names=None
	for seed in SEEDS:
		for direction in FALL_DIRECTIONS:
			e=configure_owned(MotionExperiment(CALIBRATOR,'run',0,HardwareCase(motor_curve=True,voltage=12.,physics_dt=.00125),plant=plant,operating_envelope=True),(1,));e.yaw_follows_trunk=True;constrain_recovery_head(e)
			s=e.sim
			cal=e.reset(seed);names=s.names
			starting=snapshot(e,cal,'standing',seed)
			trace,frames=prepare_fall(e,direction,capture_tilts=True)
			nonfoot=float(trace[-1,5])/(float(s.model.body_mass.sum())*9.81)
			qualified=bool(trace[-1,2]>25 and trace[-1,3]<.05 and trace[-1,4]<.5 and trace[-1,5]>1 and trace[-1,6]<.002 and nonfoot>=.35)
			candidates=[starting,*e.lean_snapshots,snapshot(e,cal,direction,seed)]
			ids=[s.model.joint('robot/'+n).id for n in names]
			joint_map={'names':names,'ids':ids,'root_qposadr':0,'root_dofadr':0};act_map={j:i for i,j in enumerate(ids)}
			include_full_fall=seed in FULL_FALL_SEEDS[direction]
			accepted=[];rejected=[]
			for index,r in enumerate(candidates):
				r=copy.deepcopy(r);r['seed']=seed;r['calibration']=cal
				is_full_fall=r['label']==direction
				if r['label'].startswith('lean_'):r['label']=direction+'_'+r['label']
				if is_full_fall and not include_full_fall:
					rejected.append({'label':r['label'],'class':'sampling_weight_excluded','pairs':None,'errors':'full-fall seed not in subset for '+direction+'; standing/lean states retained'})
					continue
				checked=audit_record(s.model,joint_map,act_map,r,index)
				if checked['record_class']=='proxy_clear_reset_candidate' and (r['label']!=direction or qualified):
					r['sample_weight']=1.0;r['state_class']='standing' if r['label']=='standing' else ('full_fall' if is_full_fall else 'lean')
					records.append(r);accepted.append(r['label'])
				else:
					rejected.append({'label':r['label'],'class':checked['record_class'],'pairs':checked['pair_max_penetration_mm'],'errors':checked['errors']})
			np.savez_compressed(dest/f'{direction}_s{seed}.npz',trace=trace,qpos=frames,frames_dt=.04)
			audit.append({'direction':direction,'seed':seed,'accepted':accepted,'rejected':rejected,'full_fall_included':include_full_fall,'fallen_start_valid':qualified,'nonfoot_weight_fraction':nonfoot,'final_tilt_deg':float(trace[-1,2]),'final_speed':float(trace[-1,3]),'final_angular_speed':float(trace[-1,4])})
			(dest/'progress.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit[-1]),flush=True)
	assert names is not None and len(records)>=20
	full=[r for r in records if r['state_class']=='full_fall']
	side_full=sum(1 for r in full if r['label'] in ('left','right'))
	assert len(full)==15 and side_full==10, (side_full,len(full))
	assert all(any(r['seed']==sd for r in records if r['label'].split('_')[0]==d or (d=='front' and r['label']=='standing')) for d in ('front','back') for sd in SEEDS)
	counts={}
	for r in records:counts[r['label']]=counts.get(r['label'],0)+1
	bank={'action_names':names,'sample_count':len(records),'records':records,'plant_sha256':hashlib.sha256((plant/'nominal.mjb').read_bytes()).hexdigest(),
	 'source_sha256':hashlib.sha256(CALIBRATOR.read_bytes()).hexdigest(),'calibration_seconds':6.,'generation_seeds':list(SEEDS),
	 'per_label_counts':counts,
	 'full_fall_sampling':{'total_full_fall':len(full),'left_right_full_fall':side_full,'front_back_full_fall':len(full)-side_full,'side_fraction':side_full/len(full),
	   'full_fall_seeds':{k:list(v) for k,v in FULL_FALL_SEEDS.items()},
	   'note':'front/back keep 5 distinct seeds via standing+lean states; no duplicated states'},
	 'generation':'Collision-active physical stand, ramped push, release and settle; no external force after reset begins; side-weighted seed subset for full falls',
	 'reset_geometry':'Native r7 seven-pair proxy checked; CAD Boolean follow-up required; not full-machine clearance',
	 'head_pitch_command_range_deg':[3.,26.5],'provenance':DEST_NAME+'/progress.json','yaw_follows_trunk':True}
	(OUT/BANK_NAME).write_text(json.dumps(bank,indent=2));print('records',len(records),'full_fall',len(full),'side_full',side_full,flush=True)

if __name__=='__main__':main()
