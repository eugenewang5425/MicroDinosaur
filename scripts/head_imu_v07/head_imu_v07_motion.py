"""Delta-scoped collision checks; require local frozen basis and pose arrays."""
from head_imu_v07_geometry import *
import hashlib
D=json.loads((P/'delta.json').read_text());N=np.load(P/'delta_meshes.npz')
old={n:sol(n) for n in B['objects'] if n+'__v' in A};new=dict(old);new.update({n:sol(n,N) for n in D['parts']})
changed=set(D['parts']);bounds={n:bb(s) for n,s in new.items()}
evidence={k:hashlib.sha256((P/f).read_bytes()).hexdigest() for k,f in [('delta_sha256','delta.json'),('delta_meshes_sha256','delta_meshes.npz')]}
parents={n:B['objects'][n]['parent'] if n in B['objects'] else D['parent'] for n in new}
names=list(new);idx={n:i for i,n in enumerate(names)}
corners=np.array([[[x,y,z,1] for x in bounds[n][:,0] for y in bounds[n][:,1] for z in bounds[n][:,2]] for n in names])
for mode,infile,array in [('regression','source_regression_poses.json','source_regression_transforms.npz'),('legs','leg_poses.json','leg_transforms.npz'),('head','dense_head_input.json','dense_head_transforms.npz')]:
 root=R/'work_in_progress/lower_module_v05';info=json.loads((root/infile).read_text());ts=np.load(root/array)['transforms'];gi={n:i for i,n in enumerate(info['groups'])}
 missing=sorted(set(parents.values())-set(gi),key=str)
 if mode!='head':assert not missing,missing
 offset=len(gi);gi.update({n:offset+i for i,n in enumerate(missing)})
 ts=np.concatenate([ts,np.tile(np.eye(4),(len(ts),len(missing),1,1))],axis=1)
 gix=np.array([gi[parents[n]] for n in names]);hits=[];prior=[];tests=0;cache={}
 for pi in range(len(ts)):
  v=np.einsum('nij,nkj->nki',ts[pi,gix],corners)[:,:,:3];pb=np.stack([v.min(1),v.max(1)],1);pos={}
  def obj(n):
   if n not in pos:pos[n]=new[n].transform(ts[pi,gi[parents[n]]][:3])
   return pos[n]
  for n in changed:
   k=idx[n];mask=np.all(np.minimum(pb[:,1],pb[k,1])-np.maximum(pb[:,0],pb[k,0])>.002,axis=1)
   for j in np.flatnonzero(mask):
    on=names[j];ta=ts[pi,gi[parents[n]]];tb=ts[pi,gi[parents[on]]]
    if on==n or np.max(abs(ta-tb))<1e-7:continue
    key=(n,on,ta.tobytes(),tb.tobytes())
    if key not in cache:
     tests+=1;vol=(obj(n)^obj(on)).volume();before=0
     if vol>.02 and n in old and on in old:before=(old[n].transform(ta[:3])^old[on].transform(tb[:3])).volume()
     cache[key]=(vol,before)
    vol,before=cache[key]
    if vol>.02:(hits if vol>before+.02 else prior).append(dict(pose_index=pi,part=n,other=on,volume_mm3=vol,before_mm3=before))
  if pi%150==0:print(mode,pi,'new',len(hits),flush=True)
 q=dict(**evidence,poses=len(ts),tests=tests,new_or_worsened_intersections=hits,inherited_intersections=prior,pose_source=str(root/infile),scope='Changed carrier and all new parts against current assembly; relative-static pairs checked separately',continuous_collision_release=False)
 (P/(mode+'_checks.json')).write_text(json.dumps(q,indent=2));print(mode,'DONE',len(hits),hits[:15],flush=True)
# Exact nominal clearances, not print tolerance/deflection guarantees.
gaps=[]
for n in D['added']:
 gaps.append(dict(part=n,other='Rex_Skull_Shell',gap_mm=new[n].min_gap(new['Rex_Skull_Shell'],10)))
added_carrier=new['Rex_Camera_Pi_Carrier']-old['Rex_Camera_Pi_Carrier']
for on in ['Rex_Skull_Shell','V10_PI_ZERO_2W_PCB']:
 gaps.append(dict(part='carrier_added_material_only',other=on,gap_mm=added_carrier.min_gap(new[on],10)))
# Remove shell, clips and bolts first; board+cap can be lifted straight out.
removed={'Rex_Skull_Shell'}|{n for n in D['added'] if 'CLIP' in n or 'M2x6' in n};board=new['JY61P_HEAD_PCB_ENVELOPE']+new['JY61P_HEAD_CAP_ENVELOPE'];lift=[]
for dz in np.arange(0,20.01,.5):
 s=board.translate((0,0,dz))
 for on,o in new.items():
  if on in removed or on in ['JY61P_HEAD_PCB_ENVELOPE','JY61P_HEAD_CAP_ENVELOPE']:continue
  if overlaps(bb(s),bounds[on]) and (s^o).volume()>.02:lift.append(dict(dz_mm=float(dz),other=on,volume_mm3=(s^o).volume()))
q=dict(**evidence,gaps=gaps,board_removal_samples=41,board_removal_lift_mm=20,board_removal_hits=lift,headers_not_installed_or_verified=True)
(P/'clearance_checks.json').write_text(json.dumps(q,indent=2));print('CLEARANCE',q,flush=True)
