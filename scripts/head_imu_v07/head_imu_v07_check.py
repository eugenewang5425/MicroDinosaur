"""Replay the v07 static delta audit with the original local frozen basis."""
from head_imu_v07_geometry import *
import itertools,hashlib
D=json.loads((P/'delta.json').read_text());N=np.load(P/'delta_meshes.npz')
old={n:sol(n) for n in B['objects'] if n+'__v' in A};new=dict(old);new.update({n:sol(n,N) for n in D['parts']})
changed=set(D['parts']);bounds={n:bb(s) for n,s in new.items()};hits=[];inherited=[];tests=0
evidence=dict(delta_sha256=hashlib.sha256((P/'delta.json').read_bytes()).hexdigest(),delta_meshes_sha256=hashlib.sha256((P/'delta_meshes.npz').read_bytes()).hexdigest())
for a,b in itertools.combinations(new,2):
 if not ({a,b}&changed) or not overlaps(bounds[a],bounds[b]):continue
 tests+=1;v=(new[a]^new[b]).volume()
 if v<=.02:continue
 prev=(old[a]^old[b]).volume() if a in old and b in old else 0
 row=dict(a=a,b=b,volume_mm3=v,before_mm3=prev,bounds_mm=bb(new[a]^new[b]).tolist())
 (hits if v>prev+.02 else inherited).append(row)
reservations=[]
for n,r in D['reserves'].items():
 s=box(r['center_mm'],r['size_mm']);h=[]
 for on,o in new.items():
  if on.startswith('JY61P_HEAD'):continue
  if overlaps(bb(s),bounds[on]) and (s^o).volume()>.02:h.append(dict(part=on,volume_mm3=(s^o).volume()))
 reservations.append(dict(name=n,hits=h))
# Pilot tool: final assembly remove top shell before access. Downward nut insertion checked separately.
paths=[];tools=[]
for sign in [-1,1]:
 nn='JY61P_HEAD_M2_NUT_'+str(sign);h=[]
 for dz in np.arange(-8,.01,.5):
  s=new[nn].translate((0,0,dz))
  for on,o in new.items():
   if on in [nn,'JY61P_HEAD_M2x6_'+str(sign)] or not overlaps(bb(s),bounds[on]):continue
   v=(s^o).volume()
   if v>.02:h.append(dict(dz=float(dz),part=on,volume_mm3=v))
 paths.append(dict(nut=nn,hits=h))
 probe=cyl((D['board_bottom_center_mm'][0],sign*10.5,276),2.2,24)
 for on,o in new.items():
  if on=='Rex_Skull_Shell' or on.startswith('JY61P_HEAD_M2x6_'):continue
  if overlaps(bb(probe),bounds[on]) and (probe^o).volume()>.02:tools.append(dict(part=on,volume_mm3=(probe^o).volume()))
q=dict(**evidence,tests=tests,new_intersections=hits,inherited_contacts=inherited,reservations=reservations,nut_insertion_paths=paths,tool_hits=tools)
(P/'static_checks.json').write_text(json.dumps(q,indent=2))
print(json.dumps(q,indent=2))
