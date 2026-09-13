"""Nominal JY61P head installation; geometry in reference-world millimetres."""
from pathlib import Path
import json, numpy as np, manifold3d as md
R=Path(__file__).resolve().parents[2];P=R/'work_in_progress/head_imu_v07'
if not (P/'basis.json').is_file() or not (P/'basis_meshes.npz').is_file():
 raise SystemExit('The v07 generation recipe requires the original local frozen basis, excluded from the latest-only repository. See scripts/head_imu_v07/README.md. Do not use current as the old basis.')
B=json.loads((P/'basis.json').read_text());A=np.load(P/'basis_meshes.npz')
M=md.Manifold;C=md.CrossSection
def sol(n,a=A):return M(md.Mesh(a[n+'__v'].astype(np.float32),a[n+'__f'].astype(np.uint32)))
def box(c,d):return M.cube(d,True).translate(c)
def cyl(c,r,h,n=48):return M.cylinder(h,r,circular_segments=n,center=True).translate(c)
def plate(c,w,h,t,r=.8):return C.square((w-2*r,h-2*r),True).offset(r,circular_segments=32).extrude(t).translate((c[0],c[1],c[2]-t/2))
def bb(s):return np.array(s.bounding_box()).reshape(2,3)
def overlaps(a,b):return bool(np.all(np.minimum(a[1],b[1])-np.maximum(a[0],b[0])>.002))
def descendants(root):
 s={root}
 while True:
  q=s|{n for n,r in B['objects'].items() if r['parent'] in s}
  if q==s:return s
  s=q
def build():
 x,y,z=35.,0.,260. # board bottom; shield uppermost=262.8
 # Open-center support; no rigid material under central solder/components.
 tray=plate((x,y,z-.8),19.5,26,1.6,1.5)-plate((x,y,z-.8),12,12,4,1)
 # Two short forward arms join the existing camera/Pi carrier, above Pi's top edge.
 arms=M()
 for yy in [y-7,y+7]:arms+=plate((49,yy,z-.8),34,3.5,1.6,1)
 for s in [-1,1]:
  cy=y+s*10.5
  tray+=plate((x,cy,z-1.6),7,6,3.2,1)
  tray+=plate((x,cy,z+.8),7,4.5,1.6,.6) # compression stop
  tray-=cyl((x,cy,z),1.1,12)
  tray-=cyl((x,cy,z-2.95),2.18,3.5,6) # 3.776 AF, top258.8; 1.2mm roof
 # Two short-end fences. Header banks on X sides remain open.
 for s in [-1,1]:tray+=plate((x,y+s*8.37,z+.7),7,1.1,1.4,.35)
 carrier=sol('Rex_Camera_Pi_Carrier')+arms+tray
 pcb=box((x,y,z+.8),(15.24,15.24,1.6))
 # Diagram specifies pitch/row spacing, not drill diameter; 1mm assumed.
 for xx in [x-6.35,x+6.35]:
  for i in range(6):pcb-=cyl((xx,y-6.35+i*2.54,z+.8),.5,3)
 cap=box((x,y,z+2.2),(11.6,13,1.2))
 parts={'Rex_Camera_Pi_Carrier':carrier,'JY61P_HEAD_PCB_ENVELOPE':pcb,'JY61P_HEAD_CAP_ENVELOPE':cap}
 props={}
 for s in [-1,1]:
  cy=y+s*10.5
  n='JY61P_HEAD_EDGE_CLIP_'+str(s)
  clip=plate((x,y+s*9.65,z+2.2),5.5,5.5,1.2,.65)-cyl((x,cy,z+2.2),1.1,4)
  parts[n]=clip
  # Nominal M2x6 machine screw with separate open-bottom nut pocket.
  parts['JY61P_HEAD_M2x6_'+str(s)]=cyl((x,cy,z-.2),1,6)+cyl((x,cy,z+3.4),1.9,1.2)
  parts['JY61P_HEAD_M2_NUT_'+str(s)]=cyl((x,cy,z-2),3.5/np.sqrt(3),1.6,6)-cyl((x,cy,z-2),1,3)
 # Empty-only connector corridors: tolerance allowances, NOT selected connectors.
 reserves={}
 for s in [-1,1]:reserves['RESERVE_JY61P_HEAD_HEADER_'+str(s)]=dict(center_mm=[x+s*12,y,z+3.6],size_mm=[10,16,4],role='NONPHYSICAL target for bent header plus mating plug above PCB; exact SKU and solder side not verified')
 reserves['RESERVE_JY61P_HEAD_CABLE_TO_PI']=dict(center_mm=[26,-12,252],size_mm=[8,5,7],role='NONPHYSICAL slack/strain relief planning only; actual harness not installed')
 return parts,reserves
if __name__=='__main__':
 parts,reserves=build();arrays={}
 for n,s in parts.items():
  assert str(s.status()).endswith('NoError'),(n,s.status())
  assert len(s.decompose())==1,(n,'disconnected',len(s.decompose()))
  m=s.to_mesh();arrays[n+'__v']=m.vert_properties[:,:3];arrays[n+'__f']=m.tri_verts
  print(n,round(s.volume(),3),'mm3',bb(s).round(3).tolist())
 np.savez_compressed(P/'delta_meshes.npz',**arrays)
 d=dict(source_sha256=B['source_sha256'],parent='BODY_jaw_soft',replaced=['Rex_Camera_Pi_Carrier'],added=[n for n in parts if n!='Rex_Camera_Pi_Carrier'],reserves=reserves,nominal_board_mm=[15.24,15.24,2.8],board_bottom_center_mm=[35,0,260],pin_pitch_mm=2.54,pin_row_spacing_mm=12.7,pcb_thickness_assumed_mm=1.6,header_hole_diameter_assumed_mm=1,shield_envelope_assumed_mm=[11.6,13,1.2],actual_enhanced_revision_fit_verified=False,drawing_url='https://cdn.shopify.com/s/files/1/0673/6848/5000/files/WitMotion-JY61P-Drawing.pdf?v=1782304757',parts={n:dict(volume_mm3=s.volume(),bounds_mm=bb(s).tolist()) for n,s in parts.items()})
 (P/'delta.json').write_text(json.dumps(d,indent=2))
