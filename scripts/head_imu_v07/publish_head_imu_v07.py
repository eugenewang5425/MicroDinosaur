"""Publish one verified candidate; preserve local rollback and screenshot history."""
import bpy,json,hashlib,shutil,csv
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'work_in_progress/head_imu_v07';C=R/'current';current=C/'MicroDinosaur_v1.blender';candidate=P/current.name
if not (P/'basis.json').is_file():
 raise SystemExit('Publication requires the excluded original local baseline and archive. See scripts/head_imu_v07/README.md; current v07 is already published.')
archive=R/'exports/retired_before_head_imu_v07_20260913';shots=R/'versions/screenshots/20260913_07_头部JY61P标准尺寸安装'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def rel(p):return Path(p).resolve().relative_to(R.resolve()).as_posix()
B=read(P/'basis.json');D=read(P/'delta.json');V=read(P/'reopen_checks.json');mass=read(P/'mass_estimate.json');H=sha(candidate)
assert Path(bpy.data.filepath).resolve()==candidate.resolve()
assert sha(current)==B['source_sha256'];assert H==V['sha256'] and not V['failures']
assert len(bpy.data.objects)==755 and len(bpy.data.meshes)==663 and len(mass['rows'])==672
assert sha(R/'exports/Microduck_assembly.blend')==V['source_blueprint_sha256']
checks={n:read(P/(n+'_checks.json')) for n in ['static','regression','legs','head','clearance']}
for q in checks.values():
 assert q['delta_sha256']==sha(P/'delta.json') and q['delta_meshes_sha256']==sha(P/'delta_meshes.npz')
assert not checks['static']['new_intersections'] and not checks['static']['tool_hits']
assert all(not r['hits'] for r in checks['static']['reservations']+checks['static']['nut_insertion_paths'])
assert all(not checks[n]['new_or_worsened_intersections'] for n in ['regression','legs','head'])
assert not checks['clearance']['board_removal_hits']
assert not archive.exists() and not shots.exists(),'Refuse repeated publication / overwriting history'
history={str(p.relative_to(R)):sha(p) for p in (R/'versions/screenshots').rglob('*.png')};assert len(history)==100
archive.mkdir();shots.mkdir();(archive/'previews').mkdir()
# Copy exact frozen source files before replacing only the current named entry.
shutil.copy2(P/'BEFORE_HEAD_IMU_RETIRED_INPUT.blender',archive/'MicroDinosaur_v1_v06_RETIRED.blender')
shutil.copy2(P/'BEFORE_HEAD_LOCAL_NOT_CURRENT.blend',archive/'HEAD_LOCAL_v06_NOT_CURRENT.blend')
for p in P.glob('source_*'):
 if p.is_file():shutil.copy2(p,archive/p.name.removeprefix('source_'))
oldpreviews=[]
for p in list((C/'previews').iterdir()):
 if not p.is_file():raise AssertionError('Unexpected nested previews')
 dest=archive/'previews'/p.name
 assert p.resolve().parent==(C/'previews').resolve() and dest.resolve().is_relative_to(archive.resolve())
 oldpreviews.append(dict(file=p.name,sha256=sha(p)));shutil.move(str(p),str(dest))
labels={'head_imu_open':'头内位置_去顶壳','head_imu_mount':'IMU固定座_局部','head_imu_top':'板边夹_俯视','head_imu_bottom':'底部螺母槽','head_imu_closed':'头壳外形不变','assembly_three_quarter':'整机斜视'};screens=[]
for name,label in labels.items():
 src=P/(name+'.png');dest=shots/('v07_'+label+'.png');shutil.copy2(src,dest);shutil.copy2(src,C/'previews'/src.name)
 screens.append(dict(file=rel(dest),label=label,source=rel(src),image_sha256=sha(dest),model_sha256=H,status='ACTUAL_CAD_RENDER',scope_note='Isolation only for render; no hidden retired parts in saved model'))
for srcname,label in [('before_head_open','v06_修改前_头部外壳'),('before_carrier','v06_修改前_相机Pi载架')]:
 src=P/(srcname+'.png');dest=shots/(label+'.png');shutil.copy2(src,dest)
 screens.append(dict(file=rel(dest),label=label,source=rel(src),image_sha256=sha(dest),model_sha256=B['source_sha256'],status='PRIOR_CAD_RENDER_NOT_CURRENT'))
write(shots/'manifest.json',dict(version='20260913_07',model_sha256=H,model_objects=755,screenshots=screens))
for p,h in history.items():assert sha(R/p)==h
shutil.copy2(candidate,current);assert sha(current)==H
shutil.copy2(P/'mass_estimate.json',C/'mass_estimate.json');shutil.copy2(P/'parts_mass_estimate.csv',C/'parts_mass_estimate.csv')
manifest=read(P/'source_model_manifest.json');manifest.update(file=rel(current),sha256=H,variant='MicroDinosaur_v1_HEAD_AND_BODY_JY61P_NOMINAL_MECHANICAL_REVIEW',object_count=755,mesh_datablocks=663,physical_mass_rows=672,review_document='HEAD_IMU_V07_REVIEW.md',source_sha256=B['source_sha256'],removed_this_revision=[],added_this_revision=D['added']+list(D['reserves'])+['FRAME_JY61P_HEAD_SENSOR'],replaced_meshes_this_revision=D['replaced'],mass_estimate_g=mass['mass_estimate_g'],recovery_copy=rel(archive/'MicroDinosaur_v1_v06_RETIRED.blender'),local_parts_backup=rel(archive/'HEAD_LOCAL_v06_NOT_CURRENT.blend'),evidence_directory=rel(P),screenshots_version='20260913_07',modeled_imu_count=2,second_imu_installed=True,second_imu_installed_means='CAD nominal envelope only; not physical hardware installation',head_imu='JY61P_HEAD_PCB_ENVELOPE + JY61P_HEAD_CAP_ENVELOPE',head_imu_frame='FRAME_JY61P_HEAD_SENSOR',imu_mounts_file='imu_mounts.json',head_header_and_sensor_axes_verified=False,obsolete_geometry_retired_by_complete_rebuild=False,prior_v06_structural_rebuild_preserved=True)
manifest['objects']=[dict(name=o.name,type=o.type,parent=o.parent.name if o.parent else None,status=o.get('part_status','UNSET'),print_part=bool(o.get('print_part',False)),collections=[c.name for c in o.users_collection],material_design=o.get('material_design',''),mass_estimate_g=float(o.get('mass_estimate_g',0)),mass_status=o.get('mass_status','')) for o in bpy.data.objects]
write(C/'model_manifest.json',manifest)
state=read(P/'source_delivery_status.json');state['previous_revision_delta_audit']=state.pop('latest_delta_audit');state.update(status='CURRENT v07: nominal second JY61P in rigid head carrier; not hardware/printing/electrical/continuous-motion release',models=[dict(file=rel(current),sha256=H,objects=755,actuators=19,drivers=19)],review_document='HEAD_IMU_V07_REVIEW.md',mass_estimate_g=mass['mass_estimate_g'],com_forward_of_foot_center_mm=mass['com_fore_aft_offset_mm'],screenshot_version='20260913_07',modeled_imu_count=2)
state['pending']+=['Head IMU follows standard JY61P drawing, not measured enhanced shield/header revision; verify physical board, bent header side, pinout and solder clearance before printing or connecting.', 'Head screw-to-skull nominal minimum0.616mm; verify actual screw head/printing tolerance, no production clearance release.', 'Head/body IMU role, axes, latency and timestamp calibration pending; do not feed head motion as body balance data. No new electrical firmware implemented.']
state['latest_delta_audit']=dict(replaced_meshes=1,added_physical_objects=8,added_nonphysical_objects=4,unchanged_objects=742,static_exact_checks=checks['static']['tests'],regression_poses=214,leg_poses=738,head_poses=653,actual_pose_matrix_checks=1605,max_pose_matrix_error_mm=V['max_pose_matrix_error_mm'],new_or_worsened_intersections=0,head_nut_paths=2,head_nut_samples_per_path=17,board_removal_samples=41,nominal_screw_to_skull_min_mm=min(r['gap_mm'] for r in checks['clearance']['gaps']),evidence_directory=rel(P))
write(C/'delivery_status.json',state)
battery=read(P/'source_battery_design_target.json');battery.update(current_model_sha256=H,geometry_revision_work_directory='work_in_progress/front_mount_v06',current_review_document='HEAD_IMU_V07_REVIEW.md',battery_geometry_and_pose_unchanged_from_v06=True);write(C/'battery_design_target.json',battery)
bus=read(P/'source_s288_joint_bus_map.json')
for r in bus:r['current_axis_source_model_sha256']=H
write(C/'s288_joint_bus_map.json',bus)
with (P/'source_fastener_interface_schedule.csv').open(encoding='utf-8-sig',newline='') as f:
 reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
before_count=len(rows)
for sign in [-1,1]:
 rows.append(dict(fastener='JY61P_HEAD_M2x6_'+str(sign),mount_part='JY61P_HEAD_EDGE_CLIP_'+str(sign),interface='M2_MACHINE_BOLT_SEPARATE_HEX_NUT',nominal_CAD_insertion_mm=1.6,oem_hole_depth_limit_mm='',actual_screw_SKU_pitch_tip_drive_verified=False,case_depth_verified=False,seat_fraction=1.,note='Through hole2.2mm; screw underhead length6mm; nutAF3.5 x1.6mm; bottom-open pocketAF3.776; nut roof contact; tip protrusion0.4mm. PCB edge clamp, NOT PCB drilled mounting hole. Envelope only; hardware and torque unverified.'))
with (C/'fastener_interface_schedule.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
write(C/'imu_mounts.json',dict(model_sha256=H,modeled_imu_count=2,purchased_imu_count=2,physical_installation_verified=False,firmware_or_synchronization_implemented=False,head=dict(model='JY61P enhanced nominal envelope based on standard drawing',parent='BODY_jaw_soft',carrier='Rex_Camera_Pi_Carrier',frame='FRAME_JY61P_HEAD_SENSOR',reference_board_center_mm=[35,0,260.8],T_parent_sensor_nominal_m=V['head_imu_frame_head_local_m'],T_world_sensor_reference_m=V['head_imu_reference_world_m'],frame_is_actual_sensor_die=False,axis_convention='Provisional PCB X=head forward Y=head left Z=up at reference. Real package axis mapping/offset must be calibrated.',role='Head rotation/camera rigid-frame reference; not body balance IMU',connection_plan='Local Pi I2C proposed; exact pins/voltage/address/latency NOT approved'),body=dict(model='JY61P nominal envelope unchanged',reference_board_center_mm=[-23.2,0,145.5],pose_rotated_z_from_v04_deg=180,role='Body attitude feedback to F411',calibrated_T_parent_sensor=None,connection_plan='F411 I2C proposed; not implemented or wiring approved'),warning='CAD extrinsics are not time synchronization or calibrated hardware extrinsics. Do not default-connect two same-address I2C modules on one bus.'))
write(C/'previews/manifest.json',dict(version='20260913_07',model_sha256=H,screenshots=[dict(file=name+'.png',label=label,image_sha256=sha(C/'previews'/(name+'.png'))) for name,label in labels.items()],old_previews_moved_to=rel(archive/'previews')))
write(archive/'retired_status.json',dict(status='RETIRED_v06_REFERENCE_NOT_CURRENT_NOT_ALTERNATIVE',model_sha256=B['source_sha256'],local_parts_library='HEAD_LOCAL_v06_NOT_CURRENT.blend',current_model=rel(current),current_sha256=H,old_previews=oldpreviews))
write(P/'publication.json',dict(current_model=rel(current),sha256=H,backup=rel(archive),screenshots=rel(shots),old_screenshots_hash_verified=len(history),new_screenshots=len(screens),all_screenshots=len(history)+len(screens),fastener_schedule_before_rows=before_count,fastener_schedule_rows=len(rows),source_blueprint_sha256=V['source_blueprint_sha256'],candidate_is_not_second_current=True))
print('PUBLISHED',H,'755 objects, 663 meshes, 672 mass rows; screenshots108',flush=True)
