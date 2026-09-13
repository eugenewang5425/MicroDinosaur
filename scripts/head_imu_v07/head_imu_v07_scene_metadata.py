"""Remove misleading active scene labels inherited from retired robot layouts."""
def clean_scene(bpy):
 s=bpy.context.scene
 obsolete=['reference_mass_kg','reference_stance_margin_mm','auxiliary_servo_count','v08_static_fit_closed','SC09_registration','v09_interfaces_closed','v09_motion_interfaces_corrected','v10_battery','v10_fallback','v11_open_service','v12_rear_hips','v13_status','nominal_shell_wall_mm','wire_corridor_diameter_mm','camera_corridor_diameter_mm','sampled_neck_module_min_gap_mm','obsolete_geometry_removed']
 removed={k:str(s[k]) for k in obsolete if k in s}
 for k in removed:del s[k]
 if 'baseline_source_sha256' in s:
  s['historical_initial_s288_source_sha256']=s['baseline_source_sha256'];del s['baseline_source_sha256']
 s['print_part_count']=sum(bool(o.get('print_part',False)) for o in bpy.data.objects)
 s['jaw_servo_status']='Unitree S288 nominal CAD assembly; hardware fit/strength/firmware not released'
 s['tail_actuator_status']='Two Unitree S288 in current symmetric tail mounts; hardware fit/thermal duty not released'
 s['tail_motion']='Current DCTL_Tail_Yaw/Pitch controls and per-joint limits; rigid nominal50g PETG tail; no passive springs; continuous range not approved'
 s['tail_connection_status']='Current S288 mounts and fixed interfaces unchanged from v06; use model_manifest and current design, not retired XL330/SC09 guide'
 s['assembly_guide']='HEAD_IMU_V07_REVIEW.md; CURRENT_DESIGN.md'
 s['documentation']='HEAD_IMU_V07_REVIEW.md'
 s['mass_estimate_status']='CAD estimates, unweighed and incomplete electrical BOM; current/mass_estimate.json is authoritative'
 s['battery_variant']='JMP_3S_11_1V_4000mAh_90x42x27_NOMINAL'
 s['battery_configuration']='User purchased JMP ordinary3S11.1V4000mAh90x42x27mm; catalogue230g not weighed; lead polarity/XT60/regen protection unverified'
 s['imu_configuration']='2 x purchased JY61P enhanced, nominal CAD modules: body on F411 carrier and head on rigid camera/Pi carrier. Standard15.24x15.24x2.8mm drawing; actual enhanced shield/header/axes and firmware pending'
 s['latest_request_status']='v07 adds head JY61P mount using online standard drawing; old742 objects unchanged; no resized skull; current layout and review in CURRENT_DESIGN.md'
 s['upper_neck_plate_decision']='No additional neck bumper. v06 lower module retained; v07 head IMU delta checked at discrete poses only'
 return removed

if __name__=='__main__':
 import bpy,json,hashlib,shutil
 from pathlib import Path
 R=Path(__file__).resolve().parents[2];P=R/'work_in_progress/head_imu_v07';f=R/'current/MicroDinosaur_v1.blender'
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 before=sha(f);assert before=='9e64649ee3db3e8390203291e22c3b40d51cc4230c5ec0a7837cefd9673c64e6'
 assert Path(bpy.data.filepath).resolve()==f.resolve()
 backup=P/'GEOMETRY_ONLY_BEFORE_SCENE_METADATA_CLEANUP_NOT_CURRENT.blender';assert not backup.exists();shutil.copy2(f,backup)
 removed=clean_scene(bpy);bpy.context.preferences.filepaths.save_version=0;bpy.ops.wm.save_as_mainfile(filepath=str(f),check_existing=False);after=sha(f)
 shutil.copy2(f,P/'MicroDinosaur_v1.blender')
 # These are publication metadata, not collision evidence. Reopen evidence is regenerated independently.
 files=list((R/'current').glob('*.json'))+[R/'current/previews/manifest.json',R/'versions/screenshots/20260913_07_头部JY61P标准尺寸安装/manifest.json',P/'publication.json',R/'exports/retired_before_head_imu_v07_20260913/retired_status.json']
 def replace(v):
  if isinstance(v,str):return v.replace(before,after)
  if isinstance(v,list):return [replace(x) for x in v]
  if isinstance(v,dict):return {k:replace(x) for k,x in v.items()}
  return v
 for p in files:p.write_text(json.dumps(replace(json.loads(p.read_text(encoding='utf-8-sig'))),ensure_ascii=False,indent=2),encoding='utf-8')
 q=dict(before_sha256=before,after_sha256=after,only_scene_metadata_changed=True,removed_scene_properties=removed,geometry_and_render_unchanged=True)
 (P/'scene_metadata_cleanup.json').write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8');print('SCENE_METADATA_CLEANED',after,flush=True)
