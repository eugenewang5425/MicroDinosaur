"""Verify the latest-only checkout without importing historical workflows.

Run with ordinary Python for file/data checks, or through Blender with the
current model open for additional live scene checks. Never saves any file.
"""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / 'current'
EVIDENCE = ROOT / 'work_in_progress/head_imu_v07'
MODEL = CURRENT / 'MicroDinosaur_v1.blender'
EXPECTED_SHA = '0a4f86aefea24e0d5260c837849a16164ec83cc2d364f58e6c03947de95ce286'
FLAGS = (
    'manufacturing_release', 'print_release', 'strength_release',
    'electrical_release', 'hardware_enabled', 'hardware_enable',
    'full_motion_range_release', 'continuous_collision_release',
    'gait_release', 'valid_com',
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def verify_files():
    require(MODEL.is_file(), 'Current model missing. Run git lfs pull.')
    with MODEL.open('rb') as stream:
        require(not stream.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'),
                'Only the Git LFS pointer is present. Run git lfs pull.')
    require(sha(MODEL) == EXPECTED_SHA, 'Current model SHA256 mismatch.')
    inventory = read(ROOT / 'repository_files.json')
    names = set()
    for row in inventory['files']:
        path = (ROOT / row['path']).resolve()
        require(path.is_relative_to(ROOT), 'Inventory path outside repository.')
        require(row['path'] not in names, 'Duplicate inventory path.')
        names.add(row['path'])
        require(path.is_file(), f'Missing upload file: {row["path"]}')
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                f'Upload file bytes/hash mismatch: {row["path"]}')
    manifest = read(CURRENT / 'model_manifest.json')
    status = read(CURRENT / 'delivery_status.json')
    mass = read(CURRENT / 'mass_estimate.json')
    publication = read(EVIDENCE / 'publication.json')
    require(manifest['sha256'] == status['models'][0]['sha256'] == publication['sha256'] == EXPECTED_SHA,
            'Model, status and publication hashes disagree.')
    require(manifest['object_count'] == len(manifest['objects']) == 755, 'Object count mismatch.')
    require(manifest['mesh_datablocks'] == 663 and manifest['physical_mass_rows'] == 672,
            'Mesh/mass count mismatch.')
    objects = {row['name']: row for row in manifest['objects']}
    require(len(objects) == 755, 'Duplicate object names in manifest.')
    require(manifest['actuator_count'] == manifest['control_count'] == 19, 'Actuator/control count mismatch.')
    require(manifest['modeled_imu_count'] == manifest['purchased_imu_count'] == 2
            and manifest['second_imu_installed'], 'IMU nominal CAD count mismatch.')
    require(not manifest['hidden_mesh_count'] and not manifest['orphan_mesh_count'], 'Hidden/orphan manifest mismatch.')
    rows = {row['name']: row for row in mass['rows']}
    require(len(rows) == len(mass['rows']) == 672, 'Mass rows duplicated or missing.')
    total = sum(row['mass_estimate_g'] for row in rows.values())
    for value in (mass['mass_estimate_g'], manifest['mass_estimate_g'], status['mass_estimate_g']):
        require(abs(total - value) < 1e-6, 'Mass total mismatch.')
    for name, row in rows.items():
        require(name in objects and abs(row['mass_estimate_g'] - objects[name]['mass_estimate_g']) < 1e-5,
                f'Mass/object mismatch: {name}')
    csv_mass = csv_rows(CURRENT / 'parts_mass_estimate.csv')
    require(len(csv_mass) == 672 and {r['name'] for r in csv_mass} == set(rows), 'CSV mass rows mismatch.')
    for row in csv_mass:
        require(abs(float(row['mass_estimate_g']) - rows[row['name']]['mass_estimate_g']) < 1e-6,
                f'CSV mass value mismatch: {row["name"]}')
    require(len(csv_rows(CURRENT / 'fastener_interface_schedule.csv')) == 222, 'Fastener row count mismatch.')
    for flag in FLAGS:
        require(manifest[flag] is False and status[flag] is False, f'Release flag changed: {flag}')
    bus = read(CURRENT / 's288_joint_bus_map.json')
    require(len(bus) == 19 and all(row['current_axis_source_model_sha256'] == EXPECTED_SHA for row in bus),
            'Bus map model binding mismatch.')
    imu = read(CURRENT / 'imu_mounts.json')
    require(imu['model_sha256'] == EXPECTED_SHA and not imu['physical_installation_verified'], 'IMU binding mismatch.')
    require(read(CURRENT / 'battery_design_target.json')['current_model_sha256'] == EXPECTED_SHA,
            'Battery model binding mismatch.')
    previews = read(CURRENT / 'previews/manifest.json')
    require(previews['model_sha256'] == EXPECTED_SHA and len(previews['screenshots']) == 6, 'Preview binding mismatch.')
    for row in previews['screenshots']:
        require(sha(CURRENT / 'previews' / row['file']) == row['image_sha256'], f'Preview mismatch: {row["file"]}')
    for filename in ('reopen_checks.json', 'current_reopen_checks.json'):
        report = read(EVIDENCE / filename)
        require(report['sha256'] == EXPECTED_SHA and not report['failures']
                and report['pose_count'] == 1605 and report['max_pose_matrix_error_mm'] == 0,
                f'Stored reopen evidence mismatch: {filename}')
    delta_sha, mesh_sha = sha(EVIDENCE / 'delta.json'), sha(EVIDENCE / 'delta_meshes.npz')
    for kind in ('static', 'regression', 'legs', 'head', 'clearance'):
        report = read(EVIDENCE / f'{kind}_checks.json')
        require(report['delta_sha256'] == delta_sha and report['delta_meshes_sha256'] == mesh_sha,
                f'Delta evidence hash mismatch: {kind}')
        if kind == 'static':
            require(report['tests'] == 58 and not report['new_intersections'] and not report['tool_hits']
                    and all(not r['hits'] for r in report['reservations'] + report['nut_insertion_paths']),
                    'Stored static audit has failures.')
        elif kind == 'clearance':
            require(not report['board_removal_hits'], 'Stored removal audit has failures.')
        else:
            require(report['poses'] == {'regression': 214, 'legs': 738, 'head': 653}[kind]
                    and not report['new_or_worsened_intersections'], f'Stored motion audit has failures: {kind}')
    cleanup = read(EVIDENCE / 'scene_metadata_cleanup.json')
    require(cleanup['after_sha256'] == EXPECTED_SHA and cleanup['only_scene_metadata_changed'], 'Metadata evidence mismatch.')
    require({p.name for p in CURRENT.iterdir() if p.suffix in ('.blend', '.blender')} == {MODEL.name},
            'Unexpected second current model.')
    return manifest, imu, {'status': 'PASS', 'model_sha256': EXPECTED_SHA,
                          'upload_files_checked': len(names), 'objects_in_manifest': 755,
                          'meshes_in_manifest': 663, 'mass_rows': 672, 'fastener_rows': 222,
                          'preview_hashes_checked': 6, 'mass_estimate_g': total,
                          'stored_pose_evidence_checked': 1605, 'historical_poses_replayed_this_run': False}


def verify_blender(bpy, manifest, imu):
    import bmesh
    import numpy as np

    require(Path(bpy.data.filepath).resolve() == MODEL.resolve(), 'Open current/MicroDinosaur_v1.blender for live checks.')
    for obj in bpy.data.objects:
        if 'angle_deg' in obj and 'reference_deg' in obj and obj.name.startswith(('CTRL_', 'DCTL_')):
            obj['angle_deg'] = obj['reference_deg']
            obj.update_tag()
    bpy.context.view_layer.update()
    expected = {row['name']: row for row in manifest['objects']}
    require(set(bpy.data.objects.keys()) == set(expected), 'Live object set differs from manifest.')
    require(len(bpy.data.meshes) == 663, 'Live mesh count differs.')
    for obj in bpy.data.objects:
        row = expected[obj.name]
        require(obj.type == row['type'] and (obj.parent.name if obj.parent else None) == row['parent'],
                f'Live type/parent mismatch: {obj.name}')
        require(obj.get('part_status', 'UNSET') == row['status'], f'Live status mismatch: {obj.name}')
        require(abs(float(obj.get('mass_estimate_g', 0)) - row['mass_estimate_g']) < 1e-5,
                f'Live mass mismatch: {obj.name}')
    hidden = [o.name for o in bpy.data.objects if o.type == 'MESH' and (o.hide_get() or o.hide_render or o.hide_viewport)]
    require(not hidden and not [m.name for m in bpy.data.meshes if not m.users], 'Hidden or orphan meshes detected.')
    drivers = [(o.name, f.driver.expression, f.driver.is_valid) for o in bpy.data.objects
               if o.animation_data for f in o.animation_data.drivers]
    saved = read(EVIDENCE / 'current_reopen_checks.json')['drivers']
    require(len(drivers) == 19 and all(d[2] for d in drivers), 'Invalid live drivers.')
    require({(d[0], d[1]) for d in drivers} == {(d['owner'], d['expression']) for d in saved}, 'Live driver expressions changed.')
    delta = read(EVIDENCE / 'delta.json')
    with np.load(EVIDENCE / 'delta_meshes.npz', allow_pickle=False) as arrays:
        for name in delta['parts']:
            obj = bpy.data.objects[name]
            vertices = np.array([obj.matrix_world @ vertex.co for vertex in obj.data.vertices]) * 1000
            require(vertices.shape == arrays[name + '__v'].shape
                    and float(np.max(np.abs(vertices - arrays[name + '__v']))) < .001,
                    f'Current v07 geometry differs from delta: {name}')
            mesh = bmesh.new()
            try:
                mesh.from_mesh(obj.data)
                require(all(edge.is_manifold for edge in mesh.edges), f'Nonmanifold v07 mesh: {name}')
                remaining, components = set(mesh.verts), 0
                while remaining:
                    components += 1
                    stack = [remaining.pop()]
                    while stack:
                        vertex = stack.pop()
                        for edge in vertex.link_edges:
                            other = edge.other_vert(vertex)
                            if other in remaining:
                                remaining.remove(other)
                                stack.append(other)
                require(components == 1, f'Disconnected v07 mesh: {name}')
            finally:
                mesh.free()
    frame = bpy.data.objects[imu['head']['frame']]
    require(np.max(np.abs(np.array(frame.matrix_world) - np.array(imu['head']['T_world_sensor_reference_m']))) < 2e-6,
            'Head IMU frame differs from nominal extrinsics.')
    for flag in FLAGS:
        require(bpy.context.scene.get(flag) is False or bpy.context.scene.get(flag) == 0,
                f'Live release flag differs: {flag}')
    missing = []
    for library in bpy.data.libraries:
        if not Path(bpy.path.abspath(library.filepath)).is_file():
            missing.append(library.filepath)
    for item in list(bpy.data.images) + list(bpy.data.fonts):
        if not item.users or getattr(item, 'packed_file', None) or getattr(item, 'packed_files', None):
            continue
        if getattr(item, 'source', '') in ('GENERATED', 'VIEWER'):
            continue
        path = getattr(item, 'filepath', '')
        if path and path != '<builtin>' and not Path(bpy.path.abspath(path, library=item.library)).is_file():
            missing.append(path)
    require(not missing, f'Missing external Blender resources: {missing}')
    require(sha(MODEL) == EXPECTED_SHA, 'Model file changed during read-only verification.')
    return {'live_scene': 'PASS', 'blender_version': bpy.app.version_string, 'objects': 755,
            'meshes': 663, 'valid_drivers': 19, 'closed_connected_delta_meshes': len(delta['parts']),
            'missing_external_resources': missing, 'model_saved': False}


if __name__ == '__main__':
    manifest, imu, report = verify_files()
    try:
        import bpy
    except ImportError:
        report['live_scene'] = 'NOT_RUN: use Blender for live scene checks'
    else:
        report.update(verify_blender(bpy, manifest, imu))
    print(json.dumps(report, ensure_ascii=False, indent=2))
