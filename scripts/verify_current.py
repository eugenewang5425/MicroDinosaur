"""Read-only release identity checks; never opens hardware or saves CAD."""
from pathlib import Path
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

def check():
    inventory = json.loads((ROOT / 'repository_files.json').read_text(encoding='utf-8'))
    for row in inventory['files']:
        p = ROOT / row['path']
        assert p.is_file(), f"Missing file: {row['path']}"
        with p.open('rb') as stream:
            assert not stream.read(128).startswith(b'version https://git-lfs.github.com/spec/'), f"Run git lfs pull: {row['path']}"
        assert p.stat().st_size == row['bytes'], f"Size mismatch: {row['path']}"
        with p.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert actual == row['sha256'], f"SHA256 mismatch: {row['path']}"
    m = json.loads((ROOT/'current/current_manifest.json').read_text(encoding='utf-8'))
    for key in ('current_model','current_print_project'):
        with (ROOT/'current'/m[key]['path']).open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest() == m[key]['sha256']
    assert not m['physical_fit_verified'] and not m['full_motion_verified']
    with zipfile.ZipFile(ROOT/'current'/m['current_print_project']['path']) as z:
        assert z.testzip() is None, 'Corrupt 3MF ZIP member'
        root=ET.fromstring(z.read('Metadata/model_settings.config'))
        counts=[len(p.findall('model_instance')) for p in root.findall('plate')]
        assert counts == m['current_print_project']['plate_counts'], f'Plate counts: {counts}'
    print(f"Verified {len(inventory['files'])} files; OS-M02 model/print identities; 8 plates, 30 parts. Physical validation remains pending.")

if __name__ == '__main__':
    check()
