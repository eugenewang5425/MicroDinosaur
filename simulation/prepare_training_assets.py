"""Verify and restore only the archived assets; never overwrite different files."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / 'training'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def main():
    manifest = json.loads((ROOT/'SOURCE_SNAPSHOT.json').read_text(encoding='utf-8'))
    expected = {x['path']: x for x in manifest['asset_members']}
    archive = ROOT/'training_assets.zip'
    archive_row = next(x for x in manifest['files'] if x['path']=='simulation/training_assets.zip')
    if digest(archive.read_bytes()) != archive_row['sha256']:
        raise ValueError('Asset archive hash mismatch. Run git lfs pull.')
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist()) != set(expected):
            raise ValueError('Asset archive inventory mismatch')
        # Validate all members and existing destinations before any extraction.
        for name, row in expected.items():
            dst = (TARGET/name).resolve()
            if not dst.is_relative_to(TARGET.resolve()):
                raise ValueError('Archive path escapes training folder')
            data = z.read(name)
            if len(data) != row['bytes'] or digest(data) != row['sha256']:
                raise ValueError(f'Asset hash mismatch: {name}')
            if dst.exists() and digest(dst.read_bytes()) != row['sha256']:
                raise FileExistsError(f'Preserving different existing asset: {dst}')
        for name in expected:
            dst = TARGET/name
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(z.read(name))
    print(json.dumps({'status':'PASS', 'assets_verified':len(expected)}))

if __name__ == '__main__':
    main()
