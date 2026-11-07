"""Rebuild release inventory from explicitly staged/tracked files only."""
from pathlib import Path
import hashlib, json, subprocess

ROOT=Path(__file__).resolve().parents[1]

def main():
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf-8').split('\0')
    rows=[]
    for name in sorted(set(names)):
        p=ROOT/name
        if not name or name=='repository_files.json' or not p.is_file(): continue
        with p.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
        rows.append({'path':name,'bytes':p.stat().st_size,'sha256':digest})
    result={'schema_version':4,'repository':'eugenewang5425/MicroDinosaur','published_on':'2026-11-07',
            'scope':'Latest OS-M02 CAD/print, HAT candidate, host software, IMU probes and perception only',
            'model_path':'current/MicroDinosaur_Current.blend','inventory_excludes_itself':True,'files':rows}
    (ROOT/'repository_files.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(f"Inventory: {len(rows)} files, {sum(x['bytes'] for x in rows)/2**20:.2f} MiB")

if __name__=='__main__': main()
