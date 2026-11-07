"""Build the ARM quantizer and pin source/binary/model identities."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess

HERE = Path(__file__).resolve().parent
MODEL_SHA = '8a22b419775cae587b5478df04f01ccc7eaf9f4a1585b705fc98781841dd0ac1'


def main():
    if platform.machine().lower() not in ('aarch64', 'arm64'):
        raise RuntimeError('Build this quantizer on the ARM64 deployment machine')
    model = HERE.parent / 'midas_optimization/midas-deconv256-int8.rknn'
    if hashlib.sha256(model.read_bytes()).hexdigest() != MODEL_SHA:
        raise RuntimeError('Unsupported model identity')
    source = HERE / 'exact_quantize.c'
    library = HERE / 'exact_quantize.so'
    temporary = HERE / 'exact_quantize.so.building'
    command = ['gcc', '-O3', '-fPIC', '-shared', '-Wall', '-Wextra', str(source), '-lm', '-o', str(temporary)]
    subprocess.run(command, check=True, timeout=30)
    temporary.replace(library)
    manifest = {'abi': 1, 'model_sha256': MODEL_SHA,
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
                'compiler': subprocess.run(['gcc', '--version'], capture_output=True, text=True, check=True).stdout.splitlines()[0],
                'command': command, 'profile': 'fused float32 multiply 235.34375 and offset -112, nearest-even saturation'}
    (HERE / 'exact_input_build.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
