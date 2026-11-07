"""Verified native input transport for one pinned MiDaS/RKNN profile.

The SDK's effective fused input conversion differs from a naive public-scale
division. Keep this profile isolated, require binary integrity, and compare
real SDK outputs at every model-context initialization before enabling it.
"""
import ctypes
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

MODEL = 'midas-optimized-int8'
MODEL_SHA256 = '8a22b419775cae587b5478df04f01ccc7eaf9f4a1585b705fc98781841dd0ac1'
ARTIFACTS = Path(__file__).resolve().parent / 'input_transport_20261029'


@lru_cache(maxsize=1)
def _quantizer():
    manifest = json.loads((ARTIFACTS / 'exact_input_build.json').read_text(encoding='utf-8'))
    library = ARTIFACTS / 'exact_quantize.so'
    source = ARTIFACTS / 'exact_quantize.c'
    if (manifest.get('model_sha256') != MODEL_SHA256
            or manifest.get('abi') != 1
            or hashlib.sha256(library.read_bytes()).hexdigest() != manifest.get('library_sha256')
            or hashlib.sha256(source.read_bytes()).hexdigest() != manifest.get('source_sha256')):
        raise RuntimeError('Native input profile/build integrity mismatch; use SDK input transport')
    loaded = ctypes.CDLL(str(library))
    method = loaded.exact_quantize
    method.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    method.restype = ctypes.c_int
    return method


def quantize_tensor(tensor):
    if (not isinstance(tensor, np.ndarray) or tensor.dtype != np.float32
            or tensor.shape != (1, 256, 256, 3) or not tensor.flags.c_contiguous):
        raise ValueError('Native input requires an owned contiguous NHWC float32 256 tensor')
    output = np.empty(tensor.shape, dtype=np.int8)
    status = _quantizer()(tensor.ctypes.data, output.ctypes.data, tensor.size)
    if status:
        raise RuntimeError('Native quantizer rejected the input/rounding environment')
    return output


def boundary_tensor():
    centers = ((np.arange(-140, 141, dtype=np.float64) + .5 + 112) / 235.34375).astype(np.float32)
    values = [centers.copy()]
    low, high = centers.copy(), centers.copy()
    for _ in range(8):
        low = np.nextafter(low, np.float32(-np.inf))
        high = np.nextafter(high, np.float32(np.inf))
        values.extend([low.copy(), high.copy()])
    values = np.concatenate(values + [np.array([-100, -1, 0, 1, 100], np.float32)])
    return np.resize(values, (1, 256, 256, 3)).astype(np.float32)


def verify_runtime(runtime):
    """Fail before model activation if the installed SDK no longer matches."""
    tensors = [boundary_tensor(), np.random.default_rng(7).random((1, 256, 256, 3), dtype=np.float32)]
    for tensor in tensors:
        normal = runtime.inference(inputs=[tensor], data_format=['nhwc'])
        if not normal or not np.isfinite(normal[0]).all():
            raise RuntimeError('SDK native-input compatibility reference failed')
        expected = np.asarray(normal[0]).copy()
        result = runtime.inference(inputs=[quantize_tensor(tensor)], data_format=['nhwc'], inputs_pass_through=[1])
        if (not result or expected.shape != result[0].shape or expected.dtype != result[0].dtype
                or expected.tobytes() != result[0].tobytes()):
            raise RuntimeError('Native input differs from installed SDK; use SDK input transport')
    return {'passed': True, 'synthetic_tensors': len(tensors), 'native_depth_bit_exact': True}
