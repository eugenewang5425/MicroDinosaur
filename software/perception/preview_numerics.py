"""Exact CPU alternatives to the frozen preview numerical functions.

Scratch allocations belong to the calling thread. Returned model inputs own
their storage independently of scratch, so a preparation worker may process the
next frame while an inference worker still consumes the previous input.
"""
import threading
import time

import cv2
import numpy as np


# Preserve float32 division *before* cubic interpolation. Multiplication by a
# rounded reciprocal, or normalization after resize, can change INT8 inference.
_INPUT_LUT = np.arange(256, dtype=np.float32) / np.float32(255)
_INPUT_LUT.setflags(write=False)
_SCRATCH = threading.local()


def _buffer(name, shape, dtype):
    buffers = getattr(_SCRATCH, 'buffers', None)
    if buffers is None:
        buffers = _SCRATCH.buffers = {}
    dtype = np.dtype(dtype)
    value = buffers.get(name)
    if value is None or value.shape != shape or value.dtype != dtype:
        value = buffers[name] = np.empty(shape, dtype=dtype)
    return value


def prepare_model_input(image, model):
    """Return an owned input tensor with the frozen channel/rounding order."""
    rgb = _buffer('rgb', image.shape, np.uint8)
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=rgb)
    if model.startswith('fast-depth'):
        return cv2.resize(rgb, (224, 224))[None].copy()
    normalized = _buffer('normalized', image.shape, np.float32)
    cv2.LUT(rgb, _INPUT_LUT, dst=normalized)
    # Allocate the final tensor: returning a scratch view races queued inference.
    resized = cv2.resize(normalized, (256, 256), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(resized[None], dtype=np.float32)


def _depth_fraction(depth, low, high):
    denominator = max(high - low, 1e-6)
    subtraction_dtype = np.result_type(depth, low)
    division_dtype = np.result_type(np.empty((1,), dtype=subtraction_dtype), denominator)
    if subtraction_dtype != division_dtype:
        # Preserve intermediate rounding if unusual limits change promotion
        # between subtraction and division (NumPy promotion varies by version).
        return np.clip((depth - low) / denominator, 0, 1)
    fraction = _buffer('fraction', depth.shape, subtraction_dtype)
    np.subtract(depth, low, out=fraction)
    np.divide(fraction, denominator, out=fraction)
    np.clip(fraction, 0, 1, out=fraction)
    return fraction


def encode_depth(image, depth, model, limits):
    """Produce identical TURBO/linear/JPEG82 bytes with thread-local scratch."""
    started = time.perf_counter()
    fraction = _depth_fraction(depth, *limits)
    if not model.startswith('midas'):
        np.subtract(1, fraction, out=fraction)
    np.multiply(fraction, 255, out=fraction)
    indices = _buffer('indices', depth.shape, np.uint8)
    np.copyto(indices, fraction, casting='unsafe')
    native_color = _buffer('native_color', (*depth.shape, 3), np.uint8)
    cv2.applyColorMap(indices, cv2.COLORMAP_TURBO, dst=native_color)
    height, width = image.shape[:2]
    pair = _buffer('pair', (height, width * 2, 3), np.uint8)
    pair[:, :width] = image
    cv2.resize(native_color, (width, height), dst=pair[:, width:])
    post_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    ok, encoded = cv2.imencode('.jpg', pair, [cv2.IMWRITE_JPEG_QUALITY, 82])
    jpeg_ms = (time.perf_counter() - started) * 1000
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return encoded.tobytes(), {'postprocess_ms': post_ms, 'jpeg_ms': jpeg_ms}
