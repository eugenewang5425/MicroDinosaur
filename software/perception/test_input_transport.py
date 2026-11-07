"""Input transport contracts independent of ARM hardware or the camera."""
from contextlib import nullcontext
import types
import unittest
from unittest.mock import patch

import numpy as np

from camera_live_server import ModelContext
import exact_input


class Backend:
    def __init__(self):
        self.calls = []

    def inference(self, **kwargs):
        self.calls.append(kwargs)
        return [np.ones((1, 256, 256), dtype=np.float32)]

    def set_inputs(self, inputs, dtype, fmt, *, inputs_pass_through):
        self.calls.append({'inputs': inputs, 'data_format': fmt,
                           'inputs_pass_through': inputs_pass_through})

    def run(self, frame_id):
        self.calls.append('run')

    def get_outputs(self, frame_id):
        return [np.ones((1, 256, 256), dtype=np.float32)]


class InputTransportTests(unittest.TestCase):
    def context(self, native, locked=False):
        value = ModelContext.__new__(ModelContext)
        value.native_input = native
        value.execution_lock = nullcontext() if locked else None
        backend = Backend()
        backend.rknn_runtime = backend
        value.runtime = backend
        return value, backend

    def test_sdk_path_keeps_ordinary_float_input(self):
        context, backend = self.context(False)
        tensor = np.zeros((1, 256, 256, 3), np.float32)
        depth, _ = context.infer_tensor(tensor)
        self.assertIs(backend.calls[0]['inputs'][0], tensor)
        self.assertNotIn('inputs_pass_through', backend.calls[0])
        self.assertEqual(depth.shape, (256, 256))

    def test_native_path_bypasses_sdk_conversion(self):
        context, backend = self.context(True)
        tensor = np.zeros((1, 256, 256, 3), np.int8)
        context.infer_tensor(tensor)
        self.assertIs(backend.calls[0]['inputs'][0], tensor)
        self.assertEqual(backend.calls[0]['inputs_pass_through'], [1])

    def test_native_lock_path_keeps_passthrough(self):
        context, backend = self.context(True, locked=True)
        context.infer_tensor(np.zeros((1, 256, 256, 3), np.int8))
        self.assertEqual(backend.calls[0]['inputs_pass_through'], [1])
        self.assertEqual(backend.calls[1], 'run')

    def test_bad_native_tensor_cannot_reach_runtime(self):
        for tensor in [np.zeros((1, 256, 256, 3), np.float32),
                       np.zeros((1, 224, 224, 3), np.int8),
                       np.zeros((1, 256, 256, 3), np.int8)[:, :, ::-1]]:
            context, backend = self.context(True)
            with self.assertRaises(ValueError):
                context.infer_tensor(tensor)
            self.assertFalse(backend.calls)

    def test_quantizer_rejects_shape_dtype_and_strides_before_loading_native_library(self):
        with patch.object(exact_input, '_quantizer') as library:
            for tensor in [np.zeros((1, 256, 256, 3), np.float64),
                           np.zeros((1, 224, 224, 3), np.float32),
                           np.zeros((1, 256, 256, 3), np.float32)[:, :, ::-1]]:
                with self.assertRaises(ValueError):
                    exact_input.quantize_tensor(tensor)
            library.assert_not_called()

    def test_runtime_selftest_rejects_changed_native_outputs(self):
        class ChangedRuntime:
            def inference(self, **kwargs):
                return [np.full((1, 256, 256), int('inputs_pass_through' in kwargs), np.float32)]
        with patch.object(exact_input, 'quantize_tensor', return_value=np.zeros((1, 256, 256, 3), np.int8)):
            with self.assertRaisesRegex(RuntimeError, 'differs'):
                exact_input.verify_runtime(ChangedRuntime())

    def test_initialization_failure_releases_context(self):
        backend = types.SimpleNamespace(load_rknn=lambda _: 0, init_runtime=lambda: 0,
                                        release=unittest.mock.Mock())
        api = types.ModuleType('rknnlite.api')
        api.RKNNLite = lambda **kwargs: backend
        model = exact_input.MODEL
        import tempfile
        from pathlib import Path
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.rknn'
            path.write_bytes(b'isolated-model-test')
            model_row = {'artifact': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            with patch.dict('sys.modules', {'rknnlite.api': api}), \
                    patch.dict('camera_live_server.MODELS', {model: model_row}), \
                    patch.object(exact_input, 'MODEL_SHA256', model_row['sha256']), \
                    patch.object(exact_input, 'verify_runtime', side_effect=RuntimeError('incompatible SDK')):
                with self.assertRaisesRegex(RuntimeError, 'incompatible SDK'):
                    ModelContext(model, native_input=True)
        backend.release.assert_called_once()

    def test_new_model_hash_cannot_reuse_native_profile(self):
        import tempfile
        from pathlib import Path
        import hashlib
        api = types.ModuleType('rknnlite.api')
        api.RKNNLite = unittest.mock.Mock()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'new-model.rknn'
            path.write_bytes(b'new-model-without-native-profile')
            model_row = {'artifact': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            with patch.dict('sys.modules', {'rknnlite.api': api}), \
                    patch.dict('camera_live_server.MODELS', {exact_input.MODEL: model_row}):
                with self.assertRaisesRegex(RuntimeError, 'profile does not match'):
                    ModelContext(exact_input.MODEL, native_input=True)
        api.RKNNLite.assert_not_called()


if __name__ == '__main__':
    unittest.main()
