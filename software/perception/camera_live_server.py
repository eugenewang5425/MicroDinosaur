"""Latest-frame CSI preview with asynchronous model switching; no actuator access."""
import argparse
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import socket
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit

from preview_runtime import MODELS as ALL_MODELS, PreviewRuntime

MODELS = {'midas-optimized-int8': ALL_MODELS['midas-optimized-int8']}
from preview_pipeline import LatestSlot, prepare_worker, inference_worker, encode_worker

HERE = Path(__file__).resolve().parent
MODEL_ROOT = Path('/home/radxa/board-acceleration-20261016')


def prepare_model_input(image, model):
    """The frozen model preprocessing contract evaluated in float32."""
    import cv2
    import numpy as np
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if model.startswith('fast-depth'):
        return cv2.resize(rgb, (224, 224))[None].copy()
    rgb = cv2.resize(rgb.astype(np.float32) / np.float32(255), (256, 256), interpolation=cv2.INTER_CUBIC)
    # MiDaS v2.1 Small contains its normalization; do not add it twice.
    return np.ascontiguousarray(rgb[None], dtype=np.float32)


def encode_depth(image, depth, model, limits):
    """Color/encode the same RGB and native depth; no NPU context access."""
    import cv2
    import numpy as np
    start = time.perf_counter()
    low, high = limits
    fraction = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)
    near = fraction if model.startswith('midas') else 1 - fraction
    colored = cv2.resize(cv2.applyColorMap(np.uint8(near * 255), cv2.COLORMAP_TURBO), (image.shape[1], image.shape[0]))
    pair = np.hstack([image, colored])
    post_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    ok, encoded = cv2.imencode('.jpg', pair, [cv2.IMWRITE_JPEG_QUALITY, 82])
    jpeg_ms = (time.perf_counter() - start) * 1000
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return encoded.tobytes(), {'postprocess_ms': post_ms, 'jpeg_ms': jpeg_ms}


class ModelContext:
    def __init__(self, model, root=MODEL_ROOT, *, native_input=False):
        import numpy as np
        from rknnlite.api import RKNNLite
        self.model = model
        self.native_input = bool(native_input)
        if self.native_input and model != 'midas-optimized-int8':
            raise ValueError('Native exact input is only verified for the pinned optimized MiDaS model')
        self.execution_lock = None
        artifact = MODELS[model].get('artifact')
        model_path = HERE / artifact if artifact else root / (model + '.rknn')
        actual_model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if actual_model_sha != MODELS[model]['sha256']:
            raise RuntimeError('Model SHA does not match the tested asset')
        if self.native_input:
            from exact_input import MODEL_SHA256
            if actual_model_sha != MODEL_SHA256:
                raise RuntimeError('Native input profile does not match this model SHA')
        self.runtime = RKNNLite(verbose=False)
        try:
            if self.runtime.load_rknn(str(model_path)) != 0 or self.runtime.init_runtime() != 0:
                raise RuntimeError('NPU initialization failed')
            reference_model = MODELS[model].get('reference_model', model)
            reference = HERE / ('clear_compare_' + reference_model + '_01') / 'canonical_depth_native.npy'
            self.limits = np.percentile(np.load(reference, allow_pickle=False), [2, 98]) if reference.is_file() else None
            if self.native_input:
                from exact_input import verify_runtime
                self.input_selftest = verify_runtime(self.runtime)
                print('NATIVE_INPUT_SELFTEST=' + json.dumps(self.input_selftest), flush=True)
        except Exception:
            self.runtime.release()
            raise

    def infer_tensor(self, tensor):
        import numpy as np
        native = getattr(self, 'native_input', False)
        if native and (tensor.dtype != np.int8 or tensor.shape != (1, 256, 256, 3)
                       or not tensor.flags.c_contiguous):
            raise ValueError('Native input context requires contiguous NHWC int8 256 input')
        passthrough = [1] if native else None
        start = time.perf_counter()
        if self.execution_lock is None:
            options = {'inputs_pass_through': passthrough} if native else {}
            outputs = self.runtime.inference(inputs=[tensor], data_format=['nhwc'], **options)
        else:
            # Each worker owns its input buffer/context. Copy/conversion may
            # overlap another context's NPU run; hardware runs stay ordered.
            backend = self.runtime.rknn_runtime
            backend.set_inputs([tensor], None, ['nhwc'], inputs_pass_through=passthrough)
            with self.execution_lock:
                backend.run(False)
                outputs = backend.get_outputs(False)
        inference_ms = (time.perf_counter() - start) * 1000
        if not outputs or not np.isfinite(outputs[0]).all():
            raise RuntimeError('Invalid NPU depth output')
        depth = np.squeeze(outputs[0])
        if depth.ndim != 2:
            raise RuntimeError('Expected a 2D depth output')
        return depth, inference_ms

    def reference_limits(self, depth):
        import numpy as np
        if self.limits is None:
            self.limits = np.percentile(depth, [2, 98])
        return tuple(self.limits)

    def render(self, image):
        start = time.perf_counter()
        tensor = prepare_model_input(image, self.model)
        if self.native_input:
            from exact_input import quantize_tensor
            tensor = quantize_tensor(tensor)
        pre_ms = (time.perf_counter() - start) * 1000
        depth, inference_ms = self.infer_tensor(tensor)
        jpeg, timings = encode_depth(image, depth, self.model, self.reference_limits(depth))
        return jpeg, {'preprocess_ms': pre_ms, 'inference_ms': inference_ms, **timings}

    def close(self):
        self.runtime.release()


def capture_worker(runtime, stopping, cv_threads=1):
    import cv2
    import numpy as np
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstVideo', '1.0')
    from gi.repository import Gst, GstVideo
    Gst.init(None)
    cv2.setNumThreads(cv_threads)
    description = ('v4l2src device=/dev/video0 io-mode=4 ! '
                   'video/x-raw,format=NV12,width=640,height=360 ! videoconvert ! '
                   'video/x-raw,format=BGR ! appsink name=frames max-buffers=1 drop=true sync=false')
    recoveries = 0
    while not stopping.is_set():
        pipeline = Gst.parse_launch(description)
        sink = pipeline.get_by_name('frames')
        try:
            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError('Camera failed to enter PLAYING')
            while not stopping.is_set():
                sample = sink.emit('try-pull-sample', 2 * Gst.SECOND)
                if sample is None:
                    message = pipeline.get_bus().timed_pop_filtered(0, Gst.MessageType.ERROR)
                    raise RuntimeError(str(message.parse_error()) if message else 'No camera frame for 2 seconds')
                info = GstVideo.VideoInfo.new_from_caps(sample.get_caps())
                buffer = sample.get_buffer()
                ok, mapped = buffer.map(Gst.MapFlags.READ)
                if not ok:
                    raise RuntimeError('Unable to map camera buffer')
                try:
                    image = np.frombuffer(mapped.data, np.uint8).reshape(info.height, info.stride[0])[:, :info.width * 3]
                    image = image.reshape(info.height, info.width, 3).copy()
                finally:
                    buffer.unmap(mapped)
                clock = pipeline.get_clock()
                current = clock.get_time() - pipeline.get_base_time() if clock else None
                age = (current - buffer.pts) / 1e9 if current is not None and buffer.pts != Gst.CLOCK_TIME_NONE else 0
                received = time.monotonic()
                runtime.publish_frame(image, received - age, received)
        except Exception as error:
            recoveries += 1
            runtime.capture_failed(repr(error), recoveries)
        finally:
            pipeline.set_state(Gst.State.NULL)
        if recoveries > 3 or stopping.wait(1):
            return


def processing_worker(runtime, stopping):
    renderer = None
    active_epoch = -1
    after_source = 0
    try:
        while not stopping.is_set():
            requested, epoch = runtime.model_request()
            if active_epoch != epoch:
                if renderer is not None:
                    renderer.close()
                    renderer = None
                try:
                    renderer = ModelContext(requested)
                except Exception as error:
                    runtime.model_failed(repr(error), epoch)
                    active_epoch = epoch
                    continue
                if not runtime.activate_model(requested, epoch):
                    renderer.close()
                    renderer = None
                    continue
                active_epoch = epoch
                after_source = 0
            if renderer is None:
                stopping.wait(.1)
                continue
            frame = runtime.wait_for_frame(after_source, active_epoch, timeout=.5)
            if frame is None:
                continue
            after_source = frame.seq
            try:
                jpeg, timings = renderer.render(frame.image)
                runtime.publish_result(frame, jpeg, renderer.model, active_epoch, timings)
            except Exception as error:
                runtime.model_failed(repr(error), active_epoch)
                renderer.close()
                renderer = None
    finally:
        if renderer is not None:
            renderer.close()


def make_handler(runtime, html_path, results_root=None):
    html_path = Path(html_path)
    results_root = Path(results_root).resolve() if results_root else None

    def state_payload():
        hardware = {}
        for key, path in {'soc_c': '/sys/class/thermal/thermal_zone0/temp',
                          'cpu_max_khz': '/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq',
                          'cpu_nominal_max_khz': '/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_max_freq',
                          'npu_hz': '/sys/class/devfreq/fde40000.npu/cur_freq'}.items():
            try:
                value = int(Path(path).read_text())
                hardware[key] = value / 1000 if key == 'soc_c' else value
            except (OSError, ValueError):
                pass
        return {**runtime.snapshot(), 'hardware': hardware,
                'sensor_mode': getattr(runtime, 'sensor_mode', None),
                'pipeline_mode': getattr(runtime, 'pipeline_mode', 'serial'),
                'cv_threads': getattr(runtime, 'cv_threads', 1),
                'cpu_stages': getattr(runtime, 'cpu_stages', 'reference'),
                'input_transport': getattr(runtime, 'input_transport', 'sdk'),
                'inference_workers': getattr(runtime, 'inference_workers', 1),
                'npu_max_freq_hz': getattr(runtime, 'npu_max_freq_hz', None),
                'npu_min_freq_hz': getattr(runtime, 'npu_min_freq_hz', None)}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connection.settimeout(20)

        def send(self, code, body=b'', content_type='application/json', headers=None):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Server-Monotonic-Ms', str(time.monotonic() * 1000))
            for key, value in (headers or {}).items():
                self.send_header(key, str(value))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def send_json(self, code, value):
            self.send(code, json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8'))

        def frame_headers(self, result):
            return {'X-Frame-Id': result.seq, 'X-Source-Frame-Id': result.source_seq,
                    'X-Model-Id': result.model, 'X-Model-Epoch': result.epoch,
                    'X-Source-Monotonic-Ms': result.source_monotonic * 1000,
                    'X-Encoded-Monotonic-Ms': result.encoded_monotonic * 1000,
                    'X-Inference-Ms': result.timings['inference_ms'],
                    'X-Driver-To-Encoded-Ms': result.timings['driver_to_encoded_ms']}

        def stream(self, after, epoch):
            if epoch != runtime.model_request()[1]:
                return self.send(204, headers={'X-Model-Epoch': runtime.model_request()[1]})
            self.close_connection = True
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
            self.connection.settimeout(2)
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=depthframe')
            self.send_header('Transfer-Encoding', 'chunked')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Connection', 'close')
            self.end_headers()
            while True:
                result, reason = runtime.wait_for_result(after, epoch, 15)
                if result is None:
                    break
                headers = {'Content-Type': 'image/jpeg', 'Content-Length': len(result.jpeg), **self.frame_headers(result)}
                prefix = b'--depthframe\r\n' + ''.join(f'{key}: {value}\r\n' for key, value in headers.items()).encode('ascii') + b'\r\n'
                part = prefix + result.jpeg + b'\r\n'
                self.wfile.write(f'{len(part):x}\r\n'.encode('ascii') + part + b'\r\n')
                self.wfile.flush()
                after = result.seq
            self.wfile.write(b'0\r\n\r\n')
            self.wfile.flush()

        def do_GET(self):
            parsed = urlsplit(self.path)
            try:
                if parsed.path == '/':
                    return self.send(200, html_path.read_bytes(), 'text/html; charset=utf-8')
                if parsed.path == '/api/config':
                    return self.send_json(200, {'models': [{'id': key, 'label': value['label']} for key, value in MODELS.items()],
                                                'selected_model': runtime.model_request()[0], 'uncapped_preview': True,
                                                'frame_transport': 'multipart'})
                if parsed.path == '/api/state':
                    return self.send_json(200, state_payload())
                if parsed.path == '/status':
                    return self.send_json(200, {'error': '页面已升级，请刷新以使用实时模型切换。', 'phase': 'reload_required'})
                if parsed.path in ('/api/frame', '/api/stream'):
                    query = parse_qs(parsed.query)
                    after = int(query.get('after', ['0'])[0])
                    epoch = int(query.get('epoch', [str(runtime.model_request()[1])])[0])
                    timeout = float(query.get('timeout', ['15'])[0])
                    if not 0 <= after <= 2**63 - 1 or epoch < 0 or not 0 <= timeout <= 15:
                        raise ValueError('Invalid frame request')
                    if parsed.path == '/api/stream':
                        return self.stream(after, epoch)
                    result, reason = runtime.wait_for_result(after, epoch, timeout)
                    if result is None:
                        code = 503 if reason in ('fault', 'stopped') else 204
                        return self.send(code, headers={'X-Model-Epoch': runtime.model_request()[1], 'X-Frame-Status': reason})
                    return self.send(200, result.jpeg, 'image/jpeg', self.frame_headers(result))
                if parsed.path.startswith('/results/') and results_root is not None:
                    target = (results_root / unquote(parsed.path[len('/results/'):])).resolve()
                    if target.is_relative_to(results_root) and target.is_file() and target.suffix in ('.html', '.json', '.jpg', '.png', '.csv'):
                        mime = {'.html': 'text/html; charset=utf-8', '.json': 'application/json', '.jpg': 'image/jpeg', '.png': 'image/png', '.csv': 'text/csv'}
                        return self.send(200, target.read_bytes(), mime[target.suffix])
                return self.send_json(404, {'error': 'Not found'})
            except (ValueError, TypeError) as error:
                self.send_json(400, {'error': str(error)})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                return

        def do_POST(self):
            if urlsplit(self.path).path != '/api/model':
                self.close_connection = True
                return self.send_json(404, {'error': 'Not found'})
            origin = self.headers.get('Origin')
            if origin and origin not in ('http://' + self.headers.get('Host', ''), 'https://' + self.headers.get('Host', '')):
                self.close_connection = True
                return self.send_json(403, {'error': 'Same-origin model control only'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    self.close_connection = True
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(length).decode('utf-8'))
                if not isinstance(body, dict):
                    raise ValueError('Expected JSON object')
                if body.get('model') not in MODELS:
                    raise ValueError('Model is not included in this release')
                runtime.request_model(body.get('model'))
                self.send_json(202, {'ok': True, **state_payload()})
            except (ValueError, TypeError, UnicodeError) as error:
                self.send_json(400, {'error': str(error)})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                return

        def log_message(self, *args):
            pass

    return Handler


@contextmanager
def temporary_npu_ceiling(frequency, root=Path('/sys/class/devfreq/fde40000.npu')):
    """Test a lower supported ceiling without changing thermal/governor control."""
    if frequency is None:
        yield
        return
    available = [int(value) for value in (root / 'available_frequencies').read_text().split()]
    path = root / 'max_freq'
    previous = path.read_text().strip()
    if (frequency not in available or frequency > int(previous)
            or frequency < int((root / 'min_freq').read_text())):
        raise ValueError('Requested NPU ceiling is outside the available range')
    try:
        path.write_text(str(frequency))
        if int(path.read_text()) != frequency:
            raise RuntimeError('NPU frequency ceiling readback mismatch')
        yield
    finally:
        if path.read_text().strip() == str(frequency):
            path.write_text(previous)


@contextmanager
def temporary_npu_floor(frequency, root=Path('/sys/class/devfreq/fde40000.npu')):
    """Use only a kernel-advertised operating point; restore the prior floor on exit."""
    if frequency is None:
        yield
        return
    available = [int(value) for value in (root / 'available_frequencies').read_text().split()]
    if frequency not in available or frequency > int((root / 'max_freq').read_text()):
        raise ValueError('Requested NPU floor is not an available operating point')
    path = root / 'min_freq'
    previous = path.read_text().strip()
    try:
        path.write_text(str(frequency))
        if int(path.read_text()) != frequency:
            raise RuntimeError('NPU frequency floor readback mismatch')
        yield
    finally:
        # Preserve another operator's later change rather than overwriting it.
        if path.read_text().strip() == str(frequency):
            path.write_text(previous)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8790)
    parser.add_argument('--model', choices=list(MODELS), default='midas-optimized-int8')
    parser.add_argument('--results-root', default=str(HERE / 'clear_depth_report'))
    parser.add_argument('--serial', action='store_true', help='Use the previous sequential processing path for A/B measurements')
    parser.add_argument('--cv-threads', type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument('--npu-min-frequency', type=int, help='Temporary kernel-supported NPU frequency floor in Hz')
    parser.add_argument('--npu-max-frequency', type=int, help='Temporary lower kernel-supported NPU frequency ceiling in Hz')
    parser.add_argument('--inference-workers', type=int, choices=[1, 2], default=1)
    parser.add_argument('--input-transport', choices=['sdk', 'native-exact'], default='sdk',
                        help='Use the verified model-specific native input conversion, or the SDK reference path')
    parser.add_argument('--sensor-mode', choices=['1920x1080', 'current'], default='1920x1080',
                        help='Restore the tested CSI mode after boot, or explicitly preserve the current mode')
    parser.add_argument('--reference-cpu-stages', action='store_true',
                        help='Use the frozen CPU numerical stages for exact A/B comparisons')
    args = parser.parse_args()
    if args.serial and args.input_transport != 'sdk':
        parser.error('Native exact input requires the overlapped pipeline; use SDK transport for serial A/B')
    with temporary_npu_floor(args.npu_min_frequency), temporary_npu_ceiling(args.npu_max_frequency):
        serve(args)


def serve(args):
    from camera_sensor import configure_sensor
    sensor_mode = configure_sensor(args.sensor_mode)
    print('CSI_MODE=' + json.dumps(sensor_mode), flush=True)
    runtime = PreviewRuntime(args.model)
    runtime.sensor_mode = {key: value for key, value in sensor_mode.items() if key != 'commands'}
    runtime.pipeline_mode = 'serial' if args.serial else 'overlapped'
    runtime.cv_threads = args.cv_threads
    runtime.npu_min_freq_hz = args.npu_min_frequency
    runtime.npu_max_freq_hz = args.npu_max_frequency
    runtime.inference_workers = 1 if args.serial else args.inference_workers
    runtime.cpu_stages = 'reference' if args.serial or args.reference_cpu_stages else 'exact_lut_buffers'
    runtime.input_transport = args.input_transport
    if runtime.cpu_stages == 'reference':
        prepare_fn, encode_fn = prepare_model_input, encode_depth
    else:
        from preview_numerics import prepare_model_input as prepare_fn, encode_depth as encode_fn
    if args.input_transport == 'native-exact':
        from exact_input import quantize_tensor
        reference_prepare = prepare_fn

        def prepare_fn(image, model):
            tensor = reference_prepare(image, model)
            return quantize_tensor(tensor) if model == 'midas-optimized-int8' else tensor
    stopping = threading.Event()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(runtime, HERE / 'camera_preview_ui.html', args.results_root))
    server.daemon_threads = True
    capture = threading.Thread(target=capture_worker, args=(runtime, stopping, args.cv_threads), name='camera-capture', daemon=True)
    prepared = LatestSlot()
    depths = LatestSlot(key=lambda job: (job.epoch, job.frame.seq))
    execution_lock = threading.Lock()

    def context_factory(model):
        context = ModelContext(model, native_input=(args.input_transport == 'native-exact'
                                                   and model == 'midas-optimized-int8'))
        if args.inference_workers > 1:
            context.execution_lock = execution_lock
        return context
    if args.serial:
        workers = [threading.Thread(target=processing_worker, args=(runtime, stopping), name='npu-model', daemon=True)]
    else:
        workers = [threading.Thread(target=prepare_worker, args=(runtime, stopping, prepared, prepare_fn), name='preprocess', daemon=True),
                   *[threading.Thread(target=inference_worker, args=(runtime, stopping, prepared, depths, context_factory), name=f'npu-model-{i}', daemon=True)
                     for i in range(args.inference_workers)],
                   threading.Thread(target=encode_worker, args=(runtime, stopping, depths, encode_fn), name='depth-encode', daemon=True)]
    capture.start()
    for worker in workers:
        worker.start()
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    try:
        server.serve_forever()
    finally:
        stopping.set()
        runtime.stop()
        prepared.close()
        depths.close()
        server.server_close()
        capture.join(4)
        for worker in workers:
            worker.join(4)


if __name__ == '__main__':
    main()
