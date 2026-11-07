"""Thread-safe newest-frame exchange and versioned model results, without hardware imports."""
from collections import deque
from dataclasses import dataclass
import math
import statistics
import threading
import time

MODELS = {
    'fast-depth-int8': {'label': 'FastDepth INT8', 'sha256': '7a03736608dcda10973f84ee45c14e1ac3dc6c94e7c4d2e55c55703a6952ec9a'},
    'fast-depth-fp16': {'label': 'FastDepth FP16', 'sha256': '3ca4445fbb863952f90ca89107302a680f1fd28eba298e17244a1d3bdbf69df0'},
    'midas-exact-int8': {'label': 'MiDaS Small INT8', 'sha256': '8e005b9f68ceafa2c653997a967ff184b1095eabc4b2f620e9a8a34989ba278f'},
    'midas-exact-fp16': {'label': 'MiDaS Small FP16', 'sha256': '73ec28ea15c8e0b9cd8bfa8e70ccc238a792f38f621b9b4c6cf13a1f58080754'},
    'midas-optimized-int8': {'label': 'MiDaS Small INT8 · 优化256',
                           'sha256': '8a22b419775cae587b5478df04f01ccc7eaf9f4a1585b705fc98781841dd0ac1',
                           'artifact': 'midas_optimization/midas-deconv256-int8.rknn',
                           'reference_model': 'midas-exact-int8'},
}


@dataclass(frozen=True)
class CameraFrame:
    seq: int
    image: object
    source_monotonic: float
    received_monotonic: float
    generation: int = 0


@dataclass(frozen=True)
class PreviewResult:
    seq: int
    source_seq: int
    jpeg: bytes
    model: str
    epoch: int
    source_monotonic: float
    encoded_monotonic: float
    timings: dict


class PreviewRuntime:
    """No frame queue: producers replace the latest immutable frame/result."""
    def __init__(self, selected_model='fast-depth-int8', *, clock=time.monotonic, freshness_s=1.0, window_s=3.0):
        if not isinstance(selected_model, str) or selected_model not in MODELS:
            raise ValueError('Unknown model')
        self.clock = clock
        self.freshness_s = freshness_s
        self.window_s = window_s
        self.condition = threading.Condition(threading.RLock())
        self.requested_model = selected_model
        self.active_model = None
        self.model_epoch = 1
        self.loading = True
        self.model_error = None
        self.capture_error = None
        self.recoveries = 0
        self.stopped = False
        self.frame = None
        self.result = None
        self.capture_seq = 0
        self.capture_generation = 0
        self.result_seq = 0
        self.processing_count = 0
        self.capture_times = deque(maxlen=2048)
        self.processing_samples = deque(maxlen=2048)

    def request_model(self, model):
        if not isinstance(model, str) or model not in MODELS:
            raise ValueError('Unknown model')
        with self.condition:
            if self.stopped:
                raise RuntimeError('Preview stopped')
            if model == self.requested_model and self.model_error is None:
                return self.model_epoch
            self.requested_model = model
            self.model_epoch += 1
            self.active_model = None
            self.loading = True
            self.model_error = None
            self.result = None
            self.processing_count = 0
            self.processing_samples.clear()
            self.condition.notify_all()
            return self.model_epoch

    def model_request(self):
        with self.condition:
            return self.requested_model, self.model_epoch

    def activate_model(self, model, epoch):
        with self.condition:
            if self.stopped or epoch != self.model_epoch or model != self.requested_model:
                return False
            self.active_model = model
            self.loading = False
            self.model_error = None
            self.condition.notify_all()
            return True

    def model_failed(self, error, epoch):
        with self.condition:
            if epoch != self.model_epoch:
                return
            self.model_error = str(error)
            self.loading = False
            self.active_model = None
            self.result = None
            self.condition.notify_all()

    def publish_frame(self, image, source_monotonic, received_monotonic=None):
        with self.condition:
            if self.stopped:
                return None
            now = self.clock() if received_monotonic is None else received_monotonic
            if not math.isfinite(source_monotonic) or not math.isfinite(now):
                raise ValueError('Frame timestamp must be finite')
            self.capture_seq += 1
            self.frame = CameraFrame(self.capture_seq, image, source_monotonic, now, self.capture_generation)
            self.capture_error = None
            self.capture_times.append(now)
            self.condition.notify_all()
            return self.frame

    def capture_failed(self, error, recoveries):
        with self.condition:
            self.capture_error = str(error)
            self.capture_generation += 1
            self.recoveries = recoveries
            self.frame = None
            self.result = None
            self.capture_times.clear()
            self.condition.notify_all()

    def wait_for_frame(self, after_seq, epoch, timeout=.5):
        deadline = time.monotonic() + max(0, timeout)
        with self.condition:
            while not self.stopped and epoch == self.model_epoch:
                now = self.clock()
                if self.frame is not None and self.frame.seq > after_seq and now - self.frame.received_monotonic <= self.freshness_s:
                    return self.frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(remaining)
            return None

    def accepts_frame(self, frame, model, epoch, active=False):
        """Validate immutable work at every pipeline boundary, including camera recovery."""
        with self.condition:
            now = self.clock()
            return (not self.stopped and frame is not None and epoch == self.model_epoch
                    and model == self.requested_model and self.model_error is None
                    and self.capture_error is None and frame.generation == self.capture_generation
                    and self.frame is not None
                    and now - self.frame.received_monotonic <= self.freshness_s
                    and now - frame.source_monotonic <= self.freshness_s
                    and (self.result is None or self.result.epoch != epoch
                         or frame.seq > self.result.source_seq)
                    and (not active or self.active_model == model))

    def publish_result(self, frame, jpeg, model, epoch, timings, encoded_monotonic=None):
        if not isinstance(jpeg, bytes) or not jpeg.startswith(b'\xff\xd8'):
            raise ValueError('Expected JPEG bytes')
        if any(not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0 for value in timings.values()):
            raise ValueError('Timings must be finite numbers')
        with self.condition:
            now = self.clock() if encoded_monotonic is None else encoded_monotonic
            if not isinstance(now, (float, int)) or not math.isfinite(now) or now < frame.source_monotonic:
                raise ValueError('Encoded timestamp must be finite and follow the source timestamp')
            if (self.stopped or epoch != self.model_epoch or model != self.active_model
                    or model != self.requested_model or self.capture_error is not None
                    or frame.generation != self.capture_generation
                    or (self.result is not None and self.result.epoch == epoch
                        and frame.seq <= self.result.source_seq)
                    or self.frame is None or self.clock() - self.frame.received_monotonic > self.freshness_s
                    or now - frame.source_monotonic > self.freshness_s):
                return None
            self.result_seq += 1
            self.processing_count += 1
            metrics = dict(timings)
            metrics['driver_to_encoded_ms'] = (now - frame.source_monotonic) * 1000
            self.result = PreviewResult(self.result_seq, frame.seq, jpeg, model, epoch,
                                        frame.source_monotonic, now, metrics)
            self.processing_samples.append((now, metrics))
            self.condition.notify_all()
            return self.result

    def _result_is_fresh(self, now):
        return (self.result is not None and self.frame is not None and self.capture_error is None
                and self.clock() - self.frame.received_monotonic <= self.freshness_s
                and now - self.result.encoded_monotonic <= self.freshness_s
                and self.result.epoch == self.model_epoch and self.result.model == self.active_model)

    def wait_for_result(self, after_seq, epoch, timeout=15):
        deadline = time.monotonic() + max(0, min(timeout, 15))
        with self.condition:
            while not self.stopped:
                if epoch != self.model_epoch:
                    return None, 'changed'
                if self.model_error is not None or self.capture_error is not None:
                    return None, 'fault'
                if self._result_is_fresh(self.clock()) and self.result.seq > after_seq:
                    return self.result, 'ready'
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, 'timeout'
                self.condition.wait(remaining)
            return None, 'stopped'

    def snapshot(self):
        with self.condition:
            now = self.clock()
            capture_times = [t for t in self.capture_times if now - t <= self.window_s]
            samples = [(t, values) for t, values in self.processing_samples if now - t <= self.window_s]
            processing = {'fps': self._fps([t for t, _ in samples], now), 'frames': self.processing_count}
            for key in ['preprocess_ms', 'inference_ms', 'postprocess_ms', 'jpeg_ms', 'driver_to_encoded_ms']:
                values = sorted(metrics[key] for _, metrics in samples if key in metrics)
                processing[key] = statistics.median(values) if values else None
                processing[key[:-3] + '_p95_ms'] = values[max(0, math.ceil(.95 * len(values)) - 1)] if values else None
            fresh = self._result_is_fresh(now)
            if self.stopped:
                phase = 'stopped'
            elif self.model_error is not None or self.recoveries > 3:
                phase = 'fault'
            elif self.loading:
                phase = 'loading'
            elif self.capture_error is not None:
                phase = 'recovering'
            elif fresh:
                phase = 'running'
            else:
                phase = 'waiting_camera'
            return {'requested_model': self.requested_model, 'active_model': self.active_model,
                    'model_epoch': self.model_epoch, 'phase': phase,
                    'error': self.model_error or self.capture_error,
                    'capture': {'fps': self._fps(capture_times, now), 'frames': self.capture_seq,
                                'age_ms': (now - self.frame.received_monotonic) * 1000 if self.frame else None,
                                'recoveries': self.recoveries},
                    'processing': processing,
                    'result': {'frame': self.result.seq, 'source_frame': self.result.source_seq,
                               'model': self.result.model, 'epoch': self.result.epoch,
                               'age_ms': (now - self.result.encoded_monotonic) * 1000,
                               'source_age_ms': (now - self.result.source_monotonic) * 1000}
                              if fresh else None,
                    'server_monotonic_ms': now * 1000}

    def _fps(self, times, now):
        if len(times) < 2 or now - times[-1] > self.freshness_s or times[-1] <= times[0]:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def stop(self):
        with self.condition:
            self.stopped = True
            self.frame = None
            self.result = None
            self.condition.notify_all()
