"""Offline behavioral tests; no camera, NumPy, GStreamer, or board imports.

Run from the project root with:
    py -3.13 -m unittest discover -s camera_test_20261027 -p test_preview_runtime.py -v
"""
from concurrent.futures import Future
import queue
import threading
import unittest

if __package__:
    from .preview_runtime import MODELS, PreviewRuntime
else:
    from preview_runtime import MODELS, PreviewRuntime


JPEG = b'\xff\xd8offline-test\xff\xd9'
DEFAULT_MODEL = 'fast-depth-int8'
OTHER_MODEL = 'midas-exact-fp16'
TIMING_KEYS = ('preprocess_ms', 'inference_ms', 'postprocess_ms', 'jpeg_ms')


class FakeClock:
    """Control data age without changing the real Condition wait deadline."""

    def __init__(self, now=100.0):
        self._now = now
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            return self._now

    def set(self, now):
        with self._lock:
            self._now = now

    def advance(self, seconds):
        with self._lock:
            self._now += seconds


class ObservedCondition(threading.Condition):
    """Expose entry to a real wait so tests never race a thread's startup."""

    def __init__(self):
        super().__init__(threading.RLock())
        self.wait_entered = queue.Queue()

    def wait(self, timeout=None):
        self.wait_entered.put(threading.get_ident())
        return super().wait(timeout)


class PreviewRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.runtime = PreviewRuntime(clock=self.clock)
        self.runtime.condition = ObservedCondition()
        self.addCleanup(self.runtime.stop)
        self.model, self.epoch = self.runtime.model_request()

    def activate(self):
        self.assertTrue(self.runtime.activate_model(self.model, self.epoch))

    def frame(self, image=None, *, source=None, received=None):
        if source is None:
            source = self.clock()
        frame = self.runtime.publish_frame(image, source, received)
        self.assertIsNotNone(frame)
        return frame

    def result(self, frame, *, model=None, epoch=None, timings=None, encoded=None):
        result = self.runtime.publish_result(
            frame, JPEG, model or self.model,
            self.epoch if epoch is None else epoch,
            dict.fromkeys(TIMING_KEYS, 1.0) if timings is None else timings,
            encoded,
        )
        self.assertIsNotNone(result)
        return result

    def _background(self, action, *, release=None):
        future = Future()

        def run():
            try:
                future.set_result(action())
            except BaseException as error:
                future.set_exception(error)

        thread = threading.Thread(target=run, daemon=True)

        def cleanup():
            self.runtime.stop()
            if release is not None:
                release.set()
            thread.join(1.0)
            self.assertFalse(thread.is_alive(), 'background test worker did not stop')

        self.addCleanup(cleanup)
        thread.start()
        return future

    def waiter(self, action):
        future = self._background(action)
        self.runtime.condition.wait_entered.get(timeout=1.0)
        self.assertFalse(future.done())
        return future

    def gated(self, action):
        entered = threading.Event()
        release = threading.Event()

        def run():
            entered.set()
            if not release.wait(5.0):
                raise AssertionError('test did not release in-flight work')
            return action()

        future = self._background(run, release=release)
        self.assertTrue(entered.wait(1.0))
        return future, release

    def test_initial_state_waits_for_model_and_camera(self):
        state = self.runtime.snapshot()
        self.assertEqual(state['phase'], 'loading')
        self.assertEqual(state['requested_model'], DEFAULT_MODEL)
        self.assertIsNone(state['active_model'])
        self.assertIsNone(state['result'])
        self.assertEqual(state['capture']['fps'], 0.0)
        self.assertEqual(state['processing']['frames'], 0)
        self.activate()
        self.assertEqual(self.runtime.snapshot()['phase'], 'waiting_camera')

    def test_slow_consumer_skips_intermediate_frames(self):
        first = self.frame(object())
        self.assertIs(self.runtime.wait_for_frame(0, self.epoch, timeout=0), first)
        intermediate = [self.frame(object()) for _ in range(50)]
        newest = intermediate[-1]
        consumed = self.runtime.wait_for_frame(first.seq, self.epoch, timeout=0)
        self.assertIs(consumed, newest)
        self.assertEqual(consumed.seq, first.seq + 50)
        self.assertIsNone(self.runtime.wait_for_frame(newest.seq, self.epoch, timeout=0))

    def test_in_flight_processing_can_finish_while_capture_advances(self):
        self.activate()
        processing_frame = self.frame('being processed')
        newest = self.frame('newest capture')
        result = self.result(processing_frame)
        self.assertEqual(result.source_seq, processing_frame.seq)
        self.assertIs(self.runtime.wait_for_frame(processing_frame.seq, self.epoch, 0), newest)

    def test_result_delivery_uses_after_seq_and_keeps_only_latest(self):
        self.activate()
        first = self.result(self.frame())
        self.assertEqual(self.runtime.wait_for_result(0, self.epoch, 0), (first, 'ready'))
        self.assertEqual(self.runtime.wait_for_result(first.seq, self.epoch, 0), (None, 'timeout'))
        self.result(self.frame())
        newest = self.result(self.frame())
        self.assertEqual(self.runtime.wait_for_result(first.seq, self.epoch, 0), (newest, 'ready'))
        self.assertEqual(self.runtime.wait_for_result(newest.seq, self.epoch, 0), (None, 'timeout'))

    def test_model_switch_clears_old_result_and_processing_statistics(self):
        self.activate()
        old_result = self.result(self.frame(), timings={'inference_ms': 999.0})
        new_epoch = self.runtime.request_model(OTHER_MODEL)
        self.assertGreater(new_epoch, self.epoch)
        state = self.runtime.snapshot()
        self.assertEqual(state['phase'], 'loading')
        self.assertIsNone(state['active_model'])
        self.assertIsNone(state['result'])
        self.assertEqual(state['processing']['frames'], 0)
        self.assertEqual(state['processing']['fps'], 0.0)
        self.assertIsNone(state['processing']['inference_ms'])
        self.assertIsNone(state['processing']['inference_p95_ms'])
        self.assertEqual(self.runtime.wait_for_result(old_result.seq, self.epoch, 0), (None, 'changed'))
        self.assertTrue(self.runtime.activate_model(OTHER_MODEL, new_epoch))
        new_result = self.result(self.frame(), model=OTHER_MODEL, epoch=new_epoch,
                                 timings={'inference_ms': 2.0})
        self.assertGreater(new_result.seq, old_result.seq)
        state = self.runtime.snapshot()
        self.assertEqual(state['processing']['frames'], 1)
        self.assertEqual(state['processing']['inference_ms'], 2.0)
        self.assertEqual(state['result']['model'], OTHER_MODEL)

    def test_requesting_same_healthy_model_preserves_result_and_epoch(self):
        self.activate()
        result = self.result(self.frame())
        self.assertEqual(self.runtime.request_model(self.model), self.epoch)
        self.assertEqual(self.runtime.wait_for_result(0, self.epoch, 0), (result, 'ready'))
        self.assertEqual(self.runtime.snapshot()['processing']['frames'], 1)

    def test_failed_model_can_be_retried_with_a_new_epoch(self):
        self.runtime.model_failed(RuntimeError('initialization failed'), self.epoch)
        self.assertEqual(self.runtime.snapshot()['phase'], 'fault')
        new_epoch = self.runtime.request_model(self.model)
        self.assertGreater(new_epoch, self.epoch)
        self.assertTrue(self.runtime.activate_model(self.model, new_epoch))
        self.assertIsNone(self.runtime.snapshot()['error'])

    def test_late_old_model_initialization_cannot_replace_new_selection(self):
        future, release = self.gated(lambda: self.runtime.activate_model(self.model, self.epoch))
        new_epoch = self.runtime.request_model(OTHER_MODEL)
        self.assertTrue(self.runtime.activate_model(OTHER_MODEL, new_epoch))
        release.set()
        self.assertFalse(future.result(timeout=1.0))
        state = self.runtime.snapshot()
        self.assertEqual(state['active_model'], OTHER_MODEL)
        self.assertEqual(state['phase'], 'waiting_camera')

    def test_late_old_model_failure_cannot_poison_new_selection(self):
        future, release = self.gated(lambda: self.runtime.model_failed('old load failed', self.epoch))
        new_epoch = self.runtime.request_model(OTHER_MODEL)
        self.assertTrue(self.runtime.activate_model(OTHER_MODEL, new_epoch))
        release.set()
        future.result(timeout=1.0)
        state = self.runtime.snapshot()
        self.assertEqual(state['active_model'], OTHER_MODEL)
        self.assertIsNone(state['error'])
        self.assertEqual(state['phase'], 'waiting_camera')

    def test_late_in_flight_result_cannot_overwrite_new_epoch_result(self):
        self.activate()
        old_frame = self.frame()
        future, release = self.gated(lambda: self.runtime.publish_result(
            old_frame, JPEG, self.model, self.epoch, {'inference_ms': 999.0}))
        new_epoch = self.runtime.request_model(OTHER_MODEL)
        self.assertTrue(self.runtime.activate_model(OTHER_MODEL, new_epoch))
        current = self.result(self.frame(), model=OTHER_MODEL, epoch=new_epoch,
                              timings={'inference_ms': 3.0})
        release.set()
        self.assertIsNone(future.result(timeout=1.0))
        self.assertEqual(self.runtime.wait_for_result(0, new_epoch, 0), (current, 'ready'))
        self.assertEqual(self.runtime.snapshot()['processing']['inference_ms'], 3.0)
        self.assertEqual(self.runtime.snapshot()['processing']['frames'], 1)

    def test_wrong_model_cannot_activate_or_publish_under_current_epoch(self):
        self.assertFalse(self.runtime.activate_model(OTHER_MODEL, self.epoch))
        self.activate()
        frame = self.frame()
        self.assertIsNone(self.runtime.publish_result(frame, JPEG, OTHER_MODEL, self.epoch, {}))
        self.assertIsNone(self.runtime.snapshot()['result'])

    def test_frame_wait_is_woken_by_new_frame(self):
        future = self.waiter(lambda: self.runtime.wait_for_frame(0, self.epoch, timeout=5))
        frame = self.frame()
        self.assertIs(future.result(timeout=1.0), frame)

    def test_frame_wait_is_woken_by_model_change(self):
        future = self.waiter(lambda: self.runtime.wait_for_frame(0, self.epoch, timeout=5))
        self.runtime.request_model(OTHER_MODEL)
        self.assertIsNone(future.result(timeout=1.0))

    def test_frame_wait_is_woken_by_stop(self):
        future = self.waiter(lambda: self.runtime.wait_for_frame(0, self.epoch, timeout=5))
        self.runtime.stop()
        self.assertIsNone(future.result(timeout=1.0))

    def test_result_wait_is_woken_by_new_result(self):
        self.activate()
        frame = self.frame()
        future = self.waiter(lambda: self.runtime.wait_for_result(0, self.epoch, timeout=5))
        result = self.result(frame)
        self.assertEqual(future.result(timeout=1.0), (result, 'ready'))

    def test_new_frame_wakes_result_wait_but_does_not_deliver_unprocessed_frame(self):
        self.activate()
        future = self.waiter(lambda: self.runtime.wait_for_result(0, self.epoch, timeout=5))
        frame = self.frame()
        self.runtime.condition.wait_entered.get(timeout=1.0)
        self.assertFalse(future.done())
        result = self.result(frame)
        self.assertEqual(future.result(timeout=1.0), (result, 'ready'))

    def test_result_wait_is_woken_by_model_change(self):
        future = self.waiter(lambda: self.runtime.wait_for_result(0, self.epoch, timeout=5))
        self.runtime.request_model(OTHER_MODEL)
        self.assertEqual(future.result(timeout=1.0), (None, 'changed'))

    def test_result_wait_is_woken_by_stop(self):
        future = self.waiter(lambda: self.runtime.wait_for_result(0, self.epoch, timeout=5))
        self.runtime.stop()
        self.assertEqual(future.result(timeout=1.0), (None, 'stopped'))

    def test_result_wait_is_woken_by_capture_fault(self):
        future = self.waiter(lambda: self.runtime.wait_for_result(0, self.epoch, timeout=5))
        self.runtime.capture_failed('camera disconnected', recoveries=1)
        self.assertEqual(future.result(timeout=1.0), (None, 'fault'))

    def test_result_wait_is_woken_by_model_failure(self):
        future = self.waiter(lambda: self.runtime.wait_for_result(0, self.epoch, timeout=5))
        self.runtime.model_failed('model unavailable', self.epoch)
        self.assertEqual(future.result(timeout=1.0), (None, 'fault'))

    def test_stale_capture_is_not_consumed_or_reported_as_live(self):
        self.activate()
        frame = self.frame()
        self.result(frame)
        self.clock.advance(self.runtime.freshness_s + .001)
        self.assertIsNone(self.runtime.wait_for_frame(0, self.epoch, timeout=0))
        self.assertEqual(self.runtime.wait_for_result(0, self.epoch, 0), (None, 'timeout'))
        self.assertIsNone(self.runtime.publish_result(frame, JPEG, self.model, self.epoch, {}))
        state = self.runtime.snapshot()
        self.assertEqual(state['phase'], 'waiting_camera')
        self.assertIsNone(state['result'])

    def test_stale_result_is_hidden_even_when_capture_is_fresh(self):
        self.activate()
        old = self.result(self.frame())
        self.clock.advance(self.runtime.freshness_s + .001)
        self.frame('capture is still working')
        self.assertEqual(self.runtime.wait_for_result(0, self.epoch, 0), (None, 'timeout'))
        self.assertIsNone(self.runtime.snapshot()['result'])
        current = self.result(self.frame())
        self.assertGreater(current.seq, old.seq)
        self.assertEqual(self.runtime.wait_for_result(old.seq, self.epoch, 0), (current, 'ready'))

    def test_stale_capture_hides_an_independently_fresh_result(self):
        self.activate()
        frame = self.frame()
        self.clock.advance(.5)
        self.result(frame)
        self.clock.advance(.501)
        self.assertIsNone(self.runtime.snapshot()['result'])
        self.assertEqual(self.runtime.wait_for_result(0, self.epoch, 0), (None, 'timeout'))

    def test_old_source_frame_cannot_be_published_as_a_new_live_result(self):
        self.activate()
        frame = self.frame(source=self.clock() - self.runtime.freshness_s - .001)
        self.assertIsNone(self.runtime.publish_result(frame, JPEG, self.model, self.epoch, {}))
        self.assertEqual(self.runtime.snapshot()['processing']['frames'], 0)

    def test_capture_fault_hides_old_result_and_recovery_requires_a_new_result(self):
        self.activate()
        before_fault = self.frame()
        old_result = self.result(before_fault)
        self.runtime.capture_failed('MIPI unavailable', recoveries=1)
        state = self.runtime.snapshot()
        self.assertEqual(state['phase'], 'recovering')
        self.assertEqual(state['error'], 'MIPI unavailable')
        self.assertIsNone(state['result'])
        self.assertIsNone(state['capture']['age_ms'])
        self.assertEqual(self.runtime.wait_for_result(0, self.epoch, 0), (None, 'fault'))
        self.assertIsNone(self.runtime.publish_result(before_fault, JPEG, self.model, self.epoch, {}))
        recovered_frame = self.frame('new capture')
        self.assertGreater(recovered_frame.seq, before_fault.seq)
        self.assertEqual(self.runtime.snapshot()['phase'], 'waiting_camera')
        self.assertIsNone(self.runtime.snapshot()['result'])
        recovered_result = self.result(recovered_frame)
        self.assertGreater(recovered_result.seq, old_result.seq)
        self.assertEqual(self.runtime.wait_for_result(old_result.seq, self.epoch, 0),
                         (recovered_result, 'ready'))
        self.assertEqual(self.runtime.snapshot()['phase'], 'running')
        self.assertIsNone(self.runtime.snapshot()['error'])

    def test_in_flight_pre_fault_frame_cannot_resurface_after_capture_recovers(self):
        self.activate()
        before_fault = self.frame('old generation')
        future, release = self.gated(lambda: self.runtime.publish_result(
            before_fault, JPEG, self.model, self.epoch, {'inference_ms': 999.0}))
        self.runtime.capture_failed('transient capture error', recoveries=1)
        current = self.result(self.frame('recovered generation'), timings={'inference_ms': 2.0})
        release.set()
        self.assertIsNone(future.result(timeout=1.0))
        self.assertEqual(self.runtime.wait_for_result(0, self.epoch, 0), (current, 'ready'))
        self.assertEqual(self.runtime.snapshot()['processing']['inference_ms'], 2.0)
        self.assertEqual(self.runtime.snapshot()['processing']['frames'], 1)

    def test_exhausted_capture_recovery_is_a_fault(self):
        self.activate()
        self.runtime.capture_failed('capture retries exhausted', recoveries=4)
        self.assertEqual(self.runtime.snapshot()['phase'], 'fault')
        self.assertEqual(self.runtime.snapshot()['capture']['recoveries'], 4)

    def test_only_whitelisted_models_are_accepted(self):
        self.assertEqual(set(MODELS), {
            'fast-depth-int8', 'fast-depth-fp16', 'midas-exact-int8', 'midas-exact-fp16', 'midas-optimized-int8'})
        for model in MODELS:
            with self.subTest(model=model):
                runtime = PreviewRuntime(model, clock=self.clock)
                self.assertEqual(runtime.model_request()[0], model)
                runtime.stop()
        for invalid in ('', 'unknown', '../model.rknn', 'FAST-DEPTH-INT8', None, [], {}, 123):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    PreviewRuntime(invalid, clock=self.clock)
                with self.assertRaises(ValueError):
                    self.runtime.request_model(invalid)
                self.assertEqual(self.runtime.model_request(), (self.model, self.epoch))

    def test_invalid_frame_timestamps_do_not_advance_capture(self):
        for invalid in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(timestamp=invalid):
                with self.assertRaises(ValueError):
                    self.runtime.publish_frame(None, invalid)
                with self.assertRaises(ValueError):
                    self.runtime.publish_frame(None, self.clock(), invalid)
        self.assertEqual(self.runtime.snapshot()['capture']['frames'], 0)
        self.assertEqual(self.frame().seq, 1)

    def test_invalid_jpeg_and_timings_do_not_publish_or_count_results(self):
        self.activate()
        frame = self.frame()
        for invalid in (b'', b'not a JPEG', bytearray(JPEG), None):
            with self.subTest(jpeg=invalid):
                with self.assertRaises(ValueError):
                    self.runtime.publish_result(frame, invalid, self.model, self.epoch, {})
        for invalid in (float('nan'), float('inf'), float('-inf'), -1.0, '12', None):
            with self.subTest(timing=invalid):
                with self.assertRaises(ValueError):
                    self.runtime.publish_result(frame, JPEG, self.model, self.epoch,
                                                {'inference_ms': invalid})
        self.assertIsNone(self.runtime.snapshot()['result'])
        self.assertEqual(self.runtime.snapshot()['processing']['frames'], 0)

    def test_invalid_encoded_timestamps_do_not_publish_or_count_results(self):
        self.activate()
        frame = self.frame()
        for invalid in (float('nan'), float('inf'), float('-inf'), self.clock() - .001, '100'):
            with self.subTest(timestamp=invalid):
                with self.assertRaises(ValueError):
                    self.runtime.publish_result(frame, JPEG, self.model, self.epoch, {}, invalid)
        self.assertIsNone(self.runtime.snapshot()['result'])
        self.assertEqual(self.runtime.snapshot()['processing']['frames'], 0)

    def test_statistics_use_measured_intervals_and_only_the_recent_window(self):
        self.runtime.window_s = 1.0
        self.activate()
        for when, inference in ((100.0, 900.0), (101.0, 800.0), (102.0, 10.0),
                                (102.2, 20.0), (102.6, 80.0)):
            self.clock.set(when)
            self.result(self.frame(), timings={'inference_ms': inference})
        state = self.runtime.snapshot()
        # The retained samples are at 102.0, 102.2 and 102.6, not count / window_s.
        self.assertAlmostEqual(state['capture']['fps'], 2.0 / .6)
        self.assertAlmostEqual(state['processing']['fps'], 2.0 / .6)
        self.assertEqual(state['capture']['frames'], 5)
        self.assertEqual(state['processing']['frames'], 5)
        self.assertEqual(state['processing']['inference_ms'], 20.0)
        self.assertEqual(state['processing']['inference_p95_ms'], 80.0)
        self.assertIsNone(state['processing']['preprocess_ms'])

    def test_p50_and_p95_are_computed_from_all_timing_samples(self):
        self.activate()
        for index in range(1, 21):
            self.clock.set(100.0 + index * .02)
            frame = self.frame(source=self.clock() - index * .001)
            self.result(frame, timings={key: float(index) for key in TIMING_KEYS})
        metrics = self.runtime.snapshot()['processing']
        self.assertAlmostEqual(metrics['fps'], 50.0)
        for key in (*TIMING_KEYS, 'driver_to_encoded_ms'):
            with self.subTest(metric=key):
                self.assertAlmostEqual(metrics[key], 10.5)
                self.assertAlmostEqual(metrics[key[:-3] + '_p95_ms'], 19.0)

    def test_idle_stream_has_zero_fps_and_expired_metrics_leave_window(self):
        self.activate()
        self.result(self.frame())
        self.clock.advance(.1)
        self.result(self.frame())
        self.assertAlmostEqual(self.runtime.snapshot()['processing']['fps'], 10.0)
        self.clock.advance(self.runtime.freshness_s + .001)
        state = self.runtime.snapshot()
        self.assertEqual(state['capture']['fps'], 0.0)
        self.assertEqual(state['processing']['fps'], 0.0)
        self.assertEqual(state['processing']['inference_ms'], 1.0)
        self.clock.advance(self.runtime.window_s)
        state = self.runtime.snapshot()
        self.assertIsNone(state['processing']['inference_ms'])
        self.assertIsNone(state['processing']['inference_p95_ms'])
        self.assertEqual(state['processing']['frames'], 2)

    def test_same_timestamp_samples_do_not_divide_by_zero(self):
        self.activate()
        self.result(self.frame())
        self.result(self.frame())
        state = self.runtime.snapshot()
        self.assertEqual(state['capture']['fps'], 0.0)
        self.assertEqual(state['processing']['fps'], 0.0)

    def test_stop_prevents_new_work_and_hides_results(self):
        self.activate()
        frame = self.frame()
        self.result(frame)
        self.runtime.stop()
        self.runtime.stop()
        self.assertIsNone(self.runtime.publish_frame(None, self.clock()))
        self.assertIsNone(self.runtime.publish_result(frame, JPEG, self.model, self.epoch, {}))
        self.assertFalse(self.runtime.activate_model(self.model, self.epoch))
        with self.assertRaises(RuntimeError):
            self.runtime.request_model(OTHER_MODEL)
        state = self.runtime.snapshot()
        self.assertEqual(state['phase'], 'stopped')
        self.assertIsNone(state['result'])


if __name__ == '__main__':
    unittest.main()
