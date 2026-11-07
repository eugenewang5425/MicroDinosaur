"""Offline concurrency tests using the real runtime and event-gated fake work.

Run: py -3.13 -m unittest discover -s camera_test_20261027 -p test_preview_pipeline.py -v
"""
from concurrent.futures import Future
import queue
import threading
import time
import unittest

if __package__:
    from .preview_pipeline import (DepthJob, LatestSlot, PreparedJob, encode_worker,
                                   inference_worker, prepare_worker)
    from .preview_runtime import PreviewRuntime
else:
    from preview_pipeline import (DepthJob, LatestSlot, PreparedJob, encode_worker,
                                  inference_worker, prepare_worker)
    from preview_runtime import PreviewRuntime


JPEG = b'\xff\xd8pipeline-test\xff\xd9'
MODEL = 'fast-depth-int8'
OTHER = 'midas-exact-fp16'


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class ObservedSlot(LatestSlot):
    def __init__(self, key=None):
        super().__init__(key)
        self.offers = queue.Queue()
        self.takes = queue.Queue()
        self.rejections = queue.Queue()

    def offer(self, value):
        offered = super().offer(value)
        if offered:
            self.offers.put(value)
        else:
            self.rejections.put(value)
        return offered

    def take(self, timeout=.2):
        value = super().take(timeout)
        if value is not None:
            self.takes.put(value)
        return value


class ObservedCondition(threading.Condition):
    def __init__(self):
        super().__init__()
        self.waiting = queue.Queue()

    def wait(self, timeout=None):
        self.waiting.put(threading.get_ident())
        return super().wait(timeout)


class FakeContext:
    def __init__(self, infer=None, *, limits=(0.0, 10.0)):
        self.limits = limits
        self.buffer = [None]
        self.inferred = []
        self.owner_calls = [threading.get_ident()]
        self.closed = threading.Event()
        self.references = 0
        self.infer = infer

    def infer_tensor(self, tensor):
        self.owner_calls.append(threading.get_ident())
        self.inferred.append(tensor)
        if self.infer is not None:
            self.infer(tensor)
        self.buffer[0] = tensor
        return self.buffer, 52.0

    def reference_limits(self, depth):
        self.owner_calls.append(threading.get_ident())
        self.references += 1
        self.limits = (0.0, 10.0)
        return self.limits

    def close(self):
        self.owner_calls.append(threading.get_ident())
        self.closed.set()


def encode(image, depth, model, limits):
    return JPEG, {'postprocess_ms': 10.0, 'jpeg_ms': 8.0}


class PreviewPipelineTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.runtime = PreviewRuntime(clock=self.clock)
        self.stopping = threading.Event()
        self.prepared = ObservedSlot()
        self.depths = ObservedSlot()
        self.threads = []
        self.releases = []
        self.addCleanup(self.stop)

    def stop(self):
        self.stopping.set()
        self.runtime.stop()
        self.prepared.close()
        self.depths.close()
        for event in self.releases:
            event.set()
        for thread, future in self.threads:
            thread.join(2.0)
            self.assertFalse(thread.is_alive(), 'pipeline worker did not stop')
            future.result(timeout=0)

    def start(self, action, *args):
        future = Future()

        def run():
            try:
                future.set_result(action(*args))
            except BaseException as error:
                future.set_exception(error)

        thread = threading.Thread(target=run, daemon=True)
        self.threads.append((thread, future))
        thread.start()
        return future

    def gate(self):
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)

        def wait():
            entered.set()
            if not release.wait(5.0):
                raise AssertionError('test did not release gated work')

        return entered, release, wait

    def state(self, predicate):
        deadline = time.monotonic() + 2.0
        with self.runtime.condition:
            while not predicate():
                remaining = deadline - time.monotonic()
                self.assertGreater(remaining, 0, 'runtime did not reach expected state')
                self.runtime.condition.wait(remaining)

    def frame(self, image=1):
        return self.runtime.publish_frame(image, self.clock())

    def prepared_job(self, frame, model=None, epoch=None):
        requested, current = self.runtime.model_request()
        return PreparedJob(frame, model or requested, current if epoch is None else epoch,
                           frame.image, 10.0)

    def depth_job(self, frame, model=None, epoch=None):
        job = self.prepared_job(frame, model, epoch)
        return DepthJob(frame, job.model, job.epoch, [frame.image], (0.0, 10.0),
                        {'preprocess_ms': 10.0, 'inference_ms': 52.0})

    def inference(self, factory):
        return self.start(inference_worker, self.runtime, self.stopping,
                          self.prepared, self.depths, factory)

    def preparation(self, prepare=lambda image, model: image):
        return self.start(prepare_worker, self.runtime, self.stopping, self.prepared, prepare)

    def encoding(self, encoder=encode):
        return self.start(encode_worker, self.runtime, self.stopping, self.depths, encoder)

    def test_latest_slot_discards_intermediate_and_close_wakes_all_waiters(self):
        slot = LatestSlot()
        self.addCleanup(slot.close)
        for value in range(10):
            self.assertTrue(slot.offer(value))
        self.assertEqual(slot.overwrites, 9)
        self.assertEqual(slot.take(0), 9)
        self.assertIsNone(slot.take(0))
        observed = slot._condition = ObservedCondition()
        waiting = [self.start(slot.take, 5.0) for _ in range(3)]
        for _ in waiting:
            observed.waiting.get(timeout=1.0)
        slot.close()
        for future in waiting:
            self.assertIsNone(future.result(timeout=1.0))
        self.assertFalse(slot.offer(11))
        slot.close()
        self.assertIsNone(slot.take(5.0))

    def test_three_stages_overlap_and_depth_buffer_is_owned_by_encoding_job(self):
        preparing3, release_prepare3, gate_prepare3 = self.gate()
        inferring2, release_infer2, gate_infer2 = self.gate()
        encoding1, release_encode1, gate_encode1 = self.gate()
        contexts, encoded = [], queue.Queue()

        def prepare(image, model):
            if image == 3:
                gate_prepare3()
            return image

        def infer(tensor):
            if tensor == 2:
                gate_infer2()

        def factory(model):
            context = FakeContext(infer, limits=None)
            contexts.append(context)
            return context

        def encoder(image, depth, model, limits):
            if image == 1:
                gate_encode1()
            encoded.put((image, list(depth), limits))
            return encode(image, depth, model, limits)

        self.preparation(prepare)
        self.inference(factory)
        self.encoding(encoder)
        self.frame(1)
        self.assertTrue(encoding1.wait(2.0))
        self.frame(2)
        self.assertTrue(inferring2.wait(2.0))
        self.frame(3)
        self.assertTrue(preparing3.wait(2.0))
        # These three events prove simultaneous in-flight work without an FPS assertion.
        release_infer2.set()
        self.assertEqual(self.depths.offers.get(timeout=1.0).frame.image, 1)
        self.assertEqual(self.depths.offers.get(timeout=1.0).frame.image, 2)
        release_encode1.set()
        self.assertEqual(encoded.get(timeout=1.0), (1, [1], (0.0, 10.0)))
        self.assertEqual(encoded.get(timeout=1.0), (2, [2], (0.0, 10.0)))
        self.state(lambda: self.runtime.processing_count >= 2)
        metrics = self.runtime.result.timings
        self.assertGreaterEqual(metrics['preprocess_ms'], 0)
        self.assertEqual(metrics['inference_ms'], 52.0)
        self.assertEqual(metrics['postprocess_ms'], 10.0)
        self.assertEqual(metrics['jpeg_ms'], 8.0)
        self.assertEqual(contexts[0].references, 1)
        self.stop()
        self.assertEqual(len(set(contexts[0].owner_calls)), 1)
        self.assertTrue(contexts[0].closed.is_set())

    def test_model_switch_during_loading_closes_stale_context_before_loading_current(self):
        loading, release_load, gate_load = self.gate()
        contexts, order = [], []

        def factory(model):
            if model == MODEL:
                gate_load()
            if contexts:
                self.assertTrue(contexts[-1].closed.is_set())
            context = FakeContext()
            contexts.append(context)
            order.append(model)
            return context

        self.inference(factory)
        self.assertTrue(loading.wait(1.0))
        old_frame = self.frame('old')
        self.prepared.offer(self.prepared_job(old_frame))
        epoch = self.runtime.request_model(OTHER)
        release_load.set()
        self.state(lambda: self.runtime.active_model == OTHER)
        self.assertEqual(order, [MODEL, OTHER])
        self.assertTrue(contexts[0].closed.is_set())
        current = self.frame('new')
        self.prepared.offer(self.prepared_job(current))
        depth = self.depths.offers.get(timeout=1.0)
        self.assertEqual((depth.model, depth.epoch, depth.depth), (OTHER, epoch, ['new']))
        self.assertEqual(contexts[0].inferred, [])
        self.assertEqual(contexts[1].inferred, ['new'])

    def test_preparation_resets_sequence_and_drops_late_old_epoch(self):
        preparing, release, gate = self.gate()

        def prepare(image, model):
            if model == MODEL:
                gate()
            return (model, image)

        frame = self.frame('same capture')
        self.preparation(prepare)
        self.assertTrue(preparing.wait(1.0))
        epoch = self.runtime.request_model(OTHER)
        release.set()
        job = self.prepared.offers.get(timeout=1.0)
        self.assertIs(job.frame, frame)
        self.assertEqual((job.model, job.epoch), (OTHER, epoch))
        self.assertEqual(job.tensor, (OTHER, 'same capture'))
        self.assertTrue(self.prepared.offers.empty())

    def test_generation_change_drops_inflight_inference_and_recovers_on_new_capture(self):
        inferring, release, gate = self.gate()

        def infer(tensor):
            if tensor == 'old':
                gate()

        self.inference(lambda model: FakeContext(infer))
        self.state(lambda: self.runtime.active_model == MODEL)
        self.prepared.offer(self.prepared_job(self.frame('old')))
        self.assertTrue(inferring.wait(1.0))
        self.runtime.capture_failed('test disconnect', 1)
        recovered = self.frame('recovered')
        release.set()
        self.prepared.offer(self.prepared_job(recovered))
        job = self.depths.offers.get(timeout=1.0)
        self.assertIs(job.frame, recovered)
        self.assertEqual(job.depth, ['recovered'])
        self.assertTrue(self.depths.offers.empty())

    def test_encoder_drops_old_epoch_and_inflight_old_generation(self):
        encoding, release, gate = self.gate()
        called = []
        model, old_epoch = self.runtime.model_request()
        stale = self.frame('old epoch')
        epoch = self.runtime.request_model(OTHER)
        self.runtime.activate_model(OTHER, epoch)
        self.depths.offer(self.depth_job(stale, model, old_epoch))

        def encoder(image, depth, model, limits):
            called.append(image)
            if image == 'old generation':
                gate()
            return encode(image, depth, model, limits)

        self.encoding(encoder)
        self.assertIs(self.depths.takes.get(timeout=1.0).frame, stale)
        old = self.frame('old generation')
        self.depths.offer(self.depth_job(old))
        self.assertTrue(encoding.wait(1.0))
        self.runtime.capture_failed('test reconnect', 1)
        fresh = self.frame('current')
        self.depths.offer(self.depth_job(fresh))
        release.set()
        self.state(lambda: self.runtime.result is not None)
        self.assertEqual(called, ['old generation', 'current'])
        self.assertEqual(self.runtime.result.source_seq, fresh.seq)
        self.assertEqual(self.runtime.processing_count, 1)

    def test_load_failure_waits_for_explicit_new_epoch_then_retries(self):
        attempts, contexts = [], []

        def factory(model):
            attempts.append(model)
            if len(attempts) == 1:
                raise RuntimeError('load failed')
            context = FakeContext()
            contexts.append(context)
            return context

        self.inference(factory)
        self.state(lambda: self.runtime.model_error is not None)
        self.assertEqual(attempts, [MODEL])
        failed_epoch = self.runtime.model_epoch
        self.prepared.offer(self.prepared_job(self.frame('failed epoch')))
        epoch = self.runtime.request_model(MODEL)
        self.assertGreater(epoch, failed_epoch)
        self.state(lambda: self.runtime.active_model == MODEL)
        self.assertEqual(attempts, [MODEL, MODEL])
        fresh = self.frame('retry')
        self.prepared.offer(self.prepared_job(fresh))
        self.assertIs(self.depths.offers.get(timeout=1.0).frame, fresh)
        self.assertEqual(contexts[0].inferred, ['retry'])

    def test_encoding_failure_closes_context_and_blocks_same_epoch_inference(self):
        contexts = []

        def factory(model):
            context = FakeContext()
            contexts.append(context)
            return context

        def broken_encoder(*args):
            raise RuntimeError('JPEG failed')

        self.inference(factory)
        self.encoding(broken_encoder)
        self.state(lambda: self.runtime.active_model == MODEL)
        self.prepared.offer(self.prepared_job(self.frame('first')))
        self.state(lambda: self.runtime.model_error == 'JPEG failed')
        self.assertTrue(contexts[0].closed.wait(1.0))
        self.prepared.offer(self.prepared_job(self.frame('must not infer')))
        self.assertEqual(contexts[0].inferred, ['first'])
        self.assertIsNone(self.runtime.active_model)
        self.assertEqual(len(contexts), 1)

    def test_preparation_failure_while_loading_cannot_be_cleared_by_activation(self):
        loading, release, gate = self.gate()
        contexts = []
        constructed = threading.Event()

        def factory(model):
            gate()
            context = FakeContext()
            contexts.append(context)
            constructed.set()
            return context

        def broken_prepare(*args):
            raise RuntimeError('prepare failed')

        self.inference(factory)
        self.assertTrue(loading.wait(1.0))
        self.frame()
        self.preparation(broken_prepare)
        self.state(lambda: self.runtime.model_error == 'prepare failed')
        release.set()
        self.assertTrue(constructed.wait(1.0))
        self.assertTrue(contexts[0].closed.wait(1.0))
        self.assertIsNone(self.runtime.active_model)
        self.assertEqual(self.runtime.model_error, 'prepare failed')

    def test_stale_source_job_is_not_inferred_or_encoded(self):
        contexts, encoded = [], []

        def factory(model):
            context = FakeContext()
            contexts.append(context)
            return context

        def encoder(image, depth, model, limits):
            encoded.append(image)
            return encode(image, depth, model, limits)

        stale = self.frame('stale')
        self.clock.now += 1.1
        current = self.frame('fresh capture')
        self.prepared.offer(self.prepared_job(stale))
        self.depths.offer(self.depth_job(stale))
        self.inference(factory)
        self.encoding(encoder)
        self.state(lambda: self.runtime.active_model == MODEL)
        self.assertIs(self.prepared.takes.get(timeout=1.0).frame, stale)
        self.assertIs(self.depths.takes.get(timeout=1.0).frame, stale)
        self.prepared.offer(self.prepared_job(current))
        self.state(lambda: self.runtime.result is not None)
        self.assertEqual(contexts[0].inferred, ['fresh capture'])
        self.assertEqual(encoded, ['fresh capture'])

    def test_stop_closes_context_and_wakes_idle_stages(self):
        contexts = []

        def factory(model):
            context = FakeContext()
            contexts.append(context)
            return context

        self.preparation()
        self.inference(factory)
        self.encoding()
        self.state(lambda: self.runtime.active_model == MODEL)
        self.stop()
        self.assertTrue(contexts[0].closed.is_set())

    def test_keyed_slot_rejects_older_and_equal_keys_even_after_take(self):
        slot = LatestSlot(key=lambda value: value[0])
        self.addCleanup(slot.close)
        self.assertTrue(slot.offer(((1, 2), 'second')))
        self.assertFalse(slot.offer(((1, 1), 'first')))
        self.assertFalse(slot.offer(((1, 2), 'duplicate')))
        self.assertEqual(slot.take(0), ((1, 2), 'second'))
        self.assertFalse(slot.offer(((1, 1), 'late first')))
        self.assertFalse(slot.offer(((1, 2), 'late duplicate')))
        self.assertTrue(slot.offer(((2, 1), 'next epoch')))
        self.assertEqual(slot.take(0), ((2, 1), 'next epoch'))
        self.assertEqual(slot.overwrites, 0)
        slot.close()
        self.assertIsNone(slot.take(0))

    def test_two_contexts_overlap_but_late_old_depth_cannot_replace_newer(self):
        self.depths = ObservedSlot(key=lambda job: (job.epoch, job.frame.seq))
        older_entered, release_older, gate_older = self.gate()
        newer_entered, release_newer, gate_newer = self.gate()
        constructed = queue.Queue()

        def infer(tensor):
            (gate_older if tensor == 'older' else gate_newer)()

        def factory(model):
            context = FakeContext(infer)
            constructed.put(context)
            return context

        self.inference(factory)
        self.inference(factory)
        contexts = [constructed.get(timeout=1.0) for _ in range(2)]
        self.state(lambda: self.runtime.active_model == MODEL)
        older = self.frame('older')
        self.prepared.offer(self.prepared_job(older))
        self.assertTrue(older_entered.wait(1.0))
        newer = self.frame('newer')
        self.prepared.offer(self.prepared_job(newer))
        self.assertTrue(newer_entered.wait(1.0))
        release_newer.set()
        self.assertIs(self.depths.offers.get(timeout=1.0).frame, newer)
        release_older.set()
        self.assertIs(self.depths.rejections.get(timeout=1.0).frame, older)
        self.assertIs(self.depths.take(0).frame, newer)
        self.assertIsNone(self.depths.take(0))
        self.stop()
        self.assertEqual(len({context.owner_calls[0] for context in contexts}), 2)
        for context in contexts:
            self.assertEqual(len(set(context.owner_calls)), 1)
            self.assertTrue(context.closed.is_set())

    def test_one_context_load_failure_faults_both_and_late_activation_cannot_revive_epoch(self):
        loading, release, gate = self.gate()
        first_constructed = threading.Event()
        lock = threading.Lock()
        attempts, contexts = [], []

        def factory(model):
            with lock:
                number = len(attempts)
                attempts.append(model)
            if number == 0:
                gate()
            elif number == 1:
                raise RuntimeError('second context load failed')
            context = FakeContext()
            contexts.append(context)
            if number == 0:
                first_constructed.set()
            return context

        self.inference(factory)
        self.assertTrue(loading.wait(1.0))
        self.inference(factory)
        self.state(lambda: self.runtime.model_error == 'second context load failed')
        release.set()
        self.assertTrue(first_constructed.wait(1.0))
        self.assertTrue(contexts[0].closed.wait(1.0))
        self.assertIsNone(self.runtime.active_model)
        self.assertEqual(len(attempts), 2)
        self.assertFalse(self.depths.closed)
        self.runtime.request_model(MODEL)
        self.state(lambda: self.runtime.active_model == MODEL)
        current = self.frame('retry')
        self.prepared.offer(self.prepared_job(current))
        self.assertIs(self.depths.offers.get(timeout=1.0).frame, current)

    def test_two_contexts_switch_epoch_and_close_each_context_on_its_owner(self):
        self.depths = ObservedSlot(key=lambda job: (job.epoch, job.frame.seq))
        first_entered, release_first, gate_first = self.gate()
        second_entered, release_second, gate_second = self.gate()
        constructed = queue.Queue()
        previous = {}
        contexts = []

        def infer(tensor):
            if tensor == 'old first':
                gate_first()
            elif tensor == 'old second':
                gate_second()

        def factory(model):
            owner = threading.get_ident()
            if owner in previous:
                self.assertTrue(previous[owner].closed.is_set())
            context = FakeContext(infer)
            previous[owner] = context
            contexts.append(context)
            constructed.put(context)
            return context

        self.inference(factory)
        self.inference(factory)
        old_contexts = [constructed.get(timeout=1.0) for _ in range(2)]
        self.state(lambda: self.runtime.active_model == MODEL)
        self.prepared.offer(self.prepared_job(self.frame('old first')))
        self.assertTrue(first_entered.wait(1.0))
        self.prepared.offer(self.prepared_job(self.frame('old second')))
        self.assertTrue(second_entered.wait(1.0))
        epoch = self.runtime.request_model(OTHER)
        release_first.set()
        release_second.set()
        new_contexts = [constructed.get(timeout=1.0) for _ in range(2)]
        self.state(lambda: self.runtime.active_model == OTHER)
        for context in old_contexts:
            self.assertTrue(context.closed.is_set())
        current = self.frame('current')
        self.prepared.offer(self.prepared_job(current))
        job = self.depths.offers.get(timeout=1.0)
        self.assertEqual((job.model, job.epoch, job.frame), (OTHER, epoch, current))
        self.assertTrue(self.depths.offers.empty())
        self.stop()
        self.assertEqual(len(contexts), 4)
        for context in old_contexts + new_contexts:
            self.assertEqual(len(set(context.owner_calls)), 1)
            self.assertTrue(context.closed.is_set())

    def test_stopping_one_inference_worker_keeps_shared_depth_slot_open(self):
        local_stop = threading.Event()
        constructed = queue.Queue()

        def factory(model):
            context = FakeContext()
            constructed.put(context)
            return context

        first = self.start(inference_worker, self.runtime, local_stop,
                           self.prepared, self.depths, factory)
        first_owner = self.threads[-1][0].ident
        self.inference(factory)
        contexts = [constructed.get(timeout=1.0) for _ in range(2)]
        self.state(lambda: self.runtime.active_model == MODEL)
        local_stop.set()
        with self.runtime.condition:
            self.runtime.condition.notify_all()
        first.result(timeout=1.0)
        closed = next(context for context in contexts if context.owner_calls[0] == first_owner)
        other = next(context for context in contexts if context is not closed)
        self.assertTrue(closed.closed.is_set())
        self.assertFalse(other.closed.is_set())
        self.assertFalse(self.depths.closed)
        current = self.frame('remaining worker')
        self.prepared.offer(self.prepared_job(current))
        self.assertIs(self.depths.offers.get(timeout=1.0).frame, current)
        self.assertEqual(other.inferred, ['remaining worker'])


if __name__ == '__main__':
    unittest.main()
