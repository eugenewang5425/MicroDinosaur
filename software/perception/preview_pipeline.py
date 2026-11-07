"""Bounded three-stage preview workers, with no camera or numerical imports.

Each boundary holds only its newest pending job. Each inference worker owns
its context, including creation, calibration and release. The caller closes
the shared depth slot when stopping all workers.
"""
from copy import deepcopy
from dataclasses import dataclass
import threading
import time


class LatestSlot:
    """One pending value, optionally rejecting keys older than any accepted key.

    The key watermark survives take(), preventing an out-of-order producer
    from putting an older result back into an empty slot.
    """

    def __init__(self, key=None):
        self._condition = threading.Condition()
        self._value = None
        self._closed = False
        self._key = key
        self._latest_key = None
        self._has_key = False
        self.overwrites = 0

    @property
    def closed(self):
        with self._condition:
            return self._closed

    def offer(self, value):
        if value is None:
            raise ValueError('None is reserved for an empty slot')
        with self._condition:
            if self._closed:
                return False
            if self._key is not None:
                value_key = self._key(value)
                if self._has_key and value_key <= self._latest_key:
                    return False
                self._latest_key = value_key
                self._has_key = True
            if self._value is not None:
                self.overwrites += 1
            self._value = value
            self._condition.notify()
            return True

    def take(self, timeout=.2):
        deadline = time.monotonic() + max(0, timeout)
        with self._condition:
            while self._value is None and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            value = self._value
            self._value = None
            return value

    def close(self):
        with self._condition:
            self._closed = True
            self._value = None
            self._latest_key = None
            self._has_key = False
            self._condition.notify_all()


@dataclass(frozen=True)
class PreparedJob:
    frame: object
    model: str
    epoch: int
    tensor: object
    preprocess_ms: float


@dataclass(frozen=True)
class DepthJob:
    frame: object
    model: str
    epoch: int
    depth: object
    limits: tuple
    timings: dict


def _stopped(runtime, stopping):
    with runtime.condition:
        return stopping.is_set() or runtime.stopped


def _healthy_request(runtime, model, epoch, *, active=False):
    with runtime.condition:
        return (not runtime.stopped and runtime.requested_model == model
                and runtime.model_epoch == epoch and runtime.model_error is None
                and (not active or runtime.active_model == model))


def _wait_state(runtime, stopping):
    # The bounded wait also handles callers that only set their stop Event.
    with runtime.condition:
        if not stopping.is_set() and not runtime.stopped:
            runtime.condition.wait(.2)


def _close_context(context, runtime, epoch):
    if context is None:
        return True
    try:
        context.close()
        return True
    except Exception as error:
        runtime.model_failed(error, epoch)
        return False


def prepare_worker(runtime, stopping, prepared_slot, prepare_fn):
    """Preprocess fresh captures while inference handles a previous capture."""
    request = None
    after_seq = 0
    try:
        while not _stopped(runtime, stopping) and not prepared_slot.closed:
            model, epoch = runtime.model_request()
            if request != (model, epoch):
                request = model, epoch
                after_seq = 0
            if not _healthy_request(runtime, model, epoch):
                _wait_state(runtime, stopping)
                continue
            frame = runtime.wait_for_frame(after_seq, epoch, timeout=.2)
            if frame is None:
                continue
            after_seq = frame.seq
            if (_stopped(runtime, stopping)
                    or not runtime.accepts_frame(frame, model, epoch, active=False)):
                continue
            try:
                started = time.perf_counter()
                tensor = prepare_fn(frame.image, model)
                elapsed = (time.perf_counter() - started) * 1000
                if (not _stopped(runtime, stopping)
                        and _healthy_request(runtime, model, epoch)
                        and runtime.accepts_frame(frame, model, epoch, active=False)):
                    prepared_slot.offer(PreparedJob(frame, model, epoch, tensor, elapsed))
            except Exception as error:
                runtime.model_failed(error, epoch)
    finally:
        prepared_slot.close()


def inference_worker(runtime, stopping, prepared_slot, depth_slot, context_factory):
    """Own one context; multiple workers may consume and produce shared slots.

    A failure faults the entire epoch. Only a new model request permits retry.
    The caller owns depth_slot.close(), because other workers may still run.
    """
    attempted = None
    context = None
    try:
        while not _stopped(runtime, stopping) and not prepared_slot.closed and not depth_slot.closed:
            model, epoch = runtime.model_request()
            if attempted != (model, epoch):
                old_context, context = context, None
                attempted = model, epoch
                if not _close_context(old_context, runtime, epoch):
                    continue
                if not _healthy_request(runtime, model, epoch):
                    continue
                candidate = None
                try:
                    candidate = context_factory(model)
                    # Keep the error check and activation atomic: a preprocessing
                    # failure during loading must not be cleared by activation.
                    with runtime.condition:
                        activated = (not stopping.is_set()
                                     and _healthy_request(runtime, model, epoch)
                                     and runtime.activate_model(model, epoch))
                    if activated:
                        context, candidate = candidate, None
                except Exception as error:
                    runtime.model_failed(error, epoch)
                finally:
                    _close_context(candidate, runtime, epoch)
            if context is None:
                _wait_state(runtime, stopping)
                continue
            if not _healthy_request(runtime, model, epoch, active=True):
                old_context, context = context, None
                _close_context(old_context, runtime, epoch)
                continue
            job = prepared_slot.take()
            if job is None:
                continue
            if ((job.model, job.epoch) != attempted
                    or _stopped(runtime, stopping)
                    or not runtime.accepts_frame(job.frame, job.model, job.epoch, active=True)):
                continue
            try:
                depth, inference_ms = context.infer_tensor(job.tensor)
                if (_stopped(runtime, stopping)
                        or not runtime.accepts_frame(job.frame, job.model, job.epoch, active=True)):
                    continue
                # RKNN may reuse an output allocation on its next invocation.
                depth = depth.copy() if hasattr(depth, 'copy') else deepcopy(depth)
                limits = context.limits
                if limits is None:
                    limits = context.reference_limits(depth)
                timings = {'preprocess_ms': job.preprocess_ms, 'inference_ms': inference_ms}
                if (not _stopped(runtime, stopping)
                        and runtime.accepts_frame(job.frame, job.model, job.epoch, active=True)):
                    depth_slot.offer(DepthJob(job.frame, job.model, job.epoch, depth,
                                              tuple(limits), timings))
            except Exception as error:
                runtime.model_failed(error, epoch)
                old_context, context = context, None
                _close_context(old_context, runtime, epoch)
    finally:
        _close_context(context, runtime, attempted[1] if attempted is not None else -1)


def encode_worker(runtime, stopping, depth_slot, encode_fn):
    """Encode only current depths and let runtime atomically accept the result."""
    while not _stopped(runtime, stopping) and not depth_slot.closed:
        job = depth_slot.take()
        if (job is None or _stopped(runtime, stopping)
                or not runtime.accepts_frame(job.frame, job.model, job.epoch, active=True)):
            continue
        try:
            jpeg, encoding_timings = encode_fn(job.frame.image, job.depth, job.model, job.limits)
            timings = dict(job.timings)
            timings.update(encoding_timings)
            if (not _stopped(runtime, stopping)
                    and runtime.accepts_frame(job.frame, job.model, job.epoch, active=True)):
                runtime.publish_result(job.frame, jpeg, job.model, job.epoch, timings)
        except Exception as error:
            runtime.model_failed(error, job.epoch)
