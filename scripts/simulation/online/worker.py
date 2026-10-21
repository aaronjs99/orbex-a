"""One bounded, spawned planning process; polling never waits for its work.

Requests contain immutable bytes and explicit activation/context identities.
The child receives no executor, actuator or authoritative runtime reference.
Products come only from trusted local code; pickle is not a network protocol.
Only one job and one bounded result buffer exist at a time. Process isolation
does not establish a hard real-time schedule on a general-purpose OS.
The worker hash covers its handler module only. Complete package/configuration
provenance requires the caller's context identity and immutable source snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib
import io
import json
import multiprocessing
from pathlib import Path
import pickle
import time
from types import MappingProxyType
from orbexa.simulation.online.clock import MONOTONIC_CLOCK, wall_time_ns


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(name+' must be a positive integer')
    return value


def _mapping_proxy(value):
    return MappingProxyType(value)


def serialize_product(value):
    """Trusted local pickle preserving immutable metadata without global hooks.

    Nonserializable arbitrary closures still refuse explicitly. Fixed proof
    sources have an ordinary importable callable class instead of a closure.
    """
    import copyreg
    stream = io.BytesIO()
    pickler = pickle.Pickler(stream, protocol=5)
    pickler.dispatch_table = copyreg.dispatch_table.copy()
    pickler.dispatch_table[MappingProxyType] = lambda item: (_mapping_proxy, (dict(item),))
    pickler.dump(value)
    return stream.getvalue()


def _handler(spec):
    module_name, separator, name = spec.partition(':')
    if not separator or not module_name or not name or '.' in name:
        raise ValueError('Use an importable module:top_level_function worker')
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    if not callable(value) or not getattr(module, '__file__', None):
        raise TypeError('Worker must be a source-backed callable')
    return value, _digest(Path(module.__file__).read_bytes())


@dataclass(frozen=True)
class PlanningRequest:
    request_id: str
    context_id: str
    activation_q: str
    hold_width: str
    submitted_ns: int
    deadline_ns: int
    payload: bytes
    wall_clock_id: str = MONOTONIC_CLOCK

    def __post_init__(self):
        from fractions import Fraction
        if any(not isinstance(value, str) or not value.strip() for value in
               (self.request_id, self.context_id, self.activation_q, self.hold_width)):
            raise ValueError('Planning requests require explicit identities and exact clock strings')
        if Fraction(self.hold_width) <= 0:
            raise ValueError('A planning request needs a positive exact hold')
        Fraction(self.activation_q)
        _positive_int(self.submitted_ns, 'Submission monotonic time')
        _positive_int(self.deadline_ns, 'Activation deadline')
        wall_time_ns(self.wall_clock_id)
        if self.deadline_ns < self.submitted_ns or not isinstance(self.payload, bytes):
            raise ValueError('Use an immutable payload and a nonpast declared deadline')

    @property
    def fingerprint(self):
        return _digest(json.dumps(dict(request_id=self.request_id, context_id=self.context_id,
            activation_q=self.activation_q, hold_width=self.hold_width,
            submitted_ns=self.submitted_ns, deadline_ns=self.deadline_ns,
            wall_clock_id=self.wall_clock_id,
            payload_sha256=_digest(self.payload)), sort_keys=True, separators=(',', ':')).encode())


@dataclass(frozen=True)
class PlanningResult:
    request: PlanningRequest
    worker_spec: str
    worker_source_sha256: str
    status: str
    started_ns: int
    completed_ns: int
    payload: bytes
    reason: str
    process_id: int
    received_ns: int | None = None

    def decode(self):
        """Decode trusted local output; its cost belongs in executor timing."""
        if self.status != 'READY':
            raise ValueError('No ready product: '+self.reason)
        return pickle.loads(self.payload)

    def as_dict(self):
        return dict(schema='orbexa.online-planning-result.v1',
            request_id=self.request.request_id, request_identity=self.request.fingerprint,
            context_id=self.request.context_id, activation_q=self.request.activation_q,
            hold_width=self.request.hold_width, status=self.status,
            worker_spec=self.worker_spec, worker_source_sha256=self.worker_source_sha256,
            started_ns=self.started_ns, completed_ns=self.completed_ns,
            received_ns=self.received_ns,
            deadline_ns=self.request.deadline_ns,
            wall_clock_id=self.request.wall_clock_id,
            ready_by_declared_deadline=(self.status == 'READY' and self.received_ns is not None
                and self.completed_ns <= self.request.deadline_ns and self.received_ns <= self.request.deadline_ns),
            timing_scope='Completed product serialization in worker; full envelope received/decoded by parent separately',
            product_sha256=_digest(self.payload), product_bytes=len(self.payload),
            reason=self.reason, process_id=self.process_id)


def _execute(spec, source_id, request, buffer, length, completed, maximum_bytes):
    """Child-only entry point. Publish the completion flag after all bytes."""
    import os
    started = wall_time_ns(request.wall_clock_id)
    try:
        worker, actual_id = _handler(spec)
        if actual_id != source_id:
            raise ValueError('Worker source changed after request creation')
        product = worker(request)
        payload = serialize_product(product)
        result = PlanningResult(request, spec, source_id, 'READY', started,
            wall_time_ns(request.wall_clock_id), payload, '', os.getpid())
    except Exception as error:
        result = PlanningResult(request, spec, source_id, 'REFUSED', started,
            wall_time_ns(request.wall_clock_id), b'', type(error).__name__+': '+str(error), os.getpid())
    wire = pickle.dumps(result, protocol=5)
    if len(wire) > maximum_bytes:
        result = PlanningResult(request, spec, source_id, 'RESULT_TOO_LARGE', started,
            wall_time_ns(request.wall_clock_id), b'', 'Prepared result exceeds the declared bounded buffer', os.getpid())
        wire = pickle.dumps(result, protocol=5)
    if len(wire) <= maximum_bytes:
        # A ctypes slice materializes one Python integer per byte.  A bounded
        # byte view preserves the exact wire format without that allocation
        # and interpreter loop on either side of the feedback boundary.
        memoryview(buffer).cast('B')[:len(wire)] = wire
        length.value = len(wire)
    completed.set()


class ProcessPlanningWorker:
    """A single nonblocking producer with no access to actuation state."""
    def __init__(self, worker_spec, *, maximum_result_bytes=32*1024*1024,
                 maximum_request_bytes=4*1024*1024):
        _positive_int(maximum_result_bytes, 'Result buffer size')
        _positive_int(maximum_request_bytes, 'Request size limit')
        if maximum_result_bytes < 4096:
            raise ValueError('Reserve at least 4096 bytes for a refusal envelope')
        _, self.source_id = _handler(worker_spec)
        self.worker_spec = worker_spec
        self.maximum_result_bytes = maximum_result_bytes
        self.maximum_request_bytes = maximum_request_bytes
        self._context = multiprocessing.get_context('spawn')
        self._process = self._request = None
        self._buffer = self._length = self._completed = None
        self._retired = []

    def _reap(self):
        remaining = []
        for process in self._retired:
            if process.is_alive():
                remaining.append(process)
            else:
                process.join(timeout=0)
                process.close()
        self._retired = remaining

    @property
    def busy(self):
        return self._process is not None

    def submit(self, request):
        self._reap()
        if self._retired:
            raise ValueError('The previous completed worker is still tearing down; retry after it exits')
        if self.busy:
            raise ValueError('Consume or explicitly cancel the single previous planning request')
        if not isinstance(request, PlanningRequest) or len(request.payload) > self.maximum_request_bytes:
            raise ValueError('Use a bounded immutable typed planning request')
        if wall_time_ns(request.wall_clock_id) > request.deadline_ns:
            raise ValueError('The planning request deadline already passed')
        self._buffer = self._context.RawArray('B', self.maximum_result_bytes)
        self._length = self._context.Value('Q', 0, lock=False)
        self._completed = self._context.Event()
        self._request = request
        self._process = self._context.Process(target=_execute,
            args=(self.worker_spec, self.source_id, request, self._buffer,
                  self._length, self._completed, self.maximum_result_bytes), daemon=True)
        try:
            self._process.start()
        except BaseException:
            process = self._process
            if process.pid is not None:
                # A partially started child is still owned. Retain at most
                # this one until explicit reap/shutdown, without joining it.
                self._retired.append(process)
            else:
                try:
                    process.close()
                except Exception:
                    pass
            self._process = self._request = None
            self._buffer = self._length = self._completed = None
            raise

    def poll(self):
        """Return a completed result or None; never join an active planner."""
        if not self.busy:
            return None
        if not self._completed.is_set() and self._process.is_alive():
            return None
        try:
            if not self._completed.is_set():
                result = PlanningResult(self._request, self.worker_spec, self.source_id,
                    'WORKER_EXITED', 0, wall_time_ns(self._request.wall_clock_id), b'',
                    'Worker exited without publishing a complete bounded result', self._process.pid)
            elif not 0 < self._length.value <= self.maximum_result_bytes:
                result = PlanningResult(self._request, self.worker_spec, self.source_id,
                    'INVALID_RESULT', 0, wall_time_ns(self._request.wall_clock_id), b'',
                    'Worker did not publish a complete result envelope', self._process.pid)
            else:
                result = pickle.loads(memoryview(self._buffer).cast('B')[:self._length.value].tobytes())
                if (not isinstance(result, PlanningResult) or result.request != self._request
                        or result.worker_spec != self.worker_spec or result.worker_source_sha256 != self.source_id):
                    raise ValueError('Worker result changed its immutable request or source identity')
            return replace(result, received_ns=wall_time_ns(self._request.wall_clock_id))
        finally:
            # A completed flag precedes final process teardown. Do not wait for
            # it here. Even malformed bytes must release the sole active slot.
            process = self._process
            if not process.is_alive():
                process.join(timeout=0)
                process.close()
            else:
                self._retired.append(process)
            self._process = self._request = None
            self._buffer = self._length = self._completed = None

    def close(self):
        """Explicit shutdown outside the feedback loop; never affect other runs."""
        processes = list(getattr(self, '_retired', ()))
        if self._process is not None:
            processes.append(self._process)
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=.5)
            if process.is_alive():
                process.kill(); process.join(timeout=.5)
            if not process.is_alive():
                process.close()
        self._retired = []
        self._process = self._request = None
        self._buffer = self._length = self._completed = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
