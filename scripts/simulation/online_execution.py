"""Executor-owned finite feedback while a separate planning process runs.

This optional driver accepts only products already proved for their declared
future activation clock. It never calls a nonlinear or geometry proof service
on the command path. A device/plant adapter owns literal actuation, sensing and
the physical clock; the maintained nonlinear mission runner remains offline.
Dispatch/acknowledgment overruns are reported as assumption violations, not
repaired by pausing physical time or by emitting an unproved zero command.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import time

from orbexa.control.fleet_fallback import FleetAdmissionRuntime, PerformanceAttempt
from orbexa.control.prepared_admission import (PreparedFleetReplacement,
    AdmissionContextChanged, activate_fleet_replacement)
from orbexa.control.reachability import _exact
from orbexa.control.refuge import fingerprint


@dataclass(frozen=True)
class OnlineHoldWindow:
    start: Fraction
    hold_width: Fraction
    release_ns: int
    dispatch_deadline_ns: int
    acknowledgement_deadline_ns: int

    def __post_init__(self):
        object.__setattr__(self, 'start', _exact(self.start))
        object.__setattr__(self, 'hold_width', _exact(self.hold_width))
        values = (self.release_ns, self.dispatch_deadline_ns, self.acknowledgement_deadline_ns)
        if (self.hold_width <= 0 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values)
                or not values[0] <= values[1] <= values[2]):
            raise ValueError('Declare a finite exact hold and ordered monotonic wall-clock deadlines')


@dataclass(frozen=True)
class AppliedHoldReceipt:
    decision_id: str
    exact_end_q: Fraction
    endpoint_observations: dict
    dispatched_ns: int
    acknowledged_ns: int
    physical_adapter_id: str

    def __post_init__(self):
        object.__setattr__(self, 'exact_end_q', _exact(self.exact_end_q))
        if (not self.decision_id or not self.physical_adapter_id or
                not isinstance(self.endpoint_observations, dict) or
                self.dispatched_ns <= 0 or self.acknowledged_ns < self.dispatched_ns):
            raise ValueError('An exact physical hold acknowledgment and adapter identity are required')


class OnlineExecutionFailure(RuntimeError):
    def __init__(self, message, *, record):
        super().__init__(message)
        self.record = record


def execution_context_id(runtime, *, source_id):
    """Context shared with the detached worker, excluding advancing tank/q."""
    if not source_id:
        raise ValueError('Bind online context to an exact source/configuration identity')
    return fingerprint(dict(source_id=source_id,
        active_fallback_id=runtime.active_fallback.fingerprint,
        acceleration_disturbance=runtime.acceleration_disturbance))


class OnlineFiniteExecutor:
    """Single owner of runtime/actuation; one buffered future product at most."""
    def __init__(self, runtime, *, worker, source_id, observe, apply_hold, event_sink=None,
                 maximum_retained_commits=1):
        if not isinstance(runtime, FleetAdmissionRuntime) or runtime._pending is not None:
            raise TypeError('Install a complete idle finite fallback before starting online execution')
        if any(not callable(value) for value in (observe, apply_hold)):
            raise TypeError('Supply independent observation and literal actuation adapters')
        if event_sink is not None and not callable(event_sink):
            raise TypeError('event_sink must be callable')
        if (isinstance(maximum_retained_commits, bool) or not isinstance(maximum_retained_commits, int)
                or maximum_retained_commits < 0):
            raise ValueError('Declare a nonnegative bounded retained-commit count')
        self.runtime, self.worker, self.source_id = runtime, worker, source_id
        self.observe, self.apply_hold, self.event_sink = observe, apply_hold, event_sink
        self.maximum_retained_commits = maximum_retained_commits
        self._ready, self._failed, self.sequence = None, None, 0

    def _emit(self, record):
        if self.event_sink is not None:
            self.event_sink(record)

    def _failure(self, reason, record):
        record.update(status='ASSUMPTION_VIOLATION', reason=reason,
            continuation_authorized=False,
            scope='No clock pause or unproved emergency input is inferred')
        self._failed = dict(record)
        try:
            self._emit(record)
        except Exception as error:
            record['failure_event_persistence'] = type(error).__name__+': '+str(error)
        raise OnlineExecutionFailure(reason, record=record)

    def _candidate(self, window, record):
        try:
            result = self.worker.poll()
        except Exception as error:
            record['worker_refusal'] = type(error).__name__+': '+str(error)
            return None, 'WORKER_RESULT_REFUSED'
        if result is not None:
            record['worker_result'] = result.as_dict()
            if self._ready is not None:
                record['replaced_ready_product'] = self._ready.request.fingerprint
            self._ready = result
        result = self._ready
        if result is None:
            return None, 'PLANNER_PENDING' if self.worker.busy else 'NO_READY_PROPOSAL'
        if result.status != 'READY':
            self._ready = None
            return None, result.status
        activation = _exact(result.request.activation_q)
        if activation > window.start:
            return None, 'PRODUCT_FOR_FUTURE_CLOCK'
        self._ready = None
        if activation < window.start:
            return None, 'STALE_PROPOSAL'
        if (_exact(result.request.hold_width) != window.hold_width or
                result.request.context_id != execution_context_id(self.runtime, source_id=self.source_id)):
            return None, 'CONTEXT_OR_SCHEDULE_CHANGED'
        now = time.monotonic_ns()
        if (result.completed_ns > result.request.deadline_ns or result.received_ns is None
                or result.received_ns > result.request.deadline_ns
                or now > min(result.request.deadline_ns, window.dispatch_deadline_ns)):
            return None, 'LATE_PROPOSAL'
        try:
            candidate = result.decode()
            if not isinstance(candidate, PreparedFleetReplacement):
                return None, 'UNPREPARED_PRODUCT_REFUSED'
            if candidate.start != activation or candidate.hold_width != _exact(result.request.hold_width):
                return None, 'PRODUCT_REQUEST_CLOCK_CHANGED'
            record['prepared_product_id'] = candidate.fingerprint
        except Exception as error:
            record['product_refusal'] = type(error).__name__+': '+str(error)
            return None, 'INVALID_PREPARED_PRODUCT'
        if time.monotonic_ns() > min(result.request.deadline_ns, window.dispatch_deadline_ns):
            return None, 'LATE_PROPOSAL'
        record['proposal_ready_deadline_ns'] = result.request.deadline_ns
        return candidate, 'READY'

    def step(self, window):
        """One scheduled hold; planning is never joined or awaited here.

        The caller supplies physical wall-clock windows. Their tolerance must
        be supported by the adapter's sensing/actuation model; this class does
        not turn a Python dispatch budget into a new physical theorem.
        """
        if self._failed is not None:
            raise OnlineExecutionFailure('A failed execution requires an explicit recovery contract', record=self._failed)
        if not isinstance(window, OnlineHoldWindow):
            raise TypeError('Use an explicit exact physical hold window')
        if time.monotonic_ns() < window.release_ns:
            raise ValueError('Call the executor when the physical window is released')
        record = dict(schema='orbexa.online-execution-event.v1', sequence=self.sequence,
            exact_start=str(window.start), exact_hold_width=str(window.hold_width),
            release_ns=window.release_ns, dispatch_deadline_ns=window.dispatch_deadline_ns,
            acknowledgement_deadline_ns=window.acknowledgement_deadline_ns,
            source_id=self.source_id, worker_independent=True)
        self.sequence += 1
        try:
            if window.start != self.runtime.q:
                raise ValueError('Physical release and authoritative exact runtime clock disagree')
            record['finite_scope_end_q'] = str(self.runtime.active_fallback.certificate.end)
            observations = self.observe(window.start)
            candidate, reason = self._candidate(window, record)
            record['proposal_status'] = reason
            decision = None
            if candidate is not None:
                old_plan = self.runtime.active_fallback
                try:
                    record['feedback_replacement'] = activate_fleet_replacement(
                        self.runtime, candidate, observations, q=window.start, hold_width=window.hold_width)
                    reason = 'PREPARED_FEEDBACK_POLICY_ACTIVATED'
                    if time.monotonic_ns() > record['proposal_ready_deadline_ns']:
                        # No motion or debit has occurred. A slow final exact
                        # gate cannot turn a late product into an active policy.
                        self.runtime.active_fallback = old_plan
                        self.runtime._pending = None
                        decision = None
                        record.pop('feedback_replacement', None)
                        reason = 'LATE_PROPOSAL'
                except AdmissionContextChanged as error:
                    self._failure(str(error), record)
                except Exception as error:
                    record['proposal_refusal'] = type(error).__name__+': '+str(error)
                    reason = 'PREPARED_ACTIVATION_REFUSED'
            record['proposal_status'] = reason
            self.runtime.active_fallback.segment_at(window.start, window.hold_width)
            if decision is None:
                decision = self.runtime.decide(observations, q=window.start, hold_width=window.hold_width,
                    failed_attempt=PerformanceAttempt(reason, 'Independent executor retained the installed finite feedback',
                        solver_id='asynchronous-source-bound-worker'), refine_fallback_resources=False)
            record['decision_id'], record['command_source'] = decision.fingerprint, decision.source
            record['decision'] = decision.as_dict()
            self._emit(dict(record, status='ADMITTED_BEFORE_MOTION'))
            record['decision_ready_ns'] = time.monotonic_ns()
            if record['decision_ready_ns'] > window.dispatch_deadline_ns:
                self._failure('Executor did not produce a command within the declared dispatch budget', record)
            # The adapter must acknowledge actual dispatch/end observations.
            # It may not advance a simulated clock and call that wall timing.
            receipt = self.apply_hold(decision, window)
            if (not isinstance(receipt, AppliedHoldReceipt) or receipt.decision_id != decision.fingerprint or
                    receipt.exact_end_q != window.start+window.hold_width or
                    receipt.dispatched_ns < record['decision_ready_ns'] or
                    receipt.acknowledged_ns > time.monotonic_ns()):
                self._failure('Actuation adapter did not acknowledge the exact admitted hold', record)
            record.update(dispatched_ns=receipt.dispatched_ns, acknowledged_ns=receipt.acknowledged_ns,
                physical_adapter_id=receipt.physical_adapter_id,
                dispatch_timely=receipt.dispatched_ns <= window.dispatch_deadline_ns,
                acknowledgement_timely=receipt.acknowledged_ns <= window.acknowledgement_deadline_ns)
            committed = self.runtime.commit(decision, receipt.endpoint_observations, q=receipt.exact_end_q)
            record['commit'] = committed
            if not record['dispatch_timely'] or not record['acknowledgement_timely']:
                self._failure('A physically acknowledged hold exceeded its declared timing contract', record)
            record['status'] = committed['status']
            self._emit(record)
            if len(self.runtime.events) > self.maximum_retained_commits:
                del self.runtime.events[:len(self.runtime.events)-self.maximum_retained_commits]
            return record
        except OnlineExecutionFailure:
            raise
        except Exception as error:
            self._failure(type(error).__name__+': '+str(error), record)
