"""Admission and execution of a validated held-input backup continuation.

A successful local optimizer is only a proposed performance command. A command
is admitted when its continuous swept flow is safe and its successor belongs to
the next already certified backup entry set. Solver failure selects the stored
backup command. No arbitrary zero input, emergency braking assumption or frozen
simulation state is used. Existence of such a backup is an admission condition,
not a universal feasibility claim.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Callable, Sequence

from orbexa.control.certification.reachability import CollisionCertificate, FlowCertificate, HeldInputRequest, IntervalBox, OrbitalModelBounds, ValidatedFlowEncloser
from orbexa.control.terminal.tracking import TerminalCertificate, TerminalReferenceSample, TerminalFeedforwardPolicy, TerminalTrackingController, _q, certify_terminal_tracking


CONTINUATION_METHOD = "validated-backup-successor-admission-v1"


def _intersection(left: IntervalBox, right: IntervalBox) -> IntervalBox:
    if left.dimension != right.dimension:
        raise ValueError("Observation and predicted-state dimensions must agree")
    lower = tuple(map(max, left.lower, right.lower))
    upper = tuple(map(min, left.upper, right.upper))
    if any(a > b for a, b in zip(lower, upper)):
        raise ValueError("Observation is inconsistent with the certified flow and sensing assumptions")
    return IntervalBox(lower, upper)


def _valid_flow(flow, request, frame):
    return (isinstance(flow, FlowCertificate) and flow.proven
            and flow.endpoint is not None and flow.swept_state is not None
            and flow.initial_state == request.initial_state
            and flow.start_anomaly == request.start_anomaly
            and flow.hold_width == request.hold_width and flow.frame == frame
            and flow.request == request.as_dict())


def _binary_command(box: IntervalBox) -> tuple[float, ...]:
    """Select an executable midpoint, refusing any uncovered conversion error."""
    values = tuple(float((lo + hi) / 2) for lo, hi in zip(box.lower, box.upper))
    if not box.contains(IntervalBox.point(values)):
        raise ValueError("Binary64 command conversion is outside the proved held-input box")
    return values


@dataclass(frozen=True)
class HeldControl:
    """One scheduled input box and an optional larger backup entry box.

    Input-box width must cover every possible held actuator value, including
    command conversion error. The physical hold must have the exact declared
    anomaly duration. A delayed/jittered execution needs a separately certified
    uncertain-clock continuation, and is not silently accepted by this API.
    """

    hold_width: Any
    held_input: IntervalBox
    entry_box: IntervalBox | None = None

    def __post_init__(self):
        width = _q(self.hold_width)
        if width <= 0 or self.held_input.dimension != 3 or (
                self.entry_box is not None and self.entry_box.dimension != 6):
            raise ValueError("A positive hold, 3-input box and optional 6-state entry box are required")
        object.__setattr__(self, "hold_width", width)

    def as_dict(self):
        return {"hold_width": str(self.hold_width), "held_input": self.held_input.as_dict(),
                "entry_box": self.entry_box.as_dict() if self.entry_box else None}


@dataclass(frozen=True)
class BackupStage:
    index: int
    start_anomaly: Fraction
    control: HeldControl
    entry_box: IntervalBox
    flow: FlowCertificate
    geometry: CollisionCertificate

    @property
    def end_anomaly(self):
        return self.start_anomaly + self.control.hold_width

    def as_dict(self, *, include_segments=True):
        return {"index": self.index, "start_anomaly": str(self.start_anomaly),
                "end_anomaly": str(self.end_anomaly), "control": self.control.as_dict(),
                "entry_box": self.entry_box.as_dict(),
                "flow": self.flow.as_dict(include_segments=include_segments),
                "geometry": self.geometry.as_dict()}


@dataclass(frozen=True)
class ContinuationCertificate:
    admissible: bool
    reasons: tuple[str, ...]
    stages: tuple[BackupStage, ...]
    terminal: TerminalCertificate
    terminal_reference: TerminalReferenceSample
    method: str = CONTINUATION_METHOD

    def as_dict(self, *, include_segments=True):
        return {"method": self.method, "admissible": self.admissible,
                "status": "CONDITIONAL_PROVEN" if self.admissible else "REFUSED",
                "reasons": self.reasons,
                "stages": tuple(stage.as_dict(include_segments=include_segments) for stage in self.stages),
                "terminal": self.terminal.as_dict(),
                "terminal_reference": self.terminal_reference.as_dict(),
                "claim": "Every admitted input has a safe continuous flow and an already admissible successor",
                "serialization_policy": "Evidence only: reconstruct and recertify before execution"}


@dataclass
class OrbitalHeldPlant:
    parameters: OrbitalModelBounds
    encloser: ValidatedFlowEncloser

    def __call__(self, request: HeldInputRequest) -> FlowCertificate:
        return self.encloser.enclose_orbital(request, self.parameters)


@dataclass
class AffineHeldPlant:
    """Six-state affine plant with control in the three acceleration rows.

    The matrix is constant and exact; the affine box may bound arbitrary
    measurable model disturbance. Feedback is evaluated outside this oracle
    and held in ``request.held_input``. An A+BK continuous-feedback substitution
    would certify a different controller and is deliberately not performed.
    """

    matrix: Sequence[Sequence[Any]]
    affine: IntervalBox
    encloser: ValidatedFlowEncloser

    def __post_init__(self):
        self.matrix = tuple(tuple(map(_q, row)) for row in self.matrix)
        if len(self.matrix) != 6 or any(len(row) != 6 for row in self.matrix) or self.affine.dimension != 6:
            raise ValueError("A six-state affine plant needs a (6,6) matrix and 6-affine box")

    def __call__(self, request: HeldInputRequest) -> FlowCertificate:
        lo, hi = list(self.affine.lower), list(self.affine.upper)
        for i in range(3):
            lo[i + 3] += request.held_input.lower[i] + request.acceleration_disturbance.lower[i]
            hi[i + 3] += request.held_input.upper[i] + request.acceleration_disturbance.upper[i]
        flow = self.encloser.enclose_affine(self.matrix, request.initial_state, IntervalBox(tuple(lo), tuple(hi)),
                                           start_anomaly=request.start_anomaly, hold_width=request.hold_width)
        return replace(flow, request=request.as_dict())


class ContinuationBuilder:
    """Validate a complete backup chain before admitting its initial set.

    ``flow_provider`` and ``geometry_checker`` are trusted proof-producing
    dependencies. The latter must enclose all occupied objects and peers for
    each entire hold, with the same frame and clock as the supplied flow.
    Their successful result is not replaced by endpoint sampling here.
    """

    def __init__(self, flow_provider: Callable[[HeldInputRequest], FlowCertificate],
                 geometry_checker: Callable[[int, FlowCertificate], CollisionCertificate], *,
                 input_limits: IntervalBox, terminal: TerminalCertificate,
                 acceleration_disturbance: IntervalBox | None = None):
        if not callable(flow_provider) or not callable(geometry_checker) or input_limits.dimension != 3:
            raise ValueError("Explicit flow/geometry proof providers and 3-input limits are required")
        if (not isinstance(terminal, TerminalCertificate)
                or terminal.policy is not TerminalFeedforwardPolicy.START_V1
                or terminal.method != TerminalFeedforwardPolicy.START_V1.certificate_method):
            raise ValueError("This v1 continuation format supports only the explicit start-feedforward terminal policy")
        self.flow_provider, self.geometry_checker = flow_provider, geometry_checker
        self.input_limits = input_limits
        self.terminal = certify_terminal_tracking(terminal.bounds)
        self.disturbance = acceleration_disturbance or IntervalBox.point((0, 0, 0))
        if self.disturbance.dimension != 3:
            raise ValueError("Acceleration disturbance has dimension 3")

    def certify(self, initial_state: IntervalBox, start_anomaly,
                controls: Sequence[HeldControl], terminal_reference: TerminalReferenceSample) -> ContinuationCertificate:
        q, current, stages, reasons = _q(start_anomaly), initial_state, [], []
        if initial_state.dimension != 6 or not controls:
            raise ValueError("A six-state initial box and a nonempty finite backup are required")
        if not self.terminal.admissible:
            reasons.append("The terminal tracking inequalities are not certified")
        for index, control in enumerate(controls):
            entry = control.entry_box or current
            if not entry.contains(current):
                reasons.append(f"Stage {index} entry does not contain the previous proven successor")
                break
            if not self.input_limits.contains(control.held_input):
                reasons.append(f"Stage {index} input exceeds admissible limits")
                break
            try:
                _binary_command(control.held_input)
            except (ValueError, ArithmeticError):
                reasons.append(f"Stage {index} input box does not cover an executable binary64 command")
                break
            request = HeldInputRequest(entry, control.held_input, (q, q),
                                       (control.hold_width, control.hold_width), self.disturbance)
            flow = self.flow_provider(request)
            if not _valid_flow(flow, request, self.terminal.bounds.scope.frame):
                reasons.append(f"Stage {index} continuous flow is not proved in the declared frame and clock")
                break
            geometry = self.geometry_checker(index, flow)
            if (not isinstance(geometry, CollisionCertificate) or not geometry.admissible
                    or geometry.flow_fingerprints != (flow.fingerprint,)):
                reasons.append(f"Stage {index} continuous occupied-set separation is not proved")
                break
            stages.append(BackupStage(index, q, control, entry, flow, geometry))
            q, current = q + control.hold_width, flow.endpoint
        if len(stages) != len(controls):
            reasons.append("No complete admissible continuation was found; initial admission is refused")
        elif terminal_reference.anomaly != q:
            reasons.append("Terminal reference is not evaluated at the exact backup end anomaly")
        elif not self.terminal.contains(current, terminal_reference):
            reasons.append("The complete final successor does not belong to the invariant terminal set")
        elif not self.terminal.bounds.scope.contains_hold(q, self.terminal.bounds.maximum_hold):
            reasons.append("The terminal proof scope does not cover even one complete following hold")
        return ContinuationCertificate(not reasons, tuple(reasons), tuple(stages), self.terminal, terminal_reference)

    def supervisor(self, certificate: ContinuationCertificate) -> ContinuationSupervisor:
        """Recertify the sequence so serialized status flags never grant admission."""
        return ContinuationSupervisor(self, certificate)


@dataclass(frozen=True)
class ContinuationDecision:
    source: str
    reason: str
    stage_index: int
    held_input: IntervalBox
    flow: FlowCertificate
    geometry: CollisionCertificate

    def float_command(self) -> tuple[float, ...]:
        """Choose the box midpoint only if binary64 rounding stays in the box."""
        return _binary_command(self.held_input)

    def as_dict(self, *, include_segments=False):
        return {"source": self.source, "reason": self.reason, "stage_index": self.stage_index,
                "held_input": self.held_input.as_dict(),
                "flow": self.flow.as_dict(include_segments=include_segments),
                "geometry": self.geometry.as_dict()}


class ContinuationSupervisor:
    """Execute only commands that preserve the already certified backup chain.

    Construct through ``ContinuationBuilder.supervisor``. ``decide`` occurs
    before applying an input; ``advance`` consumes the returned proven successor
    and an optional independently valid observation box after the exact hold.
    This class never advances or pauses a physical plant on the caller's behalf.
    """

    def __init__(self, builder: ContinuationBuilder, certificate: ContinuationCertificate):
        if not certificate.stages:
            raise ValueError("An incomplete backup has no admitted supervisor")
        rebuilt = builder.certify(certificate.stages[0].entry_box, certificate.stages[0].start_anomaly,
                                  tuple(HeldControl(s.control.hold_width, s.control.held_input, s.entry_box)
                                        for s in certificate.stages), certificate.terminal_reference)
        if not rebuilt.admissible:
            raise ValueError("Continuation refused: " + "; ".join(rebuilt.reasons))
        self.builder, self.certificate = builder, rebuilt
        self.index, self.state_enclosure = 0, rebuilt.stages[0].entry_box
        self._pending: ContinuationDecision | None = None

    @property
    def finished(self) -> bool:
        return self.index == len(self.certificate.stages)

    def decide(self, proposed_input: IntervalBox | None = None, *, solver_status="NO_PROPOSAL") -> ContinuationDecision:
        if self.finished or self._pending is not None:
            raise ValueError("No new hold is available, or the previous hold has not been advanced")
        stage = self.certificate.stages[self.index]
        if not stage.entry_box.contains(self.state_enclosure):
            raise ValueError("The current belief is outside the admitted backup entry set")
        reason = f"Stored certified backup selected: {solver_status}"
        if proposed_input is not None:
            if not isinstance(proposed_input, IntervalBox) or proposed_input.dimension != 3:
                reason = "Malformed performance input; certified backup selected"
            else:
                try:
                    if self.builder.input_limits.contains(proposed_input):
                        _binary_command(proposed_input)
                        request = HeldInputRequest(self.state_enclosure, proposed_input,
                                                   (stage.start_anomaly, stage.start_anomaly),
                                                   (stage.control.hold_width, stage.control.hold_width),
                                                   self.builder.disturbance)
                        flow = self.builder.flow_provider(request)
                        if _valid_flow(flow, request, self.certificate.terminal.bounds.scope.frame):
                            geometry = self.builder.geometry_checker(self.index, flow)
                            next_admissible = (self.certificate.stages[self.index + 1].entry_box.contains(flow.endpoint)
                                               if self.index + 1 < len(self.certificate.stages) else
                                               self.certificate.terminal.contains(flow.endpoint, self.certificate.terminal_reference))
                            if (isinstance(geometry, CollisionCertificate) and geometry.admissible
                                    and geometry.flow_fingerprints == (flow.fingerprint,) and next_admissible):
                                self._pending = ContinuationDecision("PERFORMANCE", "Safe flow and certified successor proved",
                                                                      self.index, proposed_input, flow, geometry)
                                return self._pending
                        reason = "Performance input did not prove both swept safety and an admissible successor; certified backup selected"
                    else:
                        reason = "Performance input exceeded certified limits; certified backup selected"
                except Exception as error:
                    # A proposal proof service may fail while the stored chain
                    # remains valid. Keep the already certified input available
                    # and expose the failure reason in the execution record.
                    reason = (f"Performance proof failed ({type(error).__name__}: {error}); "
                              "certified backup selected")
        self._pending = ContinuationDecision("BACKUP", reason, self.index, stage.control.held_input,
                                             stage.flow, stage.geometry)
        return self._pending

    def advance(self, observation_box: IntervalBox | None = None) -> IntervalBox:
        if self._pending is None:
            raise ValueError("A certified decision is required before advancing a hold")
        predicted = self._pending.flow.endpoint
        successor = predicted if observation_box is None else _intersection(predicted, observation_box)
        next_index = self.index + 1
        next_admissible = (self.certificate.stages[next_index].entry_box.contains(successor)
                           if next_index < len(self.certificate.stages) else
                           self.certificate.terminal.contains(successor, self.certificate.terminal_reference))
        if not next_admissible:
            raise ValueError("Successor does not belong to the certified continuation")
        self.state_enclosure, self.index, self._pending = successor, next_index, None
        return successor

    def terminal_handoff(self) -> tuple[TerminalTrackingController, IntervalBox, TerminalReferenceSample]:
        if not self.finished or not self.certificate.terminal.contains(
                self.state_enclosure, self.certificate.terminal_reference):
            raise ValueError("The full continuation has not reached its certified terminal set")
        return (TerminalTrackingController(self.certificate.terminal), self.state_enclosure,
                self.certificate.terminal_reference)
