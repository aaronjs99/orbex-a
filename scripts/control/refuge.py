"""Finite-domain sampled feedback refuge certificates.

All admission inequalities use exact rationals.  Nonlinear drift and reference
bounds are explicit trusted proof inputs; a numerical trajectory or sampled
maximum cannot replace them.  See ``docs/fleet-fallback.md`` for the first-exit
argument, finite fuel scope and executable provider contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
from functools import cached_property
import hashlib
import json
from typing import Any, Callable, Protocol, Sequence

from .reachability import IntervalBox, _exact

REFUGE_METHOD = "exact-sampled-reference-refuge-v1"


def _vec(values, length, name, *, nonnegative=False, positive=False):
    value = tuple(map(_exact, values))
    if len(value) != length or (nonnegative and any(x < 0 for x in value)) or (positive and any(x <= 0 for x in value)):
        raise ValueError(f"{name} requires {length} valid bounded components")
    return value


def _mat(value, name):
    rows = tuple(_vec(row, 3, name, nonnegative=True) for row in value)
    if len(rows) != 3:
        raise ValueError(f"{name} must be a nonnegative 3 by 3 matrix")
    return rows


def _dot(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def _add(*vectors):
    return tuple(sum(values) for values in zip(*vectors))


def _absbox(box):
    return tuple(max(abs(lo), abs(hi)) for lo, hi in zip(box.lower, box.upper))


def _identity(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"An explicit {name} is required")
    return value


def _scope(start, end):
    start, end = _exact(start), _exact(end)
    if start >= end:
        raise ValueError("A positive finite reference-clock scope is required")
    return start, end


def exact_record(value):
    """Canonical exact evidence; float display exports do not enter identities."""
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, IntervalBox):
        return dict(lower=tuple(map(str, value.lower)), upper=tuple(map(str, value.upper)))
    if is_dataclass(value):
        return {field.name: exact_record(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): exact_record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(exact_record(item) for item in value)
    if isinstance(value, float):
        return str(_exact(value))
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(exact_record(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ControlContext:
    """Immutable identities shared by bounds, observations and actual execution."""

    model_id: str
    nominal_model_id: str
    parameter_set_id: str
    target_model_id: str
    frame: str
    clock_id: str
    sensor_id: str
    actuator_id: str
    geometry_id: str
    reference_id: str
    fuel_domain_id: str

    def __post_init__(self):
        for field in fields(self):
            _identity(getattr(self, field.name), field.name)

    @cached_property
    def fingerprint(self):
        return fingerprint(self)


@dataclass(frozen=True)
class ReferenceJet:
    q: Any
    position: Sequence[Any]
    velocity: Sequence[Any]
    acceleration: Sequence[Any]
    reference_id: str
    frame: str
    clock_id: str

    def __post_init__(self):
        object.__setattr__(self, "q", _exact(self.q))
        for name in ("position", "velocity", "acceleration"):
            object.__setattr__(self, name, _vec(getattr(self, name), 3, name))
        for name in ("reference_id", "frame", "clock_id"):
            _identity(getattr(self, name), name)


@dataclass(frozen=True)
class ReferenceBounds:
    """Proved C2 reference jets; acceleration is Lipschitz with jerk_abs.

    The provider must describe one continuous reference across successive
    windows.  Bounds cover its true values, before declared evaluation errors.
    """

    start: Any
    end: Any
    position: IntervalBox
    velocity: IntervalBox
    acceleration: IntervalBox
    jerk_abs: Sequence[Any]
    reference_id: str
    frame: str
    clock_id: str
    proof_reference: str
    assumptions: tuple[str, ...]
    proof_kind: str = "validated-interval-global"

    def __post_init__(self):
        start, end = _scope(self.start, self.end)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        if any(not isinstance(box, IntervalBox) or box.dimension != 3 for box in
               (self.position, self.velocity, self.acceleration)):
            raise ValueError("Reference jets require three-dimensional interval boxes")
        object.__setattr__(self, "jerk_abs", _vec(self.jerk_abs, 3, "jerk_abs", nonnegative=True))
        for name in ("reference_id", "frame", "clock_id", "proof_reference"):
            _identity(getattr(self, name), name)
        _proof_source(self.proof_kind, self.assumptions)
        object.__setattr__(self, "assumptions", tuple(self.assumptions))

    @property
    def is_constant(self):
        return (self.position.lower == self.position.upper
                and self.velocity == IntervalBox.point((0, 0, 0))
                and self.acceleration == IntervalBox.point((0, 0, 0))
                and self.jerk_abs == (0, 0, 0))


class ReferenceProvider(Protocol):
    def sample(self, q: Fraction) -> ReferenceJet: ...
    def bounds(self, start: Fraction, end: Fraction) -> ReferenceBounds: ...


def _proof_source(kind, assumptions):
    if kind not in ("analytic-global", "validated-interval-global"):
        raise ValueError("Global analytical or validated interval proof bounds are required")
    if not assumptions or any(not isinstance(item, str) or not item.strip() for item in assumptions):
        raise ValueError("Explicit proof assumptions are required")


@dataclass(frozen=True)
class NonlinearDriftBounds:
    """Bounds for p'=v, v'=F(q,p,v,theta)+u in the declared common clock.

    mismatch_abs bounds F-F_nom on state_domain and the complete input_domain;
    nominal derivatives/range cover sensing_domain and all mean-value paths.
    Position-row or input-matrix discrepancies must be transformed or enclosed
    by the provider; an acceleration bound cannot hide incorrect kinematics.
    """

    start: Any
    end: Any
    state_domain: IntervalBox
    sensing_domain: IntervalBox
    input_domain: IntervalBox
    nominal_drift_range: IntervalBox
    mismatch_abs: Sequence[Any]
    position_jacobian_abs: Sequence[Sequence[Any]]
    velocity_jacobian_abs: Sequence[Sequence[Any]]
    clock_derivative_abs: Sequence[Any]
    evaluation_error_abs: Sequence[Any]
    model_id: str
    nominal_model_id: str
    parameter_set_id: str
    target_model_id: str
    fuel_domain_id: str
    frame: str
    clock_id: str
    proof_reference: str
    assumptions: tuple[str, ...]
    proof_kind: str = "validated-interval-global"

    def __post_init__(self):
        start, end = _scope(self.start, self.end)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        for name, size in (("state_domain", 6), ("sensing_domain", 6),
                           ("input_domain", 3), ("nominal_drift_range", 3)):
            if not isinstance(getattr(self, name), IntervalBox) or getattr(self, name).dimension != size:
                raise ValueError(f"{name} needs an explicit {size}-dimensional box")
        for name in ("mismatch_abs", "clock_derivative_abs", "evaluation_error_abs"):
            object.__setattr__(self, name, _vec(getattr(self, name), 3, name, nonnegative=True))
        for name in ("position_jacobian_abs", "velocity_jacobian_abs"):
            object.__setattr__(self, name, _mat(getattr(self, name), name))
        for name in ("model_id", "nominal_model_id", "parameter_set_id", "target_model_id", "fuel_domain_id",
                     "frame", "clock_id", "proof_reference"):
            _identity(getattr(self, name), name)
        _proof_source(self.proof_kind, self.assumptions)
        object.__setattr__(self, "assumptions", tuple(self.assumptions))


@dataclass(frozen=True)
class RefugeTrackingSpec:
    start: Any
    end: Any
    context: ControlContext
    omega: Sequence[Any]
    phi: Sequence[Any]
    lambda_gain: Sequence[Any]
    beta: Sequence[Any]
    maximum_hold: Any
    acceleration_bound: Sequence[Any]
    input_limits: IntervalBox
    sensing_error_abs: Sequence[Any]
    actuator_error_abs: Sequence[Any]
    command_rounding_abs: Sequence[Any]
    reference_error_abs: Sequence[Any]
    fuel_rate_upper_per_q: Any
    available_fuel: Any
    fuel_proof_reference: str
    fuel_assumptions: tuple[str, ...]

    def __post_init__(self):
        start, end = _scope(self.start, self.end)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        if not isinstance(self.context, ControlContext):
            raise TypeError("A complete ControlContext is required")
        for name in ("omega", "phi", "lambda_gain", "beta"):
            object.__setattr__(self, name, _vec(getattr(self, name), 3, name, positive=True))
        for name, size in (("acceleration_bound", 3), ("sensing_error_abs", 6),
                           ("actuator_error_abs", 3), ("command_rounding_abs", 3), ("reference_error_abs", 9)):
            object.__setattr__(self, name, _vec(getattr(self, name), size, name, nonnegative=True))
        hold = _exact(self.maximum_hold)
        if hold <= 0:
            raise ValueError("maximum_hold must be positive")
        object.__setattr__(self, "maximum_hold", hold)
        if not isinstance(self.input_limits, IntervalBox) or self.input_limits.dimension != 3:
            raise ValueError("A three-dimensional hardware input box is required")
        for name in ("fuel_rate_upper_per_q", "available_fuel"):
            value = _exact(getattr(self, name))
            if value < 0:
                raise ValueError("Fuel bounds must be nonnegative")
            object.__setattr__(self, name, value)
        _identity(self.fuel_proof_reference, "fuel proof reference")
        _proof_source("analytic-global", self.fuel_assumptions)
        object.__setattr__(self, "fuel_assumptions", tuple(self.fuel_assumptions))


@dataclass(frozen=True)
class RefugeObservation:
    q: Any
    measurement: Sequence[Any]
    state_enclosure: IntervalBox
    sensor_id: str
    frame: str
    clock_id: str

    def __post_init__(self):
        object.__setattr__(self, "q", _exact(self.q))
        object.__setattr__(self, "measurement", _vec(self.measurement, 6, "measurement"))
        if not isinstance(self.state_enclosure, IntervalBox) or self.state_enclosure.dimension != 6:
            raise ValueError("A valid six-state observation enclosure is required")
        for name in ("sensor_id", "frame", "clock_id"):
            _identity(getattr(self, name), name)

    @cached_property
    def fingerprint(self):
        return fingerprint(self)


@dataclass(frozen=True)
class RefugeCertificate:
    spec: RefugeTrackingSpec
    reference_bounds: ReferenceBounds
    drift_bounds: NonlinearDriftBounds
    state_box: IntervalBox
    sensing_box: IntervalBox
    velocity_error_bound: tuple[Fraction, ...]
    drift_hold_error: tuple[Fraction, ...]
    relative_acceleration_bound: tuple[Fraction, ...]
    position_margin: tuple[Fraction, ...]
    acceleration_margin: tuple[Fraction, ...]
    sliding_margin: tuple[Fraction, ...]
    input_box: IntervalBox
    required_fuel: Fraction
    reasons: tuple[str, ...]
    method: str = REFUGE_METHOD

    @property
    def admissible(self):
        return self.method == REFUGE_METHOD and not self.reasons

    @cached_property
    def fingerprint(self):
        return fingerprint(self)

    def as_dict(self):
        return dict(status="CONDITIONAL_PROVEN" if self.admissible else "REFUSED",
                    identity=self.fingerprint, **exact_record(self))


def _expanded(box, radius):
    return IntervalBox(tuple(lo-r for lo, r in zip(box.lower, radius)),
                       tuple(hi+r for hi, r in zip(box.upper, radius)))


class RefugeBoundsBuilder:
    """Derive reference/drift bounds and verify exact first-exit inequalities.

    drift_bounds_provider(spec, reference_bounds, state_box, sensing_box) must
    return genuine NonlinearDriftBounds.  nominal_drift(q, measurement) uses the
    exact declared model identity and its supplied numerical-error envelope.
    """

    def __init__(self, spec: RefugeTrackingSpec, reference_provider: ReferenceProvider,
                 drift_bounds_provider: Callable, nominal_drift: Callable, *, nominal_model_id: str):
        if not isinstance(spec, RefugeTrackingSpec) or not callable(drift_bounds_provider) or not callable(nominal_drift):
            raise TypeError("Explicit tracking spec, drift bounds and nominal drift providers are required")
        if nominal_model_id != spec.context.nominal_model_id:
            raise ValueError("Nominal drift identity differs from the certified controller")
        self.spec, self.reference_provider = spec, reference_provider
        self.drift_bounds_provider, self.nominal_drift = drift_bounds_provider, nominal_drift
        self.nominal_model_id = nominal_model_id

    def build(self):
        b, reasons = self.spec, []
        r = self.reference_provider.bounds(b.start, b.end)
        if not isinstance(r, ReferenceBounds):
            raise TypeError("Reference provider did not return proved ReferenceBounds")
        if (r.start > b.start or r.end < b.end or r.reference_id != b.context.reference_id
                or r.frame != b.context.frame or r.clock_id != b.context.clock_id):
            raise ValueError("Reference bounds do not cover the tracking scope/context")
        lam, phi, omega, beta, tau = b.lambda_gain, b.phi, b.omega, b.beta, b.maximum_hold
        ve = tuple(p+l*o for p, l, o in zip(phi, lam, omega))
        vp = _absbox(r.velocity)
        ap = _absbox(r.acceleration)
        state_box = IntervalBox(tuple(lo-o for lo, o in zip(r.position.lower, omega))
            + tuple(lo-v for lo, v in zip(r.velocity.lower, ve)),
            tuple(hi+o for hi, o in zip(r.position.upper, omega))
            + tuple(hi+v for hi, v in zip(r.velocity.upper, ve)))
        sensing_box = _expanded(state_box, b.sensing_error_abs)
        d = self.drift_bounds_provider(b, r, state_box, sensing_box)
        if not isinstance(d, NonlinearDriftBounds):
            raise TypeError("Nonlinear model bounds are unavailable; numerical predictions do not certify a refuge")
        for name in ("model_id", "nominal_model_id", "parameter_set_id", "target_model_id", "fuel_domain_id", "frame", "clock_id"):
            if getattr(d, name) != getattr(b.context, name):
                reasons.append(f"Nonlinear drift {name} differs from the execution context")
        if d.start > b.start or d.end < b.end:
            reasons.append("Nonlinear drift bounds do not cover the finite scope")
        if not d.state_domain.contains(state_box) or not d.sensing_domain.contains(sensing_box):
            reasons.append("Reference tube or sensor-expanded tube exceeds the proved nonlinear domains")
        if not d.input_domain.contains(b.input_limits):
            reasons.append("Nonlinear residual bounds do not cover every admissible held input")
        ep, ev = b.sensing_error_abs[:3], b.sensing_error_abs[3:]
        rp, rv, ra = b.reference_error_abs[:3], b.reference_error_abs[3:6], b.reference_error_abs[6:]
        eta = _add(b.actuator_error_abs, b.command_rounding_abs)
        absolute_velocity = _add(vp, ve)
        drift_error = _add(d.mismatch_abs, d.evaluation_error_abs, eta,
            tuple(tau*x for x in d.clock_derivative_abs),
            _dot(d.position_jacobian_abs, tuple(e+tau*v for e, v in zip(ep, absolute_velocity))),
            _dot(d.velocity_jacobian_abs, tuple(e+tau*v for e, v in zip(ev, b.acceleration_bound))))
        # A second bound retains cancellation of the held reference feedforward.
        # Writing e_v'=a_actual-r_a(q), the implemented law gives
        # e_v' = [F(q,x)-Fbar(q_k,xhat_k)+actuation_error]
        #        +[r_a_evaluated(q_k)-r_a(q)]
        #        -Lambda*(vhat_k-r_v_evaluated(q_k))-beta*sat(s_hat_k/Phi).
        # The four terms are bounded by drift_error, reference_hold,
        # Lambda*(Ve+Ev+Rv), and beta. Both this bound and the independent
        # absolute-acceleration-plus-reference bound are valid until first exit.
        # The physical acceleration self-bound above remains mandatory.
        reference_hold = tuple(a+tau*j for a, j in zip(ra, r.jerk_abs))
        cancelled_relative_acceleration = _add(drift_error, reference_hold,
            tuple(l*(v+e+r0) for l,v,e,r0 in zip(lam,ve,ev,rv)), beta)
        relative_acceleration = tuple(min(absolute+reference,cancelled)
            for absolute,reference,cancelled in
            zip(b.acceleration_bound,ap,cancelled_relative_acceleration))
        sliding_rate = _add(relative_acceleration, tuple(l*v for l, v in zip(lam, ve)))
        sensing_s = _add(ev, rv, tuple(l*(e+r0) for l, e, r0 in zip(lam, ep, rp)))
        acceleration_required = _add(ap, ra, tuple(l*(v+e+r0) for l, v, e, r0 in zip(lam, ve, ev, rv)), beta, drift_error)
        sliding_required = _add(drift_error, reference_hold,
            tuple(l*(e+r0+tau*dv) for l, e, r0, dv in zip(lam, ev, rv, relative_acceleration)),
            tuple((force/p)*(error+tau*rate) for force, p, error, rate in zip(beta, phi, sensing_s, sliding_rate)))
        position_margin = tuple(l*o-p for l, o, p in zip(lam, omega, phi))
        acceleration_margin = tuple(got-need for got, need in zip(b.acceleration_bound, acceleration_required))
        sliding_margin = tuple(got-need for got, need in zip(beta, sliding_required))
        for name, values in (("position inward", position_margin), ("acceleration self-bound", acceleration_margin),
                             ("sliding inward", sliding_margin)):
            if any(value < 0 for value in values):
                reasons.append(f"Negative {name} margin")
        # Nominal drift evaluations include eps_F; reference evaluations include
        # their own independent errors.  Physical actuator and conversion errors
        # are both contained in the actual hardware input interval.
        feedback = tuple(l*(v+e+r0)+force for l, v, e, r0, force in zip(lam, ve, ev, rv, beta))
        padding = _add(feedback, eta, d.evaluation_error_abs, ra)
        input_box = IntervalBox(tuple(a-f-p for a, f, p in zip(r.acceleration.lower, d.nominal_drift_range.upper, padding)),
                                tuple(a-f+p for a, f, p in zip(r.acceleration.upper, d.nominal_drift_range.lower, padding)))
        if not b.input_limits.contains(input_box):
            reasons.append("The complete sampled command/error range exceeds hardware input bounds")
        fuel = (b.end-b.start)*b.fuel_rate_upper_per_q
        if fuel > b.available_fuel:
            reasons.append("Finite feedback scope exceeds the declared fuel reserve")
        return RefugeCertificate(b, r, d, state_box, sensing_box, ve, drift_error,relative_acceleration,
            position_margin, acceleration_margin, sliding_margin, input_box, fuel, tuple(reasons))


@dataclass(frozen=True)
class CertifiedFeedbackCommand:
    q: Fraction
    hold_width: Fraction
    command: tuple[Fraction, ...]
    applied_input: IntervalBox
    rounding_error_abs: tuple[Fraction, ...]
    context: ControlContext
    certificate_id: str
    reference: ReferenceJet
    source: str = "CERTIFIED_FEEDBACK"

    def float_command(self):
        values = tuple(float(value) for value in self.command)
        if any(abs(_exact(value)-exact) > error for value, exact, error in
               zip(values, self.command, self.rounding_error_abs)):
            raise ValueError("Binary64 command conversion exceeds the certified rounding budget")
        if not self.applied_input.contains(IntervalBox.point(values)):
            raise ValueError("Executable command is outside the proved applied-input interval")
        return values

    def as_dict(self):
        return exact_record(self)


class RefugeController:
    """Rebuild global bounds once, then evaluate the held law without an NLP."""

    def __init__(self, builder: RefugeBoundsBuilder):
        if not isinstance(builder, RefugeBoundsBuilder):
            raise TypeError("Supply the original bounds builder for recertification")
        result = builder.build()
        if not result.admissible:
            raise ValueError("Refuge certificate refused: " + "; ".join(result.reasons))
        self.builder, self.certificate = builder, result

    def reference_at(self, q):
        q, b = _exact(q), self.certificate.spec
        if not b.start <= q <= b.end:
            raise ValueError("Reference query lies outside the finite certified scope")
        value = self.builder.reference_provider.sample(q)
        if (not isinstance(value, ReferenceJet) or value.q != q or value.reference_id != b.context.reference_id
                or value.frame != b.context.frame or value.clock_id != b.context.clock_id):
            raise ValueError("Reference sample clock/identity differs from the proved reference")
        r = self.certificate.reference_bounds
        for jet, domain, errors in ((value.position, r.position, b.reference_error_abs[:3]),
                                    (value.velocity, r.velocity, b.reference_error_abs[3:6]),
                                    (value.acceleration, r.acceleration, b.reference_error_abs[6:])):
            if not _expanded(domain, errors).contains(IntervalBox.point(jet)):
                raise ValueError("Reference sample contradicts the global jet bounds")
        return value

    def contains(self, state_box, q, hold_width=0):
        b, q, h = self.certificate.spec, _exact(q), _exact(hold_width)
        if (not isinstance(state_box, IntervalBox) or state_box.dimension != 6 or h < 0
                or q < b.start or q+h > b.end or h > b.maximum_hold):
            return False
        reference = self.reference_at(q)
        rp, rv = b.reference_error_abs[:3], b.reference_error_abs[3:6]
        for i in range(3):
            elo, ehi = state_box.lower[i]-reference.position[i], state_box.upper[i]-reference.position[i]
            if max(abs(elo), abs(ehi))+rp[i] > b.omega[i]:
                return False
            slo = state_box.lower[i+3]-reference.velocity[i]+b.lambda_gain[i]*elo
            shi = state_box.upper[i+3]-reference.velocity[i]+b.lambda_gain[i]*ehi
            if max(abs(slo), abs(shi))+rv[i]+b.lambda_gain[i]*rp[i] > b.phi[i]:
                return False
        return True

    def observation_box(self, observation, q):
        """Check sensing identity/error and intersect its box with the tube hull.

        This is a consistency contraction, not fresh invariant-set admission.
        Fleet runtime may retain the established relational tube invariant even
        when a rectangular sensor enclosure protrudes beyond that tube.
        """
        b = self.certificate.spec
        if (not isinstance(observation, RefugeObservation) or observation.q != _exact(q)
                or observation.sensor_id != b.context.sensor_id or observation.frame != b.context.frame
                or observation.clock_id != b.context.clock_id):
            raise ValueError("Observation identity or exact sample clock changed")
        if any(max(abs(lo-m), abs(hi-m)) > error for lo, hi, m, error in
               zip(observation.state_enclosure.lower, observation.state_enclosure.upper,
                   observation.measurement, b.sensing_error_abs)):
            raise ValueError("Observation enclosure exceeds the certified measurement-error budget")
        reference = self.reference_at(q)
        rp, rv = b.reference_error_abs[:3], b.reference_error_abs[3:6]
        radius = tuple(o+e for o, e in zip(b.omega, rp)) + tuple(v+e for v, e in
            zip(self.certificate.velocity_error_bound, rv))
        hull = IntervalBox.from_center_radius(reference.position+reference.velocity, radius)
        lower = tuple(max(a, c) for a, c in zip(hull.lower, observation.state_enclosure.lower))
        upper = tuple(min(a, c) for a, c in zip(hull.upper, observation.state_enclosure.upper))
        if any(a > c for a, c in zip(lower, upper)):
            raise ValueError("Observation contradicts the already admitted reference tube")
        # Also require possible sliding membership; all constraints separate by
        # axis, so these exact interval sums give a necessary consistency gate.
        for i in range(3):
            lo = lower[i+3]-reference.velocity[i]+b.lambda_gain[i]*(lower[i]-reference.position[i])
            hi = upper[i+3]-reference.velocity[i]+b.lambda_gain[i]*(upper[i]-reference.position[i])
            padding = rv[i]+b.lambda_gain[i]*rp[i]
            if lo-padding > b.phi[i] or hi+padding < -b.phi[i]:
                raise ValueError("Observation contradicts the admitted sliding region")
        return IntervalBox(lower, upper)

    def command(self, observation: RefugeObservation, *, q, hold_width, admitted_invariant=False):
        """Evaluate one hold; admitted_invariant is only for an owning runtime.

        A new caller must establish whole-box entry.  An owning fleet runtime
        can retain its prior proved invariant without mistaking a protruding
        observation box for a true state outside the set.
        """
        b, q, h = self.certificate.spec, _exact(q), _exact(hold_width)
        if not 0 < h <= b.maximum_hold or not b.start <= q < q+h <= b.end:
            raise ValueError("The complete feedback hold must fit its finite clock scope")
        self.observation_box(observation, q)
        if not admitted_invariant and not self.contains(observation.state_enclosure, q, h):
            raise ValueError("The complete fresh observation enclosure is outside the certified entry region")
        reference, measured = self.reference_at(q), observation.measurement
        nominal = _vec(self.builder.nominal_drift(q, measured), 3, "nominal drift evaluation")
        if not _expanded(self.certificate.drift_bounds.nominal_drift_range,
                         self.certificate.drift_bounds.evaluation_error_abs).contains(IntervalBox.point(nominal)):
            raise ValueError("Nominal drift evaluation contradicts its global range")
        relative_v = tuple(v-r for v, r in zip(measured[3:], reference.velocity))
        sliding = tuple(v+l*(p-r) for v, l, p, r in zip(relative_v, b.lambda_gain, measured[:3], reference.position))
        saturation = tuple(max(Fraction(-1), min(Fraction(1), s/phi)) for s, phi in zip(sliding, b.phi))
        command = tuple(a-f-l*v-force*sat for a, f, l, v, force, sat in
            zip(reference.acceleration, nominal, b.lambda_gain, relative_v, b.beta, saturation))
        eta = _add(b.actuator_error_abs, b.command_rounding_abs)
        applied = IntervalBox.from_center_radius(command, eta)
        if not b.input_limits.contains(applied):
            raise ValueError("The actual evaluated feedback command exceeds hardware input bounds")
        result = CertifiedFeedbackCommand(q, h, command, applied, b.command_rounding_abs,
            b.context, self.certificate.fingerprint, reference)
        result.float_command()
        return result

