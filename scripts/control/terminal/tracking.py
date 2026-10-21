"""Exact algebra certificates for a held-input terminal tracking controller.

The reference is an authoritative continuous trajectory, not an MPC shooting
center.  Certification is conditional on the supplied globally valid bounds;
it does not infer bounds from samples or certify a reference implementation.
All admission decisions use exact rational arithmetic.  See
``docs/terminal-tracking.md`` for the controller and first-exit proof.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from numbers import Integral
from typing import Any, Sequence

from orbexa.control.certification.reachability import FRAME, IntervalBox


TERMINAL_METHOD = "exact-held-reference-terminal-v1"
MIDPOINT_TERMINAL_METHOD = "exact-held-midpoint-reference-terminal-v1"


class TerminalFeedforwardPolicy(str, Enum):
    """Supported held laws; the residual factor is not a caller-tuned bound."""

    START_V1 = "start_feedforward_v1"
    MIDPOINT_V1 = "midpoint_feedforward_v1"

    @property
    def sample_fraction(self):
        return Fraction(0) if self is self.START_V1 else Fraction(1, 2)

    @property
    def residual_fraction(self):
        return Fraction(1) if self is self.START_V1 else Fraction(1, 2)

    @property
    def certificate_method(self):
        return TERMINAL_METHOD if self is self.START_V1 else MIDPOINT_TERMINAL_METHOD


def _q(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not numerical bounds")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, str):
        return Fraction(value)
    if hasattr(value, "as_integer_ratio"):
        numerator, denominator = value.as_integer_ratio()
        return Fraction(int(numerator), int(denominator))
    raise TypeError("Use finite binary floats, integers, Fraction or rational strings")


def _vector(values, length: int, name: str, *, nonnegative=False, positive=False):
    result = tuple(map(_q, values))
    if len(result) != length or (nonnegative and any(x < 0 for x in result)) or (
            positive and any(x <= 0 for x in result)):
        raise ValueError(f"{name} needs {length} finite components with valid signs")
    return result


def _matrix(values, name: str, *, nonnegative=False):
    result = tuple(_vector(row, 6, name, nonnegative=nonnegative) for row in values)
    if len(result) != 3:
        raise ValueError(f"{name} must have shape (3,6)")
    return result


def _dot(matrix, vector):
    return tuple(sum((a * x for a, x in zip(row, vector)), Fraction(0)) for row in matrix)


def _sum(*vectors):
    return tuple(map(sum, zip(*vectors)))


def _mul(left, right):
    return tuple(a * b for a, b in zip(left, right))


def _encode(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, tuple):
        return tuple(_encode(v) for v in value)
    if isinstance(value, dict):
        return {key: _encode(v) for key, v in value.items()}
    return value


@dataclass(frozen=True)
class BoundScope:
    """Declared validity of analytical or interval-proved *global* bounds.

    Finite scope is an inclusive anomaly interval.  ``all_time=True`` covers
    every future anomaly from ``start_anomaly``. A sampled maximum or a fitted
    constant is not accepted as a global proof source.
    """

    start_anomaly: Any
    end_anomaly: Any | None = None
    all_time: bool = False
    proof_kind: str = ""
    proof_reference: str = ""
    assumptions: tuple[str, ...] = ()
    frame: str = FRAME

    def __post_init__(self):
        if not isinstance(self.all_time, bool):
            raise ValueError("all_time must be an explicit Boolean")
        start = _q(self.start_anomaly)
        end = None if self.end_anomaly is None else _q(self.end_anomaly)
        if self.all_time != (end is None) or (end is not None and end < start):
            raise ValueError("Supply a finite ordered interval or explicit all-time scope")
        if self.proof_kind not in ("analytic-global", "validated-interval-global"):
            raise ValueError("A global analytical or validated interval bound source is required")
        if not self.proof_reference.strip() or not self.assumptions or not all(
                isinstance(item, str) and item.strip() for item in self.assumptions):
            raise ValueError("Bound source and explicit assumptions are required")
        if not self.frame:
            raise ValueError("An explicit common frame is required")
        object.__setattr__(self, "start_anomaly", start)
        object.__setattr__(self, "end_anomaly", end)
        object.__setattr__(self, "assumptions", tuple(self.assumptions))

    def contains(self, anomaly) -> bool:
        anomaly = _q(anomaly)
        return self.start_anomaly <= anomaly and (self.end_anomaly is None or anomaly <= self.end_anomaly)

    def contains_hold(self, anomaly, hold) -> bool:
        anomaly, hold = _q(anomaly), _q(hold)
        return hold >= 0 and self.contains(anomaly) and self.contains(anomaly + hold)

    def as_dict(self):
        return _encode(vars(self))


@dataclass(frozen=True)
class TerminalGeometryBounds:
    """Global reference-center distances from occupied geometry in metres.

    ``target_distance_lower`` bounds distance to the *solid*, not its center.
    Peer entries are (reference-center distance lower, peer position-radius
    squared upper, peer physical-body radius). Reference/pose uncertainty must
    already be included in these proven global distance lower bounds.
    """

    target_distance_lower: Any
    body_radius: Any = 0
    clearance: Any = 0
    peers: tuple[tuple[Any, Any, Any], ...] = ()

    def __post_init__(self):
        for name in ("target_distance_lower", "body_radius", "clearance"):
            value = _q(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        peers = tuple(_vector(v, 3, "peer distance/radius", nonnegative=True) for v in self.peers)
        object.__setattr__(self, "peers", peers)

    def as_dict(self):
        return _encode(vars(self))


@dataclass(frozen=True)
class TerminalTrackingBounds:
    """Complete assumptions for one fixed reference-relative terminal set.

    Position and anomaly-velocity use m and m/rad; acceleration and input use
    m/rad^2. Gains have units 1/rad. All bounds cover ``scope`` continuously.
    Reference feedforward is u_g=p_g''-Ahat*x_g-dhat. Its Lipschitz bound is
    |u_g(q)-u_g(r)| <= feedforward_lipschitz*|q-r| for any two scope clocks.
    """

    scope: BoundScope
    reference_id: str
    omega: Sequence[Any]
    phi: Sequence[Any]
    lambda_gain: Sequence[Any]
    bandwidth: Sequence[Any]
    maximum_hold: Any
    nominal_matrix_abs: Sequence[Sequence[Any]]
    matrix_uncertainty_abs: Sequence[Sequence[Any]]
    nominal_matrix_variation: Sequence[Sequence[Any]]
    reference_state_abs: Sequence[Any]
    affine_uncertainty_abs: Sequence[Any]
    acceleration_disturbance_abs: Sequence[Any]
    feedforward_range: IntervalBox
    feedforward_lipschitz: Sequence[Any]
    sensing_error_abs: Sequence[Any]
    actuator_error_abs: Sequence[Any]
    input_limits: IntervalBox
    geometry: TerminalGeometryBounds
    position_tolerance: Any
    anomaly_velocity_tolerance: Any

    def __post_init__(self):
        if not self.reference_id.strip():
            raise ValueError("An authoritative reference identity is required")
        for name in ("omega", "phi", "lambda_gain", "bandwidth"):
            object.__setattr__(self, name, _vector(getattr(self, name), 3, name, positive=True))
        for name in ("nominal_matrix_abs", "matrix_uncertainty_abs", "nominal_matrix_variation"):
            object.__setattr__(self, name, _matrix(getattr(self, name), name, nonnegative=True))
        for name, length in (("reference_state_abs", 6), ("sensing_error_abs", 6),
                             ("affine_uncertainty_abs", 3), ("acceleration_disturbance_abs", 3),
                             ("feedforward_lipschitz", 3), ("actuator_error_abs", 3)):
            object.__setattr__(self, name, _vector(getattr(self, name), length, name, nonnegative=True))
        for name in ("maximum_hold", "position_tolerance", "anomaly_velocity_tolerance"):
            value = _q(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.feedforward_range.dimension != 3 or self.input_limits.dimension != 3:
            raise ValueError("Feedforward and input-limit boxes have dimension 3")

    def as_dict(self):
        return _encode({key: value.as_dict() if hasattr(value, "as_dict") else value
                        for key, value in vars(self).items()})

    @classmethod
    def from_dict(cls, record):
        """Restore assumptions, then recertify; a stored status is never trusted."""
        data = dict(record)
        data["scope"] = BoundScope(**data["scope"])
        data["geometry"] = TerminalGeometryBounds(**data["geometry"])
        for name in ("feedforward_range", "input_limits"):
            box = data[name]
            data[name] = IntervalBox(box["exact_lower"], box["exact_upper"])
        return cls(**data)


@dataclass(frozen=True)
class TerminalCertificate:
    bounds: TerminalTrackingBounds
    admissible: bool
    reasons: tuple[str, ...]
    quantities: dict = field(default_factory=dict)
    method: str = TERMINAL_METHOD
    policy: TerminalFeedforwardPolicy = TerminalFeedforwardPolicy.START_V1

    def __post_init__(self):
        policy = TerminalFeedforwardPolicy(self.policy)
        if self.method != policy.certificate_method:
            raise ValueError("Unsupported or inconsistent terminal held-law method/policy")
        object.__setattr__(self, "policy", policy)

    def as_dict(self):
        record = {"method": self.method, "admissible": self.admissible,
                "status": "CONDITIONAL_PROVEN" if self.admissible else "REFUSED",
                "reasons": self.reasons, "bounds": self.bounds.as_dict(),
                "exact_quantities": _encode(self.quantities),
                "claim": "Invariant held-feedback tracking conditional on the declared global bounds"}
        if self.policy is not TerminalFeedforwardPolicy.START_V1:
            record["held_policy"] = self.policy.value
        return record

    def contains(self, state: IntervalBox, reference: TerminalReferenceSample) -> bool:
        if not self.admissible or not self.bounds.scope.contains(reference.anomaly) or state.dimension != 6:
            return False
        if not reference.complies(self.bounds):
            return False
        for i in range(3):
            lo_p, hi_p = state.lower[i] - reference.state[i], state.upper[i] - reference.state[i]
            if lo_p < -self.bounds.omega[i] or hi_p > self.bounds.omega[i]:
                return False
            lo_s = state.lower[i + 3] - reference.state[i + 3] + self.bounds.lambda_gain[i] * lo_p
            hi_s = state.upper[i + 3] - reference.state[i + 3] + self.bounds.lambda_gain[i] * hi_p
            if lo_s < -self.bounds.phi[i] or hi_s > self.bounds.phi[i]:
                return False
        return True


def certify_terminal_tracking(bounds: TerminalTrackingBounds, *,
                              policy=TerminalFeedforwardPolicy.START_V1) -> TerminalCertificate:
    """Check every sufficient inequality using exact rational comparisons.

    Positive algebraic margins do not prove the global model/reference bounds
    themselves. The source and validity of those bounds remain explicit proof
    assumptions in the returned certificate.
    """
    policy = TerminalFeedforwardPolicy(policy)
    b = bounds
    a, delta, variation = b.nominal_matrix_abs, b.matrix_uncertainty_abs, b.nominal_matrix_variation
    lam, kappa, omega, phi, tau = b.lambda_gain, b.bandwidth, b.omega, b.phi, b.maximum_hold
    velocity = _sum(phi, _mul(lam, omega))
    error = omega + velocity
    sensed_error = _sum(error, b.sensing_error_abs)
    nv = b.sensing_error_abs[3:]
    ns = _sum(nv, _mul(lam, b.sensing_error_abs[:3]))
    a_delta = tuple(tuple(x + y for x, y in zip(row_a, row_d)) for row_a, row_d in zip(a, delta))
    sensing = _sum(_dot(a_delta, b.sensing_error_abs), _mul(lam, nv), _mul(kappa, ns))
    base = _sum(_dot(delta, b.reference_state_abs), b.affine_uncertainty_abs,
                b.acceleration_disturbance_abs, b.actuator_error_abs,
                tuple(tau * value * policy.residual_fraction for value in b.feedforward_lipschitz), sensing)
    reserve = _sum(_dot(a_delta, sensed_error), _mul(lam, _sum(velocity, nv)), _mul(kappa, phi))
    dv = _sum(_dot(a_delta, error), base, reserve)
    de = velocity + dv
    ds = _sum(dv, _mul(lam, velocity))
    rho = _sum(tuple(2 * value for value in _dot(delta, error)), _dot(variation, error),
               tuple(tau * value for value in _sum(_dot(a, de), _mul(lam, dv), _mul(kappa, ds))))
    total = _sum(base, rho)
    sliding_margin = tuple(k * p - w for k, p, w in zip(kappa, phi, total))
    position_margin = tuple(l * o - p for l, o, p in zip(lam, omega, phi))
    actuator_reserve = _sum(reserve, b.actuator_error_abs)
    lower_margin = tuple(lo - r - lim for lo, r, lim in zip(
        b.feedforward_range.lower, actuator_reserve, b.input_limits.lower))
    upper_margin = tuple(lim - hi - r for hi, r, lim in zip(
        b.feedforward_range.upper, actuator_reserve, b.input_limits.upper))
    radius_squared, speed_squared = sum(v * v for v in omega), sum(v * v for v in velocity)
    target_gap = b.geometry.target_distance_lower - b.geometry.body_radius - b.geometry.clearance
    reasons = []
    for name, values in (("sliding inward margin", sliding_margin), ("position inward margin", position_margin),
                         ("lower input margin", lower_margin), ("upper input margin", upper_margin)):
        if any(value < 0 for value in values):
            reasons.append(f"Negative {name}")
    if radius_squared > b.position_tolerance ** 2 or speed_squared > b.anomaly_velocity_tolerance ** 2:
        reasons.append("The complete terminal set exceeds the requested pose tolerance")
    if target_gap <= 0 or target_gap * target_gap <= radius_squared:
        reasons.append("Strict whole-set target separation is not established")
    for distance, peer_radius_squared, peer_body_radius in b.geometry.peers:
        gap = distance - b.geometry.body_radius - peer_body_radius - b.geometry.clearance
        excess = gap * gap - radius_squared - peer_radius_squared
        if gap <= 0 or excess <= 0 or excess * excess <= 4 * radius_squared * peer_radius_squared:
            reasons.append("Strict whole-set peer separation is not established")
    quantities = dict(velocity_peak=velocity, error_peak=error, sensing_forcing=sensing,
                      base_forcing=base, hold_forcing=rho, total_forcing=total,
                      input_reserve=reserve, actuator_reserve=actuator_reserve,
                      velocity_derivative_bound=dv, error_derivative_bound=de,
                      sliding_derivative_bound=ds, sliding_inward_margin=sliding_margin,
                      position_inward_margin=position_margin, lower_input_margin=lower_margin,
                      upper_input_margin=upper_margin, position_radius_squared=radius_squared,
                      anomaly_velocity_radius_squared=speed_squared,
                      target_distance_after_body_and_clearance=target_gap)
    if policy is not TerminalFeedforwardPolicy.START_V1:
        quantities.update(actual_feedforward_lipschitz=b.feedforward_lipschitz,
            feedforward_residual_scale=policy.residual_fraction,
            midpoint_feedforward_variation_abs=tuple(tau * value / 2 for value in b.feedforward_lipschitz))
    return TerminalCertificate(b, not reasons, tuple(reasons), quantities, policy.certificate_method, policy)


@dataclass(frozen=True)
class TerminalReferenceSample:
    """Authoritative exact-reference values at the actual sample anomaly.

    A caller must enclose any reference/clock evaluation error in the declared
    sensing/disturbance bounds. Samples from an unvalidated shooting trajectory
    do not meet this contract merely because they have this type.
    """

    anomaly: Any
    state: Sequence[Any]
    acceleration: Sequence[Any]
    nominal_matrix: Sequence[Sequence[Any]]
    nominal_affine: Sequence[Any]
    reference_id: str

    def __post_init__(self):
        object.__setattr__(self, "anomaly", _q(self.anomaly))
        object.__setattr__(self, "state", _vector(self.state, 6, "reference state"))
        object.__setattr__(self, "acceleration", _vector(self.acceleration, 3, "reference acceleration"))
        object.__setattr__(self, "nominal_matrix", _matrix(self.nominal_matrix, "nominal matrix"))
        object.__setattr__(self, "nominal_affine", _vector(self.nominal_affine, 3, "nominal affine"))
        if not self.reference_id.strip():
            raise ValueError("An authoritative reference identity is required")

    @property
    def feedforward(self):
        drift = _dot(self.nominal_matrix, self.state)
        return tuple(a - d - f for a, d, f in zip(self.acceleration, drift, self.nominal_affine))

    def complies(self, bounds: TerminalTrackingBounds) -> bool:
        return (self.reference_id == bounds.reference_id and bounds.scope.contains(self.anomaly)
                and all(abs(v) <= lim for v, lim in zip(self.state, bounds.reference_state_abs))
                and all(abs(v) <= lim for row, row_lim in zip(self.nominal_matrix, bounds.nominal_matrix_abs)
                        for v, lim in zip(row, row_lim))
                and bounds.feedforward_range.contains(IntervalBox.point(self.feedforward)))

    def as_dict(self):
        return _encode(vars(self))


@dataclass(frozen=True)
class TerminalHoldReferences:
    """Two canonical reference events and their actual evaluation query clocks.

    The feedback sample belongs to the actual observation/flow start.  Only
    feedforward is previewed. Accuracy of the evaluated values is a separate
    declared contract, not established by these clock consistency checks.
    """

    policy: TerminalFeedforwardPolicy
    feedback: TerminalReferenceSample
    feedforward: TerminalReferenceSample
    feedback_query_anomaly: Any | None = None
    feedforward_query_anomaly: Any | None = None
    maximum_query_displacement: Any = 0

    def __post_init__(self):
        object.__setattr__(self, "policy", TerminalFeedforwardPolicy(self.policy))
        if not isinstance(self.feedback, TerminalReferenceSample) or not isinstance(self.feedforward, TerminalReferenceSample):
            raise TypeError("Both hold references must be typed reference samples")
        if self.feedback.reference_id != self.feedforward.reference_id:
            raise ValueError("Both hold references must name the same authoritative reference")
        for name, sample in (("feedback_query_anomaly", self.feedback), ("feedforward_query_anomaly", self.feedforward)):
            value = getattr(self, name)
            object.__setattr__(self, name, sample.anomaly if value is None else _q(value))
        budget = _q(self.maximum_query_displacement)
        if budget < 0:
            raise ValueError("Reference query displacement must be nonnegative")
        object.__setattr__(self, "maximum_query_displacement", budget)

    def validate(self, bounds, hold_width, *, maximum_query_displacement=0):
        hold, budget = _q(hold_width), _q(maximum_query_displacement)
        if (self.maximum_query_displacement != budget or not 0 < hold <= bounds.maximum_hold
                or not bounds.scope.contains_hold(self.feedback.anomaly, hold)
                or self.feedforward.anomaly != self.feedback.anomaly + self.policy.sample_fraction * hold
                or self.feedback.reference_id != bounds.reference_id):
            raise ValueError("Held policy, reference identity, exact clocks, width or query contract changed")
        if self.policy is TerminalFeedforwardPolicy.START_V1 and self.feedback != self.feedforward:
            raise ValueError("Start feedforward must use the same reference evaluation as feedback")
        if (abs(self.feedback_query_anomaly - self.feedback.anomaly) > budget
                or abs(self.feedforward_query_anomaly - self.feedforward.anomaly) > budget):
            raise ValueError("A reference query clock exceeds its declared evaluation budget")
        return True

    def as_dict(self):
        return dict(held_policy=self.policy.value, feedback=self.feedback.as_dict(),
            feedforward=self.feedforward.as_dict(), feedback_query_anomaly=str(self.feedback_query_anomaly),
            feedforward_query_anomaly=str(self.feedforward_query_anomaly),
            maximum_query_displacement=str(self.maximum_query_displacement),
            feedback_query_displacement=str(abs(self.feedback_query_anomaly - self.feedback.anomaly)),
            feedforward_query_displacement=str(abs(self.feedforward_query_anomaly - self.feedforward.anomaly)))


@dataclass(frozen=True)
class TerminalControlCommand:
    anomaly: Fraction
    hold_width: Fraction
    exact_command: tuple[Fraction, ...]
    applied_input: IntervalBox
    actuator_error_abs: tuple[Fraction, ...]
    reference_id: str
    hold_references: TerminalHoldReferences | None = None

    def float_command(self) -> tuple[float, ...]:
        """Round only when the declared actuator-error budget covers it."""
        result = tuple(map(float, self.exact_command))
        if any(abs(_q(value) - exact) > allowance for value, exact, allowance in
               zip(result, self.exact_command, self.actuator_error_abs)):
            raise ValueError("Binary64 command conversion exceeds the declared actuator-error budget")
        return result

    def as_dict(self):
        record = {"anomaly": str(self.anomaly), "hold_width": str(self.hold_width),
                "exact_command": tuple(map(str, self.exact_command)),
                "applied_input": self.applied_input.as_dict(),
                "actuator_error_abs": tuple(map(str, self.actuator_error_abs)),
                "reference_id": self.reference_id}
        if self.hold_references is not None:
            record.update(held_policy=self.hold_references.policy.value,
                end_anomaly=str(self.anomaly + self.hold_width), reference_samples=self.hold_references.as_dict())
        return record


class TerminalTrackingController:
    """Held feedforward plus bounded sliding feedback inside a certified set."""

    def __init__(self, certificate: TerminalCertificate):
        # Recompute instead of trusting a hand-constructed/stored status flag.
        if (not isinstance(certificate, TerminalCertificate)
                or certificate.method != TerminalFeedforwardPolicy(certificate.policy).certificate_method):
            raise ValueError("Unsupported or inconsistent terminal certificate policy")
        self.certificate = certify_terminal_tracking(certificate.bounds, policy=certificate.policy)
        if not self.certificate.admissible:
            raise ValueError("Terminal certificate refused: " + "; ".join(self.certificate.reasons))

    def command(self, measured_state, state_enclosure: IntervalBox,
                reference: TerminalReferenceSample, *, hold_width,
                hold_references: TerminalHoldReferences | None = None) -> TerminalControlCommand:
        b = self.certificate.bounds
        hold = _q(hold_width)
        measurement = _vector(measured_state, 6, "measurement")
        policy = self.certificate.policy
        if hold_references is None:
            if policy is not TerminalFeedforwardPolicy.START_V1:
                raise ValueError("Midpoint feedforward requires both exact held-reference events")
        elif (not isinstance(hold_references, TerminalHoldReferences)
              or hold_references.policy is not policy or hold_references.feedback != reference):
            raise ValueError("The held-reference policy or actual feedback sample changed")
        else:
            hold_references.validate(b, hold)
        feedforward = reference if hold_references is None else hold_references.feedforward
        if not 0 < hold <= b.maximum_hold or not b.scope.contains_hold(reference.anomaly, hold):
            raise ValueError("The complete feedback hold must belong to the certified scope")
        if not self.certificate.contains(state_enclosure, reference):
            raise ValueError("Current state enclosure is outside the certified terminal set")
        if not feedforward.complies(b):
            raise ValueError("The held feedforward sample violates its global reference contract")
        if any(max(abs(lo - m), abs(hi - m)) > n for lo, hi, m, n in
               zip(state_enclosure.lower, state_enclosure.upper, measurement, b.sensing_error_abs)):
            raise ValueError("Measurement and state enclosure violate the sensing-error budget")
        error = tuple(x - g for x, g in zip(measurement, reference.state))
        sliding = _sum(error[3:], _mul(b.lambda_gain, error[:3]))
        saturation = tuple(max(Fraction(-1), min(Fraction(1), s / p)) for s, p in zip(sliding, b.phi))
        robust = _sum(_dot(b.matrix_uncertainty_abs, tuple(map(abs, error))), _mul(b.bandwidth, b.phi))
        cancellation = _sum(_dot(reference.nominal_matrix, error), _mul(b.lambda_gain, error[3:]),
                            _mul(robust, saturation))
        command = tuple(g - c for g, c in zip(feedforward.feedforward, cancellation))
        applied = IntervalBox.from_center_radius(command, b.actuator_error_abs)
        if not b.input_limits.contains(applied):
            raise ValueError("Command exceeds certified input limits")
        return TerminalControlCommand(reference.anomaly, hold, command, applied,
            b.actuator_error_abs, reference.reference_id,
            hold_references if policy is not TerminalFeedforwardPolicy.START_V1 else None)
