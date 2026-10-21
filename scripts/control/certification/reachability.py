"""Validated continuous held-input flow and swept collision certificates.

This module encloses the *continuous ODE*, not an RK4 map.  Its optional
``python-flint`` dependency supplies Arb ball arithmetic; no binary64
transcendental evaluation participates in a proof.  See
``docs/continuous-safety.md`` for the theorem, units and conditional scope.

Only ``PROVEN`` flows and ``PROVEN_SAFE`` collision decisions permit execution.
An unresolved enclosure, missing dependency or nonseparating box never does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from numbers import Integral
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Sequence
import math
import hashlib
import json


_ARB_LOCK = RLock()
FRAME = "physical-lvlh-v1"
METHOD = "arb-picard-continuous-v1"
# Optional source authentication must not make legacy imports require a source
# file. The separate orbital-hold service refuses an unavailable identity.
try:
    _LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
except OSError:
    _LOADED_SOURCE_SHA256 = None


def _exact(value: Any) -> Fraction:
    """Interpret floats as their exact binary rational, strings as rationals."""
    if isinstance(value, bool):
        raise ValueError("Boolean values are not numerical bounds")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, str):
        return Fraction(value)
    # as_integer_ratio also supports NumPy binary floating scalars without
    # silently converting them to a possibly narrower Python float first.
    if hasattr(value, "as_integer_ratio"):
        numerator, denominator = value.as_integer_ratio()
        return Fraction(int(numerator), int(denominator))
    raise TypeError("Use finite binary floats, integers, Fraction, or rational strings")


def _pair(values: Sequence[Any], name: str) -> tuple[Fraction, Fraction]:
    if len(values) != 2:
        raise ValueError(f"{name} needs lower and upper bounds")
    lo, hi = map(_exact, values)
    if lo > hi:
        raise ValueError(f"{name} lower bound exceeds upper bound")
    return lo, hi


def _float_outward(value: Fraction, upward: bool) -> float:
    """Export only: check the rounded float against the exact rational."""
    try:
        rounded = float(value)
    except OverflowError:
        rounded = math.inf if value > 0 else -math.inf
    if math.isinf(rounded):
        if (rounded > 0) == upward:
            return rounded
        return math.nextafter(rounded, -math.inf if upward is False else math.inf)
    if (Fraction.from_float(rounded) < value if upward else
            Fraction.from_float(rounded) > value):
        rounded = math.nextafter(rounded, math.inf if upward else -math.inf)
    return rounded


@dataclass(frozen=True)
class IntervalBox:
    """Closed componentwise box with exact rational endpoints.

    ``float_bounds`` are outward exports; proof decisions use rational bounds
    or Arb predicates directly.  No caller must recover bounds from a printed
    midpoint, decimal display, or a midpoint-only conversion.
    """

    lower: tuple[Any, ...]
    upper: tuple[Any, ...]

    def __post_init__(self):
        lo, hi = tuple(map(_exact, self.lower)), tuple(map(_exact, self.upper))
        if not lo or len(lo) != len(hi) or any(a > b for a, b in zip(lo, hi)):
            raise ValueError("A box requires equally sized nonempty ordered endpoints")
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)

    @classmethod
    def point(cls, values: Iterable[Any]) -> IntervalBox:
        values = tuple(values)
        return cls(values, values)

    @classmethod
    def from_center_radius(cls, center, radius) -> IntervalBox:
        center, radius = tuple(map(_exact, center)), tuple(map(_exact, radius))
        if len(center) != len(radius) or any(r < 0 for r in radius):
            raise ValueError("Center and nonnegative radius dimensions must agree")
        return cls(tuple(c - r for c, r in zip(center, radius)),
                   tuple(c + r for c, r in zip(center, radius)))

    @property
    def dimension(self) -> int:
        return len(self.lower)

    def contains(self, other: IntervalBox) -> bool:
        return (self.dimension == other.dimension and
                all(a <= c and d <= b for a, b, c, d in
                    zip(self.lower, self.upper, other.lower, other.upper)))

    def hull(self, other: IntervalBox) -> IntervalBox:
        if self.dimension != other.dimension:
            raise ValueError("Cannot hull different box dimensions")
        return IntervalBox(tuple(map(min, self.lower, other.lower)),
                           tuple(map(max, self.upper, other.upper)))

    def slice(self, start: int, end: int) -> IntervalBox:
        return IntervalBox(self.lower[start:end], self.upper[start:end])

    def float_bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return (tuple(_float_outward(v, False) for v in self.lower),
                tuple(_float_outward(v, True) for v in self.upper))

    def as_dict(self) -> dict:
        # Repeated certificate fingerprints export the same immutable boxes.
        # Cache only derived immutable tuples, with copied exact primitive
        # endpoints as the guard. Even a caller bypassing the frozen dataclass
        # or mutating a Fraction cannot hide changed bounds behind this cache.
        key = (tuple((v.numerator, v.denominator) for v in self.lower),
               tuple((v.numerator, v.denominator) for v in self.upper))
        cached = getattr(self, "_export_cache", None)
        if cached is None or cached[0] != key:
            lo, hi = self.float_bounds()
            values = (lo, hi, tuple(map(str, self.lower)), tuple(map(str, self.upper)))
            cached = (key, values)
            object.__setattr__(self, "_export_cache", cached)
        lo, hi, exact_lo, exact_hi = cached[1]
        # Callers own their dictionary; changing it never changes a later export.
        return {"lower": lo, "upper": hi, "exact_lower": exact_lo, "exact_upper": exact_hi}


@dataclass(frozen=True)
class OrbitalModelBounds:
    """Full parameter domain for the maintained analytical orbital model.

    The semi-major axis is fixed and authoritative: h(e)^2=mu*a*(1-e^2).
    Drag constants are dimensionless; phase and anomaly use radians.
    """

    eccentricity: tuple[Any, Any] = (0, 0)
    alpha: tuple[Any, Any] = (0, 0)
    beta: tuple[Any, Any] = (0, 0)
    semi_major_axis: Any = 7_000_000
    mu: Any = "398600441800000"
    theta0: tuple[Any, Any] = (0, 0)

    def __post_init__(self):
        for name in ("eccentricity", "alpha", "beta", "theta0"):
            object.__setattr__(self, name, _pair(getattr(self, name), name))
        for name in ("semi_major_axis", "mu"):
            value = _exact(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if not 0 <= self.eccentricity[0] <= self.eccentricity[1] < 1:
            raise ValueError("Elliptical eccentricity bounds must lie in [0,1)")
        if self.alpha[0] < 0 or self.beta[0] < 0:
            raise ValueError("Drag constants must be nonnegative")

    def as_dict(self) -> dict:
        return {name: (tuple(map(str, value)) if isinstance(value, tuple) else str(value))
                for name, value in vars(self).items()}


@dataclass(frozen=True)
class HeldInputRequest:
    """A single six-state hold with declared clock and measurement boxes.

    Initial state is [p,dp/dq] (m, m/rad). Input and additional disturbance
    are anomaly accelerations (m/rad^2). ``start_anomaly`` encloses the true
    start clock; ``hold_width`` encloses the actual nonnegative duration in
    anomaly, including delay/jitter. A variable stop time is enclosed too.
    """

    initial_state: IntervalBox
    held_input: IntervalBox
    start_anomaly: tuple[Any, Any]
    hold_width: tuple[Any, Any]
    acceleration_disturbance: IntervalBox = field(
        default_factory=lambda: IntervalBox.point((0, 0, 0)))

    def __post_init__(self):
        if (self.initial_state.dimension != 6 or self.held_input.dimension != 3
                or self.acceleration_disturbance.dimension != 3):
            raise ValueError("Orbital state/input/disturbance dimensions are 6/3/3")
        object.__setattr__(self, "start_anomaly", _pair(self.start_anomaly, "start_anomaly"))
        hold = _pair(self.hold_width, "hold_width")
        if hold[0] < 0:
            raise ValueError("Hold widths must be nonnegative")
        object.__setattr__(self, "hold_width", hold)

    def as_dict(self) -> dict:
        return {"initial_state": self.initial_state.as_dict(),
                "held_input": self.held_input.as_dict(),
                "acceleration_disturbance": self.acceleration_disturbance.as_dict(),
                "start_anomaly": tuple(map(str, self.start_anomaly)),
                "hold_width": tuple(map(str, self.hold_width))}


@dataclass(frozen=True)
class ReachabilityConfig:
    precision_bits: int = 128
    max_step: Any = "0.002"
    min_step: Any = "1e-12"
    max_substeps: int = 4096
    max_picard_iterations: int = 16
    inflation_fraction: Any = "1/8"
    absolute_padding: Any = "1e-24"

    def __post_init__(self):
        for name in ("precision_bits", "max_substeps", "max_picard_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.precision_bits < 64:
            raise ValueError("Use at least 64 working precision bits")
        for name in ("max_step", "min_step", "inflation_fraction", "absolute_padding"):
            value = _exact(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.min_step > self.max_step:
            raise ValueError("min_step must not exceed max_step")


class EnclosureStatus(str, Enum):
    PROVEN = "PROVEN"
    UNRESOLVED = "UNRESOLVED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class SweptSegment:
    offset: tuple[Fraction, Fraction]
    anomaly: tuple[Fraction, Fraction]
    initial_state: IntervalBox
    swept_state: IntervalBox
    endpoint: IntervalBox
    picard_image: IntervalBox
    derivative: IntervalBox
    picard_iterations: int
    anomaly_rate: IntervalBox | None = None

    def as_dict(self) -> dict:
        return {"offset": tuple(map(str, self.offset)),
                "anomaly": tuple(map(str, self.anomaly)),
                "initial_state": self.initial_state.as_dict(),
                "swept_state": self.swept_state.as_dict(),
                "endpoint": self.endpoint.as_dict(),
                "picard_image": self.picard_image.as_dict(),
                "derivative": self.derivative.as_dict(),
                "picard_iterations": self.picard_iterations,
                "anomaly_rate": self.anomaly_rate.as_dict() if self.anomaly_rate else None}


@dataclass(frozen=True)
class FlowCertificate:
    status: EnclosureStatus
    reason: str
    start_anomaly: tuple[Fraction, Fraction]
    hold_width: tuple[Fraction, Fraction]
    initial_state: IntervalBox
    endpoint: IntervalBox | None = None
    swept_state: IntervalBox | None = None
    segments: tuple[SweptSegment, ...] = ()
    precision_bits: int = 0
    backend_version: str = ""
    frame: str = FRAME
    model: str = ""
    model_bounds: dict = field(default_factory=dict)
    request: dict = field(default_factory=dict)
    swept_physical_velocity: IntervalBox | None = None
    elapsed_seconds: IntervalBox | None = None
    method: str = METHOD

    @property
    def proven(self) -> bool:
        return self.status == EnclosureStatus.PROVEN

    @property
    def fingerprint(self) -> str:
        """Bind cached geometry to this complete proved flow and request."""
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()

    @property
    def swept_position(self) -> IntervalBox | None:
        return self.swept_state.slice(0, 3) if self.swept_state else None

    @property
    def swept_anomaly_velocity(self) -> IntervalBox | None:
        return self.swept_state.slice(3, 6) if self.swept_state and self.swept_state.dimension == 6 else None

    def as_dict(self, *, include_segments: bool = True) -> dict:
        result = {"status": self.status.value, "reason": self.reason,
                  "method": self.method, "precision_bits": self.precision_bits,
                  "backend_version": self.backend_version, "frame": self.frame,
                  "model": self.model, "model_bounds": self.model_bounds,
                  "request": self.request,
                  "start_anomaly": tuple(map(str, self.start_anomaly)),
                  "hold_width": tuple(map(str, self.hold_width)),
                  "initial_state": self.initial_state.as_dict(),
                  "endpoint": self.endpoint.as_dict() if self.endpoint else None,
                  "swept_state": self.swept_state.as_dict() if self.swept_state else None,
                  "swept_physical_velocity": self.swept_physical_velocity.as_dict() if self.swept_physical_velocity else None,
                  "elapsed_seconds": self.elapsed_seconds.as_dict() if self.elapsed_seconds else None,
                  "substeps": len(self.segments)}
        if include_segments:
            result["segments"] = tuple(segment.as_dict() for segment in self.segments)
        return result


class _ProofUnresolved(Exception):
    pass


class _ArbArithmetic:
    def __init__(self, flint):
        self.arb = flint.arb

    def scalar(self, value):
        value = _exact(value)
        return self.arb(value.numerator) / self.arb(value.denominator)

    def interval(self, bounds):
        lo, hi = bounds
        return self.scalar(lo).union(self.scalar(hi)) if lo != hi else self.scalar(lo)

    def vector(self, box):
        return tuple(self.interval(bounds) for bounds in zip(box.lower, box.upper))

    @staticmethod
    def exact_point(ball):
        mantissa, exponent = ball.man_exp()
        exponent = int(exponent)
        return Fraction(int(mantissa)) * (Fraction(2) ** exponent)

    def box(self, balls):
        if not all(ball.is_finite() for ball in balls):
            raise _ProofUnresolved("Arb returned a nonfinite enclosure")
        return IntervalBox(tuple(self.exact_point(v.lower()) for v in balls),
                           tuple(self.exact_point(v.upper()) for v in balls))

    def inflate(self, balls, config):
        fraction, padding = self.scalar(config.inflation_fraction), self.scalar(config.absolute_padding)
        return tuple(v + self.arb(0, (v.rad() * fraction + padding).upper()) for v in balls)


def _orbital_coefficients(q, parameters: OrbitalModelBounds, ar):
    """Shared Arb coefficients of the supported analytical orbital model.

    Input and additional acceleration are deliberately separate from this
    state matrix, affine drift and physical anomaly rate.
    """
    e, alpha, beta, phase = (ar.interval(getattr(parameters, name))
                              for name in ("eccentricity", "alpha", "beta", "theta0"))
    mu, axis = ar.scalar(parameters.mu), ar.scalar(parameters.semi_major_axis)
    h2 = mu * axis * (1 - e * e)
    exponential = (2 * alpha * q).exp()
    denominator = exponential + e * (q - phase).cos()
    if not denominator > 0 or not h2 > 0:
        raise _ProofUnresolved("Orbital denominator or h(e)^2 is not proved positive")
    scale = 1 + 4 * alpha * alpha
    radius = h2 * scale / (mu * denominator)
    k = (e * (q - phase).sin() - 2 * alpha * exponential) / denominator
    curvature = scale * exponential / denominator
    gamma, damping = beta - alpha, 2 * k - (beta - alpha)
    zero, one = ar.arb(0), ar.arb(1)
    matrix = ((zero, zero, zero, one, zero, zero),
              (zero, zero, zero, zero, one, zero),
              (zero, zero, zero, zero, zero, one),
              (1 - curvature, -damping, zero, damping, 2 * one, zero),
              (damping, 1 + 2 * curvature, zero, -2 * one, damping, zero),
              (zero, zero, -curvature, zero, zero, damping))
    drift = (gamma * radius, -gamma * radius * k, zero)
    rate = h2.sqrt() * (-alpha * q).exp() / (radius * radius)
    if not rate > 0:
        raise _ProofUnresolved("dq/dt is not proved positive")
    return matrix, (zero, zero, zero) + drift, rate


class ValidatedFlowEncloser:
    """Finite-budget interval Picard integration with strict self-containment.

    Arb is imported only at the proof boundary. A missing library returns
    UNAVAILABLE. Domain failure, substep exhaustion and failed self-inclusion
    return UNRESOLVED. Unexpected programming errors propagate to the caller.
    """

    def __init__(self, config: ReachabilityConfig | None = None):
        self.config = config or ReachabilityConfig()

    def enclose_orbital(self, request: HeldInputRequest,
                        parameters: OrbitalModelBounds) -> FlowCertificate:
        def coefficients(q, ar):
            matrix, affine, rate = _orbital_coefficients(q, parameters, ar)
            control, disturbance = ar.vector(request.held_input), ar.vector(request.acceleration_disturbance)
            forcing = affine[:3] + tuple(
                d + u + w for d, u, w in zip(affine[3:], control, disturbance))
            return matrix, forcing, rate

        return self._enclose(request.initial_state, request.start_anomaly, request.hold_width,
                             coefficients, FRAME, "physical-lvlh-v1",
                             parameters.as_dict(), request.as_dict())

    def enclose_affine(self, matrix: Sequence[Sequence[Any]], initial_state: IntervalBox,
                       affine: IntervalBox, *, start_anomaly=(0, 0), hold_width=(0, 0),
                       frame: str = FRAME) -> FlowCertificate:
        """Validated constant affine differential inclusion x' in A*x+b.

        Matrix entries are exact numbers. ``affine`` may enclose a bounded,
        time-varying additive forcing. Useful for exact limiting cases and
        other affine plants; no unvalidated callback can masquerade as Arb.
        """
        matrix = tuple(tuple(map(_exact, row)) for row in matrix)
        dimension = initial_state.dimension
        if len(matrix) != dimension or any(len(row) != dimension for row in matrix) or affine.dimension != dimension:
            raise ValueError("Affine matrix, initial state and forcing dimensions must agree")
        start, hold = _pair(start_anomaly, "start_anomaly"), _pair(hold_width, "hold_width")
        if hold[0] < 0 or not frame:
            raise ValueError("Nonnegative hold and explicit coordinate frame are required")

        def coefficients(q, ar):
            return (tuple(tuple(ar.scalar(v) for v in row) for row in matrix),
                    ar.vector(affine), None)

        return self._enclose(initial_state, start, hold, coefficients, frame,
                             "constant-affine-inclusion", {"matrix": tuple(tuple(map(str, row)) for row in matrix),
                             "affine": affine.as_dict()}, {})

    def _enclose(self, initial, start, hold, coefficients, frame, model, bounds, request):
        common = dict(start_anomaly=start, hold_width=hold, initial_state=initial,
                      precision_bits=self.config.precision_bits, frame=frame,
                      model=model, model_bounds=bounds, request=request)
        try:
            import flint
        except ImportError as error:
            return FlowCertificate(EnclosureStatus.UNAVAILABLE,
                                   f"python-flint is required: {error}", **common)
        common["backend_version"] = getattr(flint, "__version__", "unknown")
        with _ARB_LOCK, flint.ctx.workprec(self.config.precision_bits):
            ar = _ArbArithmetic(flint)
            try:
                return self._integrate(initial, start, hold, coefficients, ar, common)
            except _ProofUnresolved as error:
                return FlowCertificate(EnclosureStatus.UNRESOLVED, str(error), **common)

    def _integrate(self, initial, start, hold, coefficients, ar, common):
        config, final_time = self.config, hold[1]
        if final_time == 0:
            return FlowCertificate(EnclosureStatus.PROVEN, "Zero-width hold: exact initial box",
                                   endpoint=initial, swept_state=initial, **common)
        current, offset, proposed = ar.vector(initial), Fraction(0), min(config.max_step, final_time)
        segments, swept, terminal = [], None, None
        velocity, elapsed_upper = None, ar.arb(0)
        while offset < final_time:
            if len(segments) >= config.max_substeps:
                raise _ProofUnresolved("Continuous enclosure exhausted max_substeps")
            # Partition the remaining *exact* rational duration evenly. Repeated
            # decimal max_step subtraction from a binary float hold can leave a
            # tiny final tail below min_step, even for a zero vector field.
            remaining = final_time - offset
            ratio = remaining / proposed
            pieces = (ratio.numerator + ratio.denominator - 1) // ratio.denominator
            width = remaining / pieces
            while True:
                if width < config.min_step:
                    raise _ProofUnresolved("Continuous enclosure needs a substep below min_step")
                q_bounds = start[0] + offset, start[1] + offset + width
                q = ar.interval(q_bounds)
                try:
                    matrix, forcing, rate = coefficients(q, ar)
                    if not all(v.is_finite() for row in matrix for v in row) or not all(v.is_finite() for v in forcing):
                        raise _ProofUnresolved("Nonfinite affine coefficient enclosure")
                    witness = self._picard(current, matrix, forcing, width, ar)
                except _ProofUnresolved:
                    witness = None
                if witness is not None:
                    break
                width /= 2
            candidate, image, derivative, endpoint, iterations = witness
            initial_box, candidate_box = ar.box(current), ar.box(candidate)
            endpoint_box = ar.box(endpoint)
            segment = SweptSegment((offset, offset + width), q_bounds, initial_box,
                                   candidate_box, endpoint_box, ar.box(image),
                                   ar.box(derivative), iterations,
                                   ar.box((rate,)) if rate is not None else None)
            segments.append(segment)
            swept = candidate_box if swept is None else swept.hull(candidate_box)
            # An uncertain stop time may be anywhere in [hold[0],hold[1]].
            if hold[0] != hold[1] and offset + width >= hold[0]:
                stop_offsets = ar.interval((max(Fraction(0), hold[0] - offset), width))
                stop_box = ar.box(tuple(x + stop_offsets * f for x, f in zip(current, derivative)))
                terminal = stop_box if terminal is None else terminal.hull(stop_box)
            if rate is not None:
                physical_velocity = ar.box(tuple(v * rate for v in candidate[3:6]))
                velocity = physical_velocity if velocity is None else velocity.hull(physical_velocity)
                elapsed_upper += ar.scalar(width) / rate
            current, offset = endpoint, offset + width
            proposed = min(config.max_step, 2 * width)
        if hold[0] == hold[1]:
            terminal = endpoint_box
        elapsed = None if velocity is None else IntervalBox((0,), (ar.exact_point(elapsed_upper.upper()),))
        return FlowCertificate(EnclosureStatus.PROVEN,
                               "Every substep satisfies strict interval Picard self-inclusion",
                               endpoint=terminal, swept_state=swept, segments=tuple(segments),
                               swept_physical_velocity=velocity, elapsed_seconds=elapsed, **common)

    def _picard(self, initial, matrix, forcing, width, ar):
        h, offsets = ar.scalar(width), ar.interval((Fraction(0), width))

        def derivative(state):
            return tuple(sum((a * x for a, x in zip(row, state)), b)
                         for row, b in zip(matrix, forcing))

        first = derivative(initial)
        candidate = ar.inflate(tuple(x.union(x + offsets * f) for x, f in zip(initial, first)), self.config)
        for iteration in range(1, self.config.max_picard_iterations + 1):
            drift = derivative(candidate)
            image = tuple(x + offsets * f for x, f in zip(initial, drift))
            if not all(v.is_finite() for v in image):
                return None
            if all(y.contains_interior(p) for y, p in zip(candidate, image)):
                endpoint = tuple(x + h * f for x, f in zip(initial, drift))
                return candidate, image, drift, endpoint, iteration
            candidate = ar.inflate(tuple(y.union(p) for y, p in zip(candidate, image)), self.config)
        return None


@dataclass(frozen=True)
class RotatingCylinderBounds:
    """Occupied solid cylinder, bounded pose at hold start and motion rates.

    Orientation is an XYZ Euler-angle box for active Rx*Ry*Rz, or a matrix M
    with a declared operator-norm error ||M-R_start||_2. The matrix entries
    are exact rationals; a computed floating rotation usually needs positive
    error. This bound must enclose sensing/integration error independently.
    The rate bounds *relative angular velocity per anomaly*, not an individual
    Euler rate. Center speed bounds translation in this common frame. Radius
    and half-length are conservative physical upper bounds in metres.
    """

    radius: Any
    half_length: Any
    orientation_at_start: IntervalBox = field(default_factory=lambda: IntervalBox.point((0, 0, 0)))
    center_at_start: IntervalBox = field(default_factory=lambda: IntervalBox.point((0, 0, 0)))
    angular_rate_per_anomaly: Any = 0
    center_speed_per_anomaly: Any = 0
    orientation_matrix_at_start: tuple[tuple[Any, ...], ...] | None = None
    orientation_matrix_error: Any = 0
    frame: str = FRAME
    scope: str = "Declared geometry and pose/rate bounds over this complete hold"

    def __post_init__(self):
        if self.orientation_at_start.dimension != 3 or self.center_at_start.dimension != 3:
            raise ValueError("Cylinder orientation and center boxes have dimension 3")
        for name in ("radius", "half_length", "angular_rate_per_anomaly", "center_speed_per_anomaly", "orientation_matrix_error"):
            value = _exact(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if self.orientation_matrix_at_start is not None:
            matrix = tuple(tuple(map(_exact, row)) for row in self.orientation_matrix_at_start)
            if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
                raise ValueError("Orientation matrix must have shape (3,3)")
            if self.orientation_at_start != IntervalBox.point((0, 0, 0)):
                raise ValueError("Supply either Euler-angle bounds or a matrix with an error bound")
            # Necessary compatibility conditions for existence of a proper R
            # satisfying ||M-R||_2 <= error. The bound itself remains an
            # explicit pose assumption, not an estimate of sensor/ODE error.
            error = self.orientation_matrix_error
            limit = 2 * error + error * error
            for i in range(3):
                for j in range(3):
                    gram = sum(matrix[k][i] * matrix[k][j] for k in range(3))
                    if abs(gram - int(i == j)) > limit:
                        raise ValueError("Orientation matrix is incompatible with its declared error")
            determinant = (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
                           - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
                           + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))
            if error < 1 and determinant <= 0:
                raise ValueError("Orientation matrix cannot be close to a proper rotation")
            object.__setattr__(self, "orientation_matrix_at_start", matrix)
        elif self.orientation_matrix_error != 0:
            raise ValueError("A matrix error bound requires an orientation matrix")
        if not self.frame or not self.scope:
            raise ValueError("Explicit geometry frame and bound scope are required")

    def as_dict(self) -> dict:
        return {"radius": str(self.radius), "half_length": str(self.half_length),
                "orientation_at_start": self.orientation_at_start.as_dict(),
                "center_at_start": self.center_at_start.as_dict(),
                "angular_rate_per_anomaly": str(self.angular_rate_per_anomaly),
                "center_speed_per_anomaly": str(self.center_speed_per_anomaly),
                "orientation_matrix_at_start": (tuple(tuple(map(str, row)) for row in self.orientation_matrix_at_start)
                                                 if self.orientation_matrix_at_start is not None else None),
                "orientation_matrix_error": str(self.orientation_matrix_error),
                "frame": self.frame, "scope": self.scope}


class GuardStatus(str, Enum):
    PROVEN_SAFE = "PROVEN_SAFE"
    UNRESOLVED = "UNRESOLVED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class CollisionCertificate:
    status: GuardStatus
    reason: str
    evidence: tuple[dict, ...] = ()
    target_bounds: dict | None = None
    body_radii: tuple[str, ...] = ()
    clearance: str = "0"
    minimum_pair_center_distance: str = "0"
    method: str = "arb-swept-separation-v1"
    flow_fingerprints: tuple[str, ...] = ()

    @property
    def admissible(self) -> bool:
        return self.status == GuardStatus.PROVEN_SAFE

    def as_dict(self) -> dict:
        return {"status": self.status.value, "reason": self.reason,
                "method": self.method, "evidence": self.evidence,
                "target_bounds": self.target_bounds, "body_radii": self.body_radii,
                "clearance": self.clearance,
                "minimum_pair_center_distance": self.minimum_pair_center_distance,
                "flow_fingerprints": self.flow_fingerprints}


class SweptCollisionGuard:
    """Separate every swept position box from target and other finite bodies.

    Chasers are enclosed by physical balls. A failed separating-plane search
    means UNRESOLVED, not a claim of collision. Pair separation uses exact
    rational box distance and handles different integration partitions.
    """

    def __init__(self, *, precision_bits: int = 128):
        if isinstance(precision_bits, bool) or not isinstance(precision_bits, Integral) or precision_bits < 64:
            raise ValueError("Use an integer precision_bits >= 64")
        self.precision_bits = precision_bits

    def check(self, flows: Sequence[FlowCertificate], body_radii: Sequence[Any], *,
              target: RotatingCylinderBounds | None = None, clearance: Any = 0,
              minimum_pair_center_distance: Any = 0,
              normals: Sequence[Sequence[Any]] = ()) -> CollisionCertificate:
        flows, radii, clearance = tuple(flows), tuple(map(_exact, body_radii)), _exact(clearance)
        pair_distance = _exact(minimum_pair_center_distance)
        if not flows or len(flows) != len(radii) or any(r < 0 for r in radii) or clearance < 0 or pair_distance < 0:
            raise ValueError("A nonnegative physical radius is required per flow, with nonnegative clearance")
        common = dict(target_bounds=target.as_dict() if target else None,
                      body_radii=tuple(map(str, radii)), clearance=str(clearance),
                      minimum_pair_center_distance=str(pair_distance))
        if any(not isinstance(flow, FlowCertificate) for flow in flows):
            return CollisionCertificate(GuardStatus.UNRESOLVED, "Every flow must be a FlowCertificate", **common)
        if any(not isinstance(flow.status, EnclosureStatus) or not flow.proven for flow in flows):
            return CollisionCertificate(GuardStatus.UNRESOLVED, "Every flow must first be PROVEN", **common)
        try:
            for flow in flows:
                self._check_flow_structure(flow)
            common['flow_fingerprints'] = tuple(flow.fingerprint for flow in flows)
        except (ValueError, TypeError, OverflowError, AttributeError, RecursionError) as error:
            # Cached evidence with missing coverage, nonfinite metadata or an
            # overflowing export cannot grant admission. In particular, an
            # unresolved flow must not fail while merely hashing its metadata.
            return CollisionCertificate(GuardStatus.UNRESOLVED,
                f"Flow identity or continuous coverage cannot be validated: {type(error).__name__}: {error}", **common)
        first = flows[0]
        if any(flow.initial_state.dimension != 6 or flow.frame != first.frame or
               flow.start_anomaly != first.start_anomaly or flow.hold_width != first.hold_width
               for flow in flows) or (target and target.frame != first.frame):
            return CollisionCertificate(GuardStatus.UNRESOLVED,
                                        "Incompatible state dimensions, frames, or common clock bounds", **common)
        if target is None and len(flows) == 1:
            return CollisionCertificate(GuardStatus.UNRESOLVED, "No occupied geometry was supplied", **common)
        fixed_normals = tuple(tuple(map(_exact, n)) for n in normals)
        if any(len(n) != 3 or not any(n) for n in fixed_normals):
            raise ValueError("Separating normals must be nonzero three-vectors")
        try:
            import flint
        except ImportError as error:
            return CollisionCertificate(GuardStatus.UNAVAILABLE, f"python-flint is required: {error}", **common)
        with _ARB_LOCK, flint.ctx.workprec(self.precision_bits):
            ar, evidence = _ArbArithmetic(flint), []
            try:
                for index, (flow, radius) in enumerate(zip(flows, radii)):
                    if target:
                        for segment in self._segments(flow):
                            proof = self._target_segment(segment, radius + clearance, target, fixed_normals, ar)
                            if proof is None:
                                return CollisionCertificate(GuardStatus.UNRESOLVED,
                                    f"No target separation proved for chaser {index}, offset {segment.offset}", tuple(evidence), **common)
                            evidence.append({"kind": "target", "chaser": index, **proof})
                for i in range(len(flows)):
                    for j in range(i + 1, len(flows)):
                        for left in self._segments(flows[i]):
                            for right in self._segments(flows[j]):
                                overlap = max(left.offset[0], right.offset[0]), min(left.offset[1], right.offset[1])
                                if overlap[0] > overlap[1]:
                                    continue
                                distance2 = self._box_distance_squared(left.swept_state.slice(0, 3), right.swept_state.slice(0, 3))
                                required = max(radii[i] + radii[j] + clearance, pair_distance)
                                if distance2 <= required * required:
                                    return CollisionCertificate(GuardStatus.UNRESOLVED,
                                        f"No swept pair separation proved for chasers {i}/{j}, offset {overlap}", tuple(evidence), **common)
                                evidence.append({"kind": "pair", "chasers": (i, j),
                                    "offset": tuple(map(str, overlap)), "distance_squared_lower": str(distance2),
                                    "required_distance": str(required)})
            except _ProofUnresolved as error:
                return CollisionCertificate(GuardStatus.UNRESOLVED, str(error), tuple(evidence), **common)
        return CollisionCertificate(GuardStatus.PROVEN_SAFE,
            "Strict separation proved throughout every overlapping continuous swept interval", tuple(evidence), **common)

    @staticmethod
    def _check_flow_structure(flow):
        """Reject incomplete purported proofs before examining their geometry."""
        for box in (flow.initial_state, flow.endpoint, flow.swept_state):
            if not isinstance(box, IntervalBox) or box.dimension != 6:
                raise ValueError("A proved flow requires six-state initial, endpoint and swept boxes")
        start, hold = _pair(flow.start_anomaly, "flow start"), _pair(flow.hold_width, "flow hold")
        if hold[0] < 0:
            raise ValueError("A proved flow cannot have a negative hold")
        if not flow.segments:
            if hold[1] != 0 or flow.endpoint != flow.initial_state or flow.swept_state != flow.initial_state:
                raise ValueError("Only an exact zero-width flow can omit continuous segments")
            return
        offset, previous = Fraction(0), flow.initial_state
        for segment in flow.segments:
            if not isinstance(segment, SweptSegment):
                raise TypeError("Continuous coverage requires SweptSegment records")
            lo, hi = _pair(segment.offset, "segment offset")
            if lo != offset or hi <= lo:
                raise ValueError("Continuous segments must partition the complete hold without gaps")
            qlo, qhi = _pair(segment.anomaly, "segment anomaly")
            if qlo > start[0] + lo or qhi < start[1] + hi:
                raise ValueError("Segment anomaly does not cover its declared clock and offset")
            for box in (segment.initial_state, segment.swept_state, segment.endpoint,
                        segment.picard_image, segment.derivative):
                if not isinstance(box, IntervalBox) or box.dimension != 6:
                    raise ValueError("Continuous segment boxes must all have six states")
            if (not segment.initial_state.contains(previous)
                    or not segment.swept_state.contains(segment.initial_state)
                    or not segment.swept_state.contains(segment.endpoint)
                    or not flow.swept_state.contains(segment.swept_state)):
                raise ValueError("Continuous segment boxes do not enclose their complete state chain")
            offset, previous = hi, segment.endpoint
        if offset != hold[1] or not flow.swept_state.contains(flow.endpoint):
            raise ValueError("Continuous segments do not cover the complete hold and final successor")

    @staticmethod
    def _segments(flow):
        if flow.segments:
            return flow.segments
        # A proven zero-width flow has no integration segments.
        box = flow.initial_state
        return (SweptSegment((Fraction(0), Fraction(0)), flow.start_anomaly,
                             box, box, box, box, IntervalBox.point((0,) * 6), 0),)

    @staticmethod
    def _box_distance_squared(left, right):
        return sum((max(Fraction(0), lo1 - hi2, lo2 - hi1) ** 2 for lo1, hi1, lo2, hi2 in
                    zip(left.lower, left.upper, right.lower, right.upper)), Fraction(0))

    def _target_segment(self, segment, body_radius, target, normals, ar):
        position = segment.swept_state.slice(0, 3)
        matrix = None
        if target.orientation_matrix_at_start is not None:
            matrix = tuple(tuple(ar.scalar(value) for value in row) for row in target.orientation_matrix_at_start)
            axis = tuple(row[2] for row in matrix)
        else:
            roll, pitch, _ = ar.vector(target.orientation_at_start)
            # The cylinder body axis is z; yaw cannot change its occupied set.
            axis = (pitch.sin(), -roll.sin() * pitch.cos(), roll.cos() * pitch.cos())
        axis_mid = tuple(ar.exact_point(a.mid()) for a in axis)
        relative_mid = tuple((lo + hi - clo - chi) / 2 for lo, hi, clo, chi in
                             zip(position.lower, position.upper, target.center_at_start.lower, target.center_at_start.upper))
        axis2 = sum(a * a for a in axis_mid)
        projection = (sum(p * a for p, a in zip(relative_mid, axis_mid)) / axis2) if axis2 else Fraction(0)
        radial = tuple(p - projection * a for p, a in zip(relative_mid, axis_mid))
        defaults = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                    relative_mid, radial, axis_mid, tuple(-a for a in axis_mid))
        center = ar.vector(target.center_at_start)
        p = ar.vector(position)
        r, half = ar.scalar(target.radius), ar.scalar(target.half_length)
        circumsphere = (r * r + half * half).sqrt()
        elapsed = ar.scalar(segment.offset[1])
        # ||R(q)-R(q0)|| <= min(2, integral ||omega_rel|| dq).
        rotation = min(Fraction(2), target.angular_rate_per_anomaly * segment.offset[1])
        motion_reserve = (circumsphere * ar.scalar(rotation + target.orientation_matrix_error)
                          + ar.scalar(target.center_speed_per_anomaly) * elapsed)
        for normal in normals + defaults:
            if not any(normal):
                continue
            n = tuple(ar.scalar(value) for value in normal)
            norm2 = sum((a * a for a in n), ar.arb(0))
            norm = norm2.sqrt()
            if matrix is None:
                axial = sum((a * b for a, b in zip(n, axis)), ar.arb(0))
                radial2 = (norm2 - axial * axial).nonnegative_part()
                support = r * radial2.sqrt() + half * abs(axial)
            else:
                body_normal = tuple(sum((n[i] * matrix[i][j] for i in range(3)), ar.arb(0)) for j in range(3))
                support = r * (body_normal[0] ** 2 + body_normal[1] ** 2).sqrt() + half * abs(body_normal[2])
            separation = sum((a * (b - c) for a, b, c in zip(n, p, center)), ar.arb(0))
            margin = separation - support - norm * (motion_reserve + ar.scalar(body_radius))
            if margin.is_finite() and margin > 0:
                return {"offset": tuple(map(str, segment.offset)),
                        "normal": tuple(map(str, normal)), "margin": ar.box((margin,)).as_dict(),
                        "rotation_norm_bound": str(rotation),
                        "rotation_and_translation_reserve": ar.box((motion_reserve,)).as_dict()}
        return None
