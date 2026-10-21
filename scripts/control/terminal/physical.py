"""Derived finite-domain terminal bounds for an eccentric rigid-body target.

Arb interval Taylor coefficients bound the analytical orbit continuously. Exact
rational first-exit comparisons bound the target spin, and rotation-invariant
geometry bounds the assigned side ports. No sampled maximum is a proof input.
The numerical target preview and sensor/actuator errors remain explicit,
declared assumptions. See ``docs/physical-terminal.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from numbers import Integral
from typing import Any, Sequence

from orbexa.control.certification.reachability import IntervalBox, OrbitalModelBounds, _ARB_LOCK, _ArbArithmetic, _ProofUnresolved, _exact
from orbexa.control.terminal.tracking import BoundScope, TerminalCertificate, TerminalControlCommand, TerminalFeedforwardPolicy, TerminalGeometryBounds, TerminalHoldReferences, TerminalReferenceSample, TerminalTrackingBounds, certify_terminal_tracking


PHYSICAL_TERMINAL_METHOD = "arb-orbit-rigid-body-terminal-v1"


def _vector(values, size, name, *, nonnegative=False, positive=False):
    result = tuple(map(_exact, values))
    if len(result) != size or (nonnegative and any(v < 0 for v in result)) or (
            positive and any(v <= 0 for v in result)):
        raise ValueError(f"{name} needs {size} components with valid signs")
    return result


def _matrix(values, rows, columns, name, *, nonnegative=False):
    result = tuple(_vector(row, columns, name, nonnegative=nonnegative) for row in values)
    if len(result) != rows:
        raise ValueError(f"{name} must have shape ({rows},{columns})")
    return result


def _encode(value):
    if isinstance(value, Fraction):
        return str(value)
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_encode(item) for item in value)
    return value


def _abs(ar, ball):
    enclosure = ar.box((ball,))
    return max(abs(enclosure.lower[0]), abs(enclosure.upper[0]))


def _upper(ar, ball):
    return ar.box((ball,)).upper[0]


def _sqrt_upper(ar, value):
    if value < 0:
        raise ValueError("A squared magnitude cannot be negative")
    return _upper(ar, ar.scalar(value).sqrt())


def _dot(matrix, vector):
    return tuple(sum((a * v for a, v in zip(row, vector)), Fraction(0)) for row in matrix)


def _add(*vectors):
    return tuple(map(sum, zip(*vectors)))


def _orbit_series(flint, ar, domain, parameters):
    """Formal q-jets with interval constant term: coefficient k encloses f^(k)/k!."""
    q = flint.arb_series([ar.interval(domain), ar.arb(1)], prec=4)
    e, alpha, beta, phase = (ar.interval(getattr(parameters, name))
                             for name in ("eccentricity", "alpha", "beta", "theta0"))
    mu, axis = ar.scalar(parameters.mu), ar.scalar(parameters.semi_major_axis)
    h2 = mu * axis * (1 - e * e)
    exponential = (2 * alpha * q).exp()
    denominator = exponential + e * (q - phase).cos()
    if not denominator[0] > 0 or not h2 > 0:
        raise ValueError("The entire orbital denominator and h(e)^2 must be proved positive")
    scale = 1 + 4 * alpha * alpha
    radius = h2 * scale / (mu * denominator)
    k = (e * (q - phase).sin() - 2 * alpha * exponential) / denominator
    curvature = scale * exponential / denominator
    gamma, damping = beta - alpha, 2 * k - (beta - alpha)
    zero = flint.arb_series([ar.arb(0)], prec=4)
    one = flint.arb_series([ar.arb(1)], prec=4)
    matrix = ((1 - curvature, -damping, zero, damping, 2 * one, zero),
              (damping, 1 + 2 * curvature, zero, -2 * one, damping, zero),
              (zero, zero, -curvature, zero, zero, damping))
    affine = (gamma * radius, -gamma * radius * k, zero)
    rate = h2.sqrt() * (-alpha * q).exp() / (radius * radius)
    if not radius[0] > 0 or not rate[0] > 0:
        raise ValueError("Radius and dq/dt must be proved strictly positive on the entire domain")
    return dict(radius=radius, radius_log_derivative=k, rate=rate,
                matrix=matrix, affine=affine)


@dataclass(frozen=True)
class OrbitalClockBounds:
    """Arb-proved global orbital radius/rate boxes on one finite q interval."""

    anomaly: tuple[Fraction, Fraction]
    parameters: OrbitalModelBounds
    radius: IntervalBox
    anomaly_rate: IntervalBox
    anomaly_rate_prime: IntervalBox
    anomaly_rate_double_prime: IntervalBox
    physical_duration_upper: Fraction
    precision_bits: int
    backend_version: str
    method: str = PHYSICAL_TERMINAL_METHOD

    def as_dict(self):
        return _encode(vars(self))


def bound_orbital_clock(start_anomaly, end_anomaly, parameters: OrbitalModelBounds,
                        *, precision_bits: int = 160) -> OrbitalClockBounds:
    """Enclose dq/dt globally without samples; useful for force/fuel budgets.

    The duration bound is (q_end-q_start)/min(dq/dt). Floats are interpreted
    as exact binary rationals. Invalid or unresolved domains raise; absence of
    python-flint raises ImportError. Neither condition yields a certificate.
    """
    start, end = _exact(start_anomaly), _exact(end_anomaly)
    if end < start or isinstance(precision_bits, bool) or not isinstance(precision_bits, Integral) or precision_bits < 64:
        raise ValueError("Use an ordered finite interval and at least 64 precision bits")
    import flint
    with _ARB_LOCK, flint.ctx.workprec(int(precision_bits)):
        ar = _ArbArithmetic(flint)
        old_cap = flint.ctx.cap
        try:
            flint.ctx.cap = 4
            orbit = _orbit_series(flint, ar, (start, end), parameters)
            radius = ar.box((orbit["radius"][0],))
            rate = ar.box((orbit["rate"][0],))
            prime = ar.box((orbit["rate"][1],))
            double = ar.box((2 * orbit["rate"][2],))
            return OrbitalClockBounds((start, end), parameters, radius, rate, prime, double,
                                      (end - start) / rate.lower[0], int(precision_bits),
                                      getattr(flint, "__version__", "unknown"))
        finally:
            flint.ctx.cap = old_cap


def _determinant(matrix):
    if len(matrix) == 1:
        return matrix[0][0]
    return sum(((-1) ** j * matrix[0][j] * _determinant(
        tuple(tuple(row[k] for k in range(len(matrix)) if k != j) for row in matrix[1:]))
        for j in range(len(matrix))), Fraction(0))


def _positive_definite(matrix):
    return all(_determinant(tuple(tuple(matrix[i][j] for j in range(size))
                                   for i in range(size))) > 0 for size in range(1, 4))


def _inertia_bounds(values):
    """Exact Sylvester/PSD and rational spectral brackets; no numerical eigenvalues."""
    inertia = _matrix(values, 3, 3, "Body inertia")
    if any(inertia[i][j] != inertia[j][i] for i in range(3) for j in range(3)) or not _positive_definite(inertia):
        raise ValueError("Body inertia must be exactly symmetric and positive definite")
    trace = sum(inertia[i][i] for i in range(3))
    physical = tuple(tuple((trace if i == j else 0) - 2 * inertia[i][j]
                           for j in range(3)) for i in range(3))
    for size in range(1, 4):
        for indices in combinations(range(3), size):
            if _determinant(tuple(tuple(physical[i][j] for j in indices) for i in indices)) < 0:
                raise ValueError("Body inertia violates the physical principal-moment triangle")
    if all(inertia[i][j] == 0 for i in range(3) for j in range(3) if i != j):
        return inertia, min(inertia[i][i] for i in range(3)), max(inertia[i][i] for i in range(3))
    lo_min, hi_min = Fraction(0), min(inertia[i][i] for i in range(3))
    lo_max = max(inertia[i][i] for i in range(3))
    hi_max = max(inertia[i][i] + sum(abs(inertia[i][j]) for j in range(3) if i != j)
                 for i in range(3))
    for _ in range(96):
        mid = (lo_min + hi_min) / 2
        shifted = tuple(tuple(inertia[i][j] - (mid if i == j else 0) for j in range(3)) for i in range(3))
        if _positive_definite(shifted):
            lo_min = mid
        else:
            hi_min = mid
        mid = (lo_max + hi_max) / 2
        shifted = tuple(tuple((mid if i == j else 0) - inertia[i][j] for j in range(3)) for i in range(3))
        if _positive_definite(shifted):
            hi_max = mid
        else:
            lo_max = mid
    if lo_min <= 0:
        raise ValueError("The finite exact spectral refinement did not obtain a positive lower inertia bound")
    return inertia, lo_min, hi_max


@dataclass(frozen=True)
class PhysicalTerminalErrorBounds:
    """Declared instrumentation/evaluation budgets, never inferred from samples.

    Reference evaluation errors are relative to the exact continuous physical
    reference. The anomaly error shifts evaluation queries, not execution
    timing; actual holds still obey the separately declared maximum_hold.
    """

    sensing_error_abs: Sequence[Any] = (0, 0, 0, 0, 0, 0)
    reference_state_error_abs: Sequence[Any] = (0, 0, 0, 0, 0, 0)
    reference_acceleration_error_abs: Sequence[Any] = (0, 0, 0)
    nominal_matrix_evaluation_error_abs: Sequence[Sequence[Any]] = ((0,) * 6,) * 3
    nominal_affine_evaluation_error_abs: Sequence[Any] = (0, 0, 0)
    anomaly_evaluation_error_abs: Any = 0
    actuator_error_abs: Sequence[Any] = (0, 0, 0)
    acceleration_disturbance_abs: Sequence[Any] = (0, 0, 0)
    target_solid_position_error_radius: Any = 0
    declaration: str = ""

    def __post_init__(self):
        for name, size in (("sensing_error_abs", 6), ("reference_state_error_abs", 6),
                           ("reference_acceleration_error_abs", 3), ("nominal_affine_evaluation_error_abs", 3),
                           ("actuator_error_abs", 3), ("acceleration_disturbance_abs", 3)):
            object.__setattr__(self, name, _vector(getattr(self, name), size, name, nonnegative=True))
        object.__setattr__(self, "nominal_matrix_evaluation_error_abs", _matrix(
            self.nominal_matrix_evaluation_error_abs, 3, 6, "nominal matrix error", nonnegative=True))
        for name in ("anomaly_evaluation_error_abs", "target_solid_position_error_radius"):
            value = _exact(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if not isinstance(self.declaration, str) or not self.declaration.strip():
            raise ValueError("Identify the source/assumption of the declared sensor and evaluation error budgets")

    def as_dict(self):
        return _encode(vars(self))


@dataclass(frozen=True)
class PhysicalTerminalResult:
    status: str
    reason: str
    certificate: TerminalCertificate | None
    derived: dict = field(default_factory=dict)
    assumptions: dict = field(default_factory=dict)
    attempts: tuple[dict, ...] = ()
    method: str = PHYSICAL_TERMINAL_METHOD

    @property
    def admissible(self):
        return self.status == "CONDITIONAL_PROVEN" and self.certificate is not None and self.certificate.admissible

    def as_dict(self):
        return _encode(vars(self))

    def reference_values_reason(self, reference: TerminalReferenceSample) -> str:
        """Reject values that contradict the derived global reference contract.

        This necessary consistency check does not establish pointwise numerical
        evaluation accuracy; those declared error bounds retain their own source.
        Admission and command execution use the same checks.
        """
        if not self.admissible:
            return "The physical terminal global bounds were not proved"
        b, derived = self.certificate.bounds, self.derived
        if reference.reference_id != b.reference_id or not b.scope.contains(reference.anomaly):
            return "Reference identity or anomaly lies outside the physical terminal contract"
        if any(abs(value) > bound + error for value, bound, error in zip(
                reference.state, b.reference_state_abs, derived["effective_reference_state_error_abs"])):
            return "Evaluated reference state contradicts its global derived bound"
        if any(abs(value) > derived["reference_acceleration_norm_upper"] + error for value, error in zip(
                reference.acceleration, derived["effective_reference_acceleration_error_abs"])):
            return "Evaluated reference acceleration contradicts its global derived bound"
        if any(abs(value) > bound + error for row, row_bound, row_error in zip(
                reference.nominal_matrix, b.nominal_matrix_abs, derived["effective_nominal_matrix_error_abs"])
                for value, bound, error in zip(row, row_bound, row_error)):
            return "Evaluated nominal matrix contradicts its global derived bound"
        if any(abs(value) > bound + error for value, bound, error in zip(
                reference.nominal_affine, derived["nominal_affine_abs"], derived["effective_nominal_affine_error_abs"])):
            return "Evaluated nominal affine term contradicts its global derived bound"
        return ""

    def robust_contains(self, state: IntervalBox, reference: TerminalReferenceSample) -> bool:
        """Membership for every exact reference inside its declared evaluation box."""
        if not self.admissible or state.dimension != 6:
            return False
        b = self.certificate.bounds
        if self.reference_values_reason(reference):
            return False
        error = self.derived["effective_reference_state_error_abs"]
        for i in range(3):
            ep_lo = state.lower[i] - reference.state[i] - error[i]
            ep_hi = state.upper[i] - reference.state[i] + error[i]
            ev_lo = state.lower[i + 3] - reference.state[i + 3] - error[i + 3]
            ev_hi = state.upper[i + 3] - reference.state[i + 3] + error[i + 3]
            if (ep_lo < -b.omega[i] or ep_hi > b.omega[i]
                    or ev_lo + b.lambda_gain[i] * ep_lo < -b.phi[i]
                    or ev_hi + b.lambda_gain[i] * ep_hi > b.phi[i]):
                return False
        return True


@dataclass(frozen=True)
class PhysicalTerminalBoundsBuilder:
    """Derive global bounds and certify one side port over a finite anomaly window.

    ``entry_spin`` encloses the true body angular velocity in rad/s at
    start_anomaly. Inertia is fixed, exactly symmetric and physically realizable.
    Every body port shares the target's true rigid rotation. Peers obey the
    declared position-radius upper bound (own radius by default) and common
    physical enclosing body radius.
    """

    start_anomaly: Any
    end_anomaly: Any
    target_orbit: OrbitalModelBounds
    plant_parameters: OrbitalModelBounds
    nominal_parameters: OrbitalModelBounds
    inertia: Sequence[Sequence[Any]]
    entry_spin: IntervalBox
    body_ports: Sequence[Sequence[Any]]
    errors: PhysicalTerminalErrorBounds
    reference_id: str
    port_index: int = 0
    target_radius: Any = "0.8"
    target_half_length: Any = "0.3"
    target_tolerance: Any = "0.0001"
    body_radius: Any = "0.15"
    clearance: Any = "0.001"
    omega: Sequence[Any] = ("0.00115",) * 3
    phi: Sequence[Any] = ("0.020",) * 3
    lambda_gain: Sequence[Any] = (20,) * 3
    bandwidth_candidates: Sequence[Any] = (80, 120, 160, 240)
    maximum_hold: Any = "0.00049087384375"
    input_limits: IntervalBox = field(default_factory=lambda: IntervalBox((-100,) * 3, (100,) * 3))
    position_tolerance: Any = "0.002"
    anomaly_velocity_tolerance: Any = "0.08"
    peer_position_radius_squared_upper: Any | None = None
    gravity_gradient: bool = True
    precision_bits: int = 160

    def __post_init__(self):
        for name in ("start_anomaly", "end_anomaly", "target_radius", "target_half_length", "target_tolerance",
                     "body_radius", "clearance", "maximum_hold", "position_tolerance", "anomaly_velocity_tolerance"):
            object.__setattr__(self, name, _exact(getattr(self, name)))
        if self.end_anomaly <= self.start_anomaly:
            raise ValueError("A physical terminal window must have positive finite width")
        for name in ("target_radius", "target_half_length", "body_radius", "maximum_hold",
                     "position_tolerance", "anomaly_velocity_tolerance"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.target_tolerance < 0 or self.clearance < 0:
            raise ValueError("Target tolerance and clearance must be nonnegative")
        if self.peer_position_radius_squared_upper is not None:
            value = _exact(self.peer_position_radius_squared_upper)
            if value < 0:
                raise ValueError("Peer position-radius squared must be nonnegative")
            object.__setattr__(self, "peer_position_radius_squared_upper", value)
        if self.entry_spin.dimension != 3 or self.input_limits.dimension != 3:
            raise ValueError("Spin and input boxes have dimension three")
        ports = tuple(_vector(point, 3, "Body port") for point in self.body_ports)
        if not ports or isinstance(self.port_index, bool) or not isinstance(self.port_index, Integral) or not 0 <= self.port_index < len(ports):
            raise ValueError("Supply a nonempty port list and a valid port index")
        object.__setattr__(self, "body_ports", ports)
        for name in ("omega", "phi", "lambda_gain"):
            object.__setattr__(self, name, _vector(getattr(self, name), 3, name, positive=True))
        candidates = tuple(map(_exact, self.bandwidth_candidates))
        if not candidates or any(k <= 0 for k in candidates):
            raise ValueError("Supply at least one positive candidate bandwidth")
        object.__setattr__(self, "bandwidth_candidates", candidates)
        if not isinstance(self.gravity_gradient, bool):
            raise ValueError("gravity_gradient must be Boolean")
        if isinstance(self.precision_bits, bool) or not isinstance(self.precision_bits, Integral) or self.precision_bits < 64:
            raise ValueError("Use at least 64 precision bits")
        if not isinstance(self.reference_id, str) or not self.reference_id.strip():
            raise ValueError("An authoritative reference identity is required")
        for name in ("eccentricity", "alpha", "beta", "theta0"):
            lo, hi = getattr(self.nominal_parameters, name)
            if lo != hi:
                raise ValueError("The nominal controller model must be a fixed point context")
        # The force model and reference clock describe one central body and
        # one prescribed fixed semi-major-axis context. e/drag may differ.
        for model in (self.target_orbit, self.nominal_parameters):
            if (model.mu != self.plant_parameters.mu or model.semi_major_axis != self.plant_parameters.semi_major_axis):
                raise ValueError("Target, plant and nominal model need the same mu and semi-major axis")
        inertia, _, _ = _inertia_bounds(self.inertia)
        object.__setattr__(self, "inertia", inertia)

    def build(self) -> PhysicalTerminalResult:
        assumptions = _encode(vars(self))
        try:
            import flint
        except ImportError as error:
            return PhysicalTerminalResult("UNAVAILABLE", f"python-flint is required: {error}", None,
                                          assumptions=assumptions)
        with _ARB_LOCK, flint.ctx.workprec(int(self.precision_bits)):
            old_cap = flint.ctx.cap
            try:
                flint.ctx.cap = 4
                return self._derive(flint, _ArbArithmetic(flint), assumptions)
            except (ValueError, ZeroDivisionError, _ProofUnresolved) as error:
                return PhysicalTerminalResult("REFUSED", str(error), None, assumptions=assumptions)
            finally:
                flint.ctx.cap = old_cap

    def _geometry_distance(self, ar, point):
        """Default cylinder geometry; explicit composite builders override it.

        Keeping geometry behind this hook leaves the previously qualified
        cylinder assumptions, bound arithmetic and serialized results intact.
        """
        radial_squared = point[0] ** 2 + point[1] ** 2
        radial = ar.box((ar.scalar(radial_squared).sqrt(),)).lower[0]
        distance = radial - self.target_radius * (1 + self.target_tolerance) - self.errors.target_solid_position_error_radius
        if distance < 0:
            raise ValueError("The selected port does not have a positive radial distance to the inflated target solid")
        return distance, {}

    def _geometry_proof_reference(self):
        return PHYSICAL_TERMINAL_METHOD

    def _terminal_proof_reference(self):
        return self._geometry_proof_reference()

    def _terminal_policy(self):
        return TerminalFeedforwardPolicy.START_V1

    def _reference_jet_bounds(self, flint, ar, *, point, point_radius, rotation_derivatives,
                              duration, rate, rate_prime, rate_double, torque_factor, radius_log_prime):
        return (point_radius,) + tuple(point_radius * value for value in rotation_derivatives), {}

    def _feedforward_range(self, ar, nominal, p2, reference_state, matrix_abs):
        state_balls = tuple(ar.interval((-value, value)) for value in reference_state)
        return ar.box(tuple(ar.interval((-p2, p2)) - sum(
            (value[0] * x for value, x in zip(row, state_balls)), ar.arb(0)) - drift[0]
            for row, drift in zip(nominal["matrix"], nominal["affine"])))

    def _evaluation_input_error(self, *, acceleration_error, affine_error, matrix_error,
                                reference_state, error_state, reference_error, matrix_abs):
        return _add(acceleration_error, affine_error,
            _dot(matrix_error, _add(reference_state, error_state, self.errors.sensing_error_abs)),
            _dot(matrix_abs, reference_error))

    def _spin_bound(self, ar, entry_norm, duration, quadratic, constant, jmin):
        """Original scalar first-exit comparison, retained for cylinder records."""
        c = entry_norm + constant * duration
        discriminant = 1 - 4 * quadratic * duration * c
        if discriminant <= 0:
            raise ValueError("Spin first-exit comparison has no finite certified branch; shorten the window")
        if quadratic:
            root = 2 * ar.scalar(c) / (1 + ar.scalar(discriminant).sqrt())
            spin = _upper(ar, root) * Fraction(1000000000001, 1000000000000) + Fraction(1, 10 ** 30)
        else:
            spin = c + Fraction(1, 10 ** 30)
        margin = spin - entry_norm - (quadratic * spin * spin + constant) * duration
        if margin <= 0:
            raise ValueError("The exact spin self-inclusion inequality was not strict")
        return spin, discriminant, margin, {}

    def _derive(self, flint, ar, assumptions):
        q_error = self.errors.anomaly_evaluation_error_abs
        domain = (self.start_anomaly - q_error, self.end_anomaly + q_error)
        target = _orbit_series(flint, ar, domain, self.target_orbit)
        plant = _orbit_series(flint, ar, domain, self.plant_parameters)
        nominal = _orbit_series(flint, ar, domain, self.nominal_parameters)
        radius = ar.box((target["radius"][0],))
        rate = ar.box((target["rate"][0],))
        rate_prime = _abs(ar, target["rate"][1])
        rate_double = _abs(ar, 2 * target["rate"][2])
        duration = (self.end_anomaly - self.start_anomaly + q_error) / rate.lower[0]
        inertia, jmin, jmax = _inertia_bounds(self.inertia)
        spread = jmax - jmin
        entry_norm = _sqrt_upper(ar, sum(max(abs(lo), abs(hi)) ** 2
                                        for lo, hi in zip(self.entry_spin.lower, self.entry_spin.upper)))
        torque_factor = 3 * self.target_orbit.mu / radius.lower[0] ** 3 if self.gravity_gradient else Fraction(0)
        quadratic = spread / jmin
        constant = torque_factor * spread / (2 * jmin)
        spin, discriminant, spin_margin, spin_evidence = self._spin_bound(
            ar, entry_norm, duration, quadratic, constant, jmin)
        # Center J by (Jmax+Jmin)/2 I to remove its isotropic component.
        spin_dot = spread * (spin * spin + torque_factor) / (2 * jmin)
        radius_log_prime = _abs(ar, target["radius_log_derivative"][0])
        torque_factor_dot = 3 * torque_factor * radius_log_prime * rate.upper[0]
        torque_dot = torque_factor_dot * spread / 2 + torque_factor * spread * (spin + rate.upper[0])
        spin_double = (torque_dot + spread * spin * spin_dot) / jmin
        k0 = spin / rate.lower[0]
        k1 = (spin_dot + spin * rate_prime) / rate.lower[0] ** 2
        k2 = ((spin_double + 3 * spin_dot * rate_prime + 2 * spin * rate_prime ** 2) / rate.lower[0] ** 3
              + spin * rate_double / rate.lower[0] ** 2)
        rotation_derivatives = (1 + k0, (1 + k0) ** 2 + k1,
                                (1 + k0) ** 3 + 3 * (1 + k0) * k1 + k2)
        point = self.body_ports[self.port_index]
        point_radius = _sqrt_upper(ar, sum(v * v for v in point))
        (p0, p1, p2, p3), reference_evidence = self._reference_jet_bounds(flint, ar,
            point=point, point_radius=point_radius, rotation_derivatives=rotation_derivatives,
            duration=duration, rate=rate, rate_prime=rate_prime, rate_double=rate_double,
            torque_factor=torque_factor, radius_log_prime=radius_log_prime)
        reference_state = (p0,) * 3 + (p1,) * 3
        reference_derivative = (p1,) * 3 + (p2,) * 3
        matrix_abs = tuple(tuple(_abs(ar, value[0]) for value in row) for row in nominal["matrix"])
        matrix_prime = tuple(tuple(_abs(ar, value[1]) for value in row) for row in nominal["matrix"])
        delta = tuple(tuple(_abs(ar, p[0] - n[0]) for p, n in zip(row_p, row_n))
                      for row_p, row_n in zip(plant["matrix"], nominal["matrix"]))
        affine_delta = tuple(_abs(ar, p[0] - n[0]) for p, n in zip(plant["affine"], nominal["affine"]))
        affine_prime = tuple(_abs(ar, value[1]) for value in nominal["affine"])
        lipschitz = _add((p3,) * 3, _dot(matrix_prime, reference_state),
                         _dot(matrix_abs, reference_derivative), affine_prime)
        feedforward = self._feedforward_range(ar, nominal, p2, reference_state, matrix_abs)
        reference_error = _add(self.errors.reference_state_error_abs,
                               tuple(q_error * value for value in reference_derivative))
        acceleration_error = _add(self.errors.reference_acceleration_error_abs, (q_error * p3,) * 3)
        matrix_error = tuple(tuple(error + q_error * derivative for error, derivative in zip(row_e, row_d))
                             for row_e, row_d in zip(self.errors.nominal_matrix_evaluation_error_abs, matrix_prime))
        affine_error = _add(self.errors.nominal_affine_evaluation_error_abs,
                            tuple(q_error * value for value in affine_prime))
        error_velocity = tuple(p + l * o for p, l, o in zip(self.phi, self.lambda_gain, self.omega))
        error_state = self.omega + error_velocity
        # Exact-reference measured error uses n-E_g. The remaining difference
        # in the evaluated command is bounded independently as actuator forcing.
        sensing = _add(self.errors.sensing_error_abs, reference_error)
        evaluation_input_error = self._evaluation_input_error(acceleration_error=acceleration_error,
            affine_error=affine_error, matrix_error=matrix_error, reference_state=reference_state,
            error_state=error_state, reference_error=reference_error, matrix_abs=matrix_abs)
        actuator = _add(self.errors.actuator_error_abs, evaluation_input_error)
        target_distance, geometry_evidence = self._geometry_distance(ar, point)
        peer_radius_squared = (sum(v * v for v in self.omega) if self.peer_position_radius_squared_upper is None
                               else self.peer_position_radius_squared_upper)
        peer_geometry = []
        for index, peer in enumerate(self.body_ports):
            if index == self.port_index:
                continue
            distance = ar.box((ar.scalar(sum((a - b) ** 2 for a, b in zip(point, peer))).sqrt(),)).lower[0]
            peer_geometry.append((distance, peer_radius_squared, self.body_radius))
        scope = BoundScope(self.start_anomaly, self.end_anomaly,
            proof_kind="validated-interval-global", proof_reference=self._terminal_proof_reference(),
            assumptions=("Arb interval jets enclose the analytical orbit over the complete finite anomaly domain",
                         "The entry spin box encloses the true target spin; fixed physical inertia and central gravity-gradient torque only",
                         "All assigned body ports share the actual target rigid rotation and peers obey the declared terminal bounds",
                         "Declared sensor, reference-evaluation, process, geometry and actuator error bounds hold throughout the window",
                         "The actual scheduled held controller and authoritative reference satisfy the documented evaluation contract"))
        derived = dict(expanded_query_domain=domain, radius=radius, anomaly_rate=rate,
            anomaly_rate_prime_abs=rate_prime, anomaly_rate_double_prime_abs=rate_double,
            physical_duration_upper=duration, inertia=inertia, inertia_eigenvalue_lower=jmin,
            inertia_eigenvalue_upper=jmax, inertia_spectral_spread_upper=spread,
            entry_spin_norm_upper=entry_norm, spin_norm_upper=spin, spin_quadratic_coefficient=quadratic,
            spin_constant_coefficient=constant, spin_discriminant=discriminant,
            spin_first_exit_margin=spin_margin, spin_dot_norm_upper=spin_dot,
            spin_double_dot_norm_upper=spin_double, torque_factor_upper=torque_factor,
            torque_dot_norm_upper=torque_dot, body_anomaly_spin_derivatives=(k0, k1, k2),
            rotation_derivative_operator_bounds=rotation_derivatives,
            reference_position_norm_upper=p0, reference_velocity_norm_upper=p1,
            reference_acceleration_norm_upper=p2, reference_jerk_norm_upper=p3,
            effective_reference_state_error_abs=reference_error,
            effective_reference_acceleration_error_abs=acceleration_error,
            effective_nominal_matrix_error_abs=matrix_error, effective_nominal_affine_error_abs=affine_error,
            nominal_affine_abs=tuple(_abs(ar, value[0]) for value in nominal["affine"]),
            evaluation_input_error_abs=evaluation_input_error, physical_sensing_error_abs=self.errors.sensing_error_abs,
            physical_actuator_error_abs=self.errors.actuator_error_abs,
            nominal_matrix_prime_abs=matrix_prime, nominal_affine_prime_abs=affine_prime,
            target_reference_distance_lower=target_distance,
            peer_position_radius_squared_upper=peer_radius_squared,
            precision_bits=int(self.precision_bits), backend_version=getattr(flint, "__version__", "unknown"),
            sampled_maxima_used=False)
        derived.update(geometry_evidence)
        derived.update(spin_evidence)
        derived.update(reference_evidence)
        attempts = []
        last = None
        for bandwidth in self.bandwidth_candidates:
            bounds = TerminalTrackingBounds(scope=scope, reference_id=self.reference_id,
                omega=self.omega, phi=self.phi, lambda_gain=self.lambda_gain, bandwidth=(bandwidth,) * 3,
                maximum_hold=self.maximum_hold, nominal_matrix_abs=matrix_abs,
                matrix_uncertainty_abs=delta,
                nominal_matrix_variation=tuple(tuple(self.maximum_hold * value for value in row) for row in matrix_prime),
                reference_state_abs=reference_state, affine_uncertainty_abs=affine_delta,
                acceleration_disturbance_abs=self.errors.acceleration_disturbance_abs,
                feedforward_range=feedforward, feedforward_lipschitz=lipschitz,
                sensing_error_abs=sensing, actuator_error_abs=actuator, input_limits=self.input_limits,
                geometry=TerminalGeometryBounds(target_distance, self.body_radius, self.clearance, tuple(peer_geometry)),
                position_tolerance=self.position_tolerance, anomaly_velocity_tolerance=self.anomaly_velocity_tolerance)
            last = certify_terminal_tracking(bounds, policy=self._terminal_policy())
            attempts.append(dict(bandwidth=bandwidth, admissible=last.admissible, reasons=last.reasons,
                                 sliding_inward_margin=last.quantities["sliding_inward_margin"],
                                 lower_input_margin=last.quantities["lower_input_margin"],
                                 upper_input_margin=last.quantities["upper_input_margin"]))
            if last.admissible:
                return PhysicalTerminalResult("CONDITIONAL_PROVEN", "All global-bound and terminal inequalities proved", last,
                                              derived, assumptions, tuple(attempts))
        return PhysicalTerminalResult("REFUSED", "No declared candidate bandwidth satisfies every terminal inequality", last,
                                      derived, assumptions, tuple(attempts))


class PhysicalTerminalController:
    """Apply the held law with declared numerical reference evaluation errors.

    This wrapper uses robust entry membership and the physical sensor budget.
    The generic exact-reference controller deliberately keeps its stricter
    contract. A numerical target preview is accepted here only under the
    explicit evaluation-error assumptions used by the builder.
    """

    def __init__(self, builder: PhysicalTerminalBoundsBuilder):
        if not isinstance(builder, PhysicalTerminalBoundsBuilder):
            raise TypeError("Supply the original builder so global bounds are rederived, not a stored status flag")
        result = builder.build()
        if not result.admissible:
            raise ValueError("A proved physical terminal result is required")
        policy = TerminalFeedforwardPolicy(builder._terminal_policy())
        expected = builder._terminal_proof_reference()
        supported = (PHYSICAL_TERMINAL_METHOD, "arb-composite-orbit-rigid-body-terminal-v1",
                     "arb-componentwise-composite-midpoint-terminal-v1")
        if (expected not in supported or result.method != expected
                or result.certificate.policy is not policy
                or result.certificate.method != policy.certificate_method
                or result.certificate.bounds.scope.proof_reference != expected
                or (policy is TerminalFeedforwardPolicy.MIDPOINT_V1) !=
                   (expected == "arb-componentwise-composite-midpoint-terminal-v1")):
            raise ValueError("Unsupported or inconsistent physical terminal method, policy or scope proof identity")
        certificate = certify_terminal_tracking(result.certificate.bounds, policy=policy)
        if not certificate.admissible:
            raise ValueError("Terminal algebra recertification refused")
        if certificate.as_dict() != result.certificate.as_dict():
            raise ValueError("Physical terminal algebra differs from its policy-aware recertification")
        self.result = result
        self.certificate = certificate
        self.anomaly_evaluation_error_abs = builder.errors.anomaly_evaluation_error_abs

    def command(self, measured_state, state_enclosure: IntervalBox,
                approximate_reference: TerminalReferenceSample, *, hold_width,
                hold_references: TerminalHoldReferences | None = None):
        b, reference = self.certificate.bounds, approximate_reference
        hold = _exact(hold_width)
        measured = _vector(measured_state, 6, "Measured state")
        policy = self.certificate.policy
        if hold_references is None:
            if policy is not TerminalFeedforwardPolicy.START_V1:
                raise ValueError("Midpoint feedforward requires both exact held-reference events")
        elif (not isinstance(hold_references, TerminalHoldReferences)
              or hold_references.policy is not policy or hold_references.feedback != reference):
            raise ValueError("The held-reference policy or actual feedback sample changed")
        else:
            hold_references.validate(b, hold, maximum_query_displacement=
                                     self.anomaly_evaluation_error_abs)
        feedforward = reference if hold_references is None else hold_references.feedforward
        if not 0 < hold <= b.maximum_hold or not b.scope.contains_hold(reference.anomaly, hold):
            raise ValueError("The complete held command must belong to the certified finite scope")
        reference_reason = self.result.reference_values_reason(reference)
        if reference_reason:
            raise ValueError(reference_reason)
        feedforward_reason = self.result.reference_values_reason(feedforward)
        if feedforward_reason:
            raise ValueError(feedforward_reason)
        if not self.result.robust_contains(state_enclosure, reference):
            raise ValueError("Robust membership for the complete reference-evaluation box failed")
        sensor = self.result.derived["physical_sensing_error_abs"]
        if any(max(abs(lo - value), abs(hi - value)) > error for lo, hi, value, error in
               zip(state_enclosure.lower, state_enclosure.upper, measured, sensor)):
            raise ValueError("The measurement and state enclosure exceed the physical sensor budget")
        error = tuple(x - g for x, g in zip(measured, reference.state))
        sliding = tuple(v + lam * p for p, v, lam in zip(error[:3], error[3:], b.lambda_gain))
        saturation = tuple(max(Fraction(-1), min(Fraction(1), s / phi)) for s, phi in zip(sliding, b.phi))
        robust = _add(_dot(b.matrix_uncertainty_abs, tuple(map(abs, error))),
                      tuple(k * phi for k, phi in zip(b.bandwidth, b.phi)))
        cancellation = _add(_dot(reference.nominal_matrix, error),
            tuple(lam * value for lam, value in zip(b.lambda_gain, error[3:])),
            tuple(gain * sat for gain, sat in zip(robust, saturation)))
        command = tuple(u - correction for u, correction in zip(feedforward.feedforward, cancellation))
        actuator = self.result.derived["physical_actuator_error_abs"]
        applied = IntervalBox.from_center_radius(command, actuator)
        if not b.input_limits.contains(applied):
            raise ValueError("The actual evaluated command exceeds the input limits")
        return TerminalControlCommand(reference.anomaly, hold, command, applied, actuator, b.reference_id,
            hold_references if policy is not TerminalFeedforwardPolicy.START_V1 else None)


@dataclass(frozen=True)
class PhysicalTerminalAdmission:
    admissible: bool
    reasons: tuple[str, ...]
    anomaly: Fraction | None
    members_contained: tuple[bool, ...]

    def as_dict(self):
        return _encode(vars(self))


class PhysicalTerminalFleet:
    """Joint physical-port admission without replacing or resetting any state.

    Call ``admit`` on the actual approach endpoint and its independently valid
    state enclosures. A refused endpoint remains refused. ``commands`` repeats
    the membership checks at every actual sample and never moves a state onto
    its reference. The execution layer supplies sensing, scheduling, physical
    propagation and any independently required continuous-flow guard.
    """

    def __init__(self, builders: Sequence[PhysicalTerminalBoundsBuilder]):
        builders = tuple(builders)
        if not builders:
            raise ValueError("A terminal fleet needs at least one physical port builder")
        first = builders[0]
        shared = ("start_anomaly", "end_anomaly", "target_orbit", "plant_parameters", "inertia",
                  "entry_spin", "body_ports", "target_radius", "target_half_length", "target_tolerance",
                  "body_radius", "clearance", "gravity_gradient")
        if any(getattr(item, name) != getattr(first, name) for item in builders for name in shared):
            raise ValueError("Joint terminal builders need a common physical target, scope and peer geometry context")
        geometries = tuple(getattr(item, "target_geometry", None) for item in builders)
        if any(value is not None for value in geometries):
            if any(value is None for value in geometries) or any(
                    value.fingerprint != geometries[0].fingerprint for value in geometries):
                raise ValueError("Every terminal member must share the same explicit composite geometry; mixed geometry refuses")
        for index, item in enumerate(builders):
            peer_limit = (sum(v * v for v in item.omega) if item.peer_position_radius_squared_upper is None
                          else item.peer_position_radius_squared_upper)
            if any(sum(v * v for v in peer.omega) > peer_limit for j, peer in enumerate(builders) if j != index):
                raise ValueError("A builder's assumed peer position radius is smaller than an actual fleet member's radius")
        if len(builders) != len(first.body_ports) or {item.port_index for item in builders} != set(range(len(builders))):
            raise ValueError("Each common body port must have exactly one admitted fleet member")
        if len({item.reference_id for item in builders}) != len(builders):
            raise ValueError("Each fleet member needs a unique reference identity")
        self.builders = builders
        self.controllers = tuple(PhysicalTerminalController(item) for item in builders)
        if any(item.certificate.policy is not self.controllers[0].certificate.policy for item in self.controllers):
            raise ValueError("A physical terminal fleet must share one explicitly supported held policy")

    def admit(self, state_enclosures: Sequence[IntervalBox],
              approximate_references: Sequence[TerminalReferenceSample]) -> PhysicalTerminalAdmission:
        states, references = tuple(state_enclosures), tuple(approximate_references)
        if len(states) != len(self.controllers) or len(references) != len(self.controllers):
            return PhysicalTerminalAdmission(False, ("State/reference count differs from the physical fleet",), None, ())
        anomaly = references[0].anomaly
        if any(reference.anomaly != anomaly for reference in references):
            return PhysicalTerminalAdmission(False, ("Fleet references do not share one actual sample anomaly",), None, ())
        contained = tuple(controller.result.robust_contains(state, reference) for controller, state, reference in
                          zip(self.controllers, states, references))
        reasons = tuple(f"Fleet member {index} is outside its robust physical terminal set"
                        for index, accepted in enumerate(contained) if not accepted)
        return PhysicalTerminalAdmission(not reasons, reasons, anomaly, contained)

    def commands(self, measurements, state_enclosures, approximate_references, *, hold_width,
                 hold_references=None):
        measurements, states, references = tuple(measurements), tuple(state_enclosures), tuple(approximate_references)
        admission = self.admit(states, references)
        if not admission.admissible or len(measurements) != len(self.controllers):
            raise ValueError("Physical terminal fleet admission refused: " + "; ".join(
                admission.reasons or ("Measurement count differs from the physical fleet",)))
        pairs = (None,) * len(self.controllers) if hold_references is None else tuple(hold_references)
        if len(pairs) != len(self.controllers):
            raise ValueError("Held-reference pairs differ from the physical fleet")
        return tuple(controller.command(measured, state, reference, hold_width=hold_width, hold_references=pair)
                     for controller, measured, state, reference, pair in
                     zip(self.controllers, measurements, states, references, pairs))
