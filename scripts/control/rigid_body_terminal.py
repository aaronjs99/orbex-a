"""Validated componentwise Euler and fixed-body-port derivative enclosures.

The supplied orbital rate, torque and elapsed-time bounds must hold globally.
Spin inclusion is a strict bidirectional first-exit comparison, not a sampled
maximum or a contraction claim. Exact polynomial coefficients retain full
off-diagonal inertia and combine cancellations before interval evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral

from .physical_terminal import _encode, _inertia_bounds, _sqrt_upper, _vector
from .reachability import IntervalBox, _ARB_LOCK, _ArbArithmetic, _exact


RIGID_BODY_TERMINAL_METHOD = "arb-componentwise-euler-port-jets-v1"
_ZERO_POWER = (0,) * 9


def _inverse(matrix):
    size = len(matrix)
    rows = [list(row) + [Fraction(int(i == j)) for j in range(size)] for i, row in enumerate(matrix)]
    for i in range(size):
        pivot = next(j for j in range(i, size) if rows[j][i])
        rows[i], rows[pivot] = rows[pivot], rows[i]
        factor = rows[i][i]
        rows[i] = [value / factor for value in rows[i]]
        for j in range(size):
            if j != i:
                factor = rows[j][i]
                rows[j] = [a - factor * b for a, b in zip(rows[j], rows[i])]
    return tuple(tuple(row[size:]) for row in rows)


def _euler_coefficients(inertia, inverse):
    coefficients = [{} for _ in range(3)]
    for row, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
        for column in range(3):
            for i in range(3):
                for left, right, value in ((a, column, -inverse[i][row] * inertia[b][column]),
                                            (b, column, inverse[i][row] * inertia[a][column])):
                    key = tuple(sorted((left, right)))
                    coefficients[i][key] = coefficients[i].get(key, Fraction(0)) + value
    return tuple({key: value for key, value in row.items() if value} for row in coefficients)


def _euler(ar, coefficients, omega, force):
    return tuple(sum((ar.scalar(value) * omega[j] * omega[k] for (j, k), value in row.items()), force[i])
                 for i, row in enumerate(coefficients))


def _jacobian(ar, coefficients, omega):
    result = []
    for row in coefficients:
        derivative = []
        for i in range(3):
            value = ar.arb(0)
            for (j, k), coefficient in row.items():
                if j == i:
                    value += ar.scalar(coefficient) * omega[k]
                if k == i:
                    value += ar.scalar(coefficient) * omega[j]
            derivative.append(value)
        result.append(tuple(derivative))
    return tuple(result)


def _norm(ar, values):
    box = ar.box(values)
    return _sqrt_upper(ar, sum(max(abs(lo), abs(hi)) ** 2 for lo, hi in zip(box.lower, box.upper)))


def _constant(value):
    return {_ZERO_POWER: value} if value else {}


def _variable(index):
    power = list(_ZERO_POWER)
    power[index] = 1
    return {tuple(power): Fraction(1)}


def _add(*items):
    result = {}
    for item in items:
        for power, value in item.items():
            result[power] = result.get(power, Fraction(0)) + value
    return {power: value for power, value in result.items() if value}


def _scale(item, value):
    return {power: coefficient * value for power, coefficient in item.items() if coefficient * value}


def _multiply(left, right):
    result = {}
    for power_a, value_a in left.items():
        for power_b, value_b in right.items():
            power = tuple(a + b for a, b in zip(power_a, power_b))
            result[power] = result.get(power, Fraction(0)) + value_a * value_b
    return {power: value for power, value in result.items() if value}


def _differentiate(item, index):
    result = {}
    for power, value in item.items():
        if power[index]:
            reduced = list(power)
            reduced[index] -= 1
            result[tuple(reduced)] = value * power[index]
    return result


def _cross(left, right):
    return tuple(_add(_multiply(left[j], right[k]), _scale(_multiply(left[k], right[j]), -1))
                 for j, k in ((1, 2), (2, 0), (0, 1)))


def _evaluate(ar, item, variables):
    result = ar.arb(0)
    for power, value in item.items():
        term = ar.scalar(value)
        for variable, exponent in zip(variables, power):
            # General interval exponentiation can be indeterminate across zero;
            # these nonnegative integer powers need only multiplication.
            for _ in range(exponent):
                term *= variable
        result += term
    return result


def _port_jets(ar, coefficients, inverse, omega, torque, torque_dot, point):
    w = tuple(_variable(i) for i in range(3))
    t, td = tuple(_variable(i + 3) for i in range(3)), tuple(_variable(i + 6) for i in range(3))
    g = tuple(_constant(value) for value in point)
    f = tuple(_add(*(_scale(_multiply(w[j], w[k]), value) for (j, k), value in row.items()),
                   *(_scale(t[j], inverse[i][j]) for j in range(3))) for i, row in enumerate(coefficients))
    fd = tuple(_add(*(_multiply(_differentiate(f[i], j), f[j]) for j in range(3)),
                    *(_scale(td[j], inverse[i][j]) for j in range(3))) for i in range(3))
    velocity = _cross(w, g)
    acceleration = tuple(_add(a, b) for a, b in zip(_cross(f, g), _cross(w, velocity)))
    jerk = tuple(_add(a, b, _scale(c, 2), d) for a, b, c, d in zip(
        _cross(fd, g), _cross(f, velocity), _cross(w, _cross(f, g)), _cross(w, _cross(w, velocity))))
    variables = tuple(omega) + tuple(torque) + tuple(torque_dot)
    jets = (velocity, acceleration, jerk)
    values = tuple(tuple(_evaluate(ar, item, variables) for item in jet) for jet in jets)
    exact = tuple(tuple({str(power): value for power, value in item.items()} for item in jet) for jet in jets)
    return values, exact


@dataclass(frozen=True)
class ComponentwiseTerminalJetBounds:
    reference_norm_bounds: tuple[Fraction, Fraction, Fraction, Fraction]
    evidence: dict
    method: str = RIGID_BODY_TERMINAL_METHOD

    def as_dict(self):
        return _encode(vars(self))


def bound_componentwise_terminal_jets(*, inertia, entry_spin: IntervalBox, body_port,
        physical_duration_upper, anomaly_rate: IntervalBox, anomaly_rate_prime_abs,
        anomaly_rate_double_prime_abs, torque_factor_upper, radius_log_derivative_abs,
        precision_bits=160) -> ComponentwiseTerminalJetBounds:
    """Prove port jets conditional on the supplied complete-domain bounds.

    ``physical_duration_upper`` bounds absolute time from entry to every query,
    including backward queries. Gravity gradient is the only torque here:
    |tau| <= torque_factor_upper * spectral_spread / 2. Set its factor to zero
    for torque-free motion; additional torques require a different derivation.
    """
    inertia, jmin, jmax = _inertia_bounds(inertia)
    point = _vector(body_port, 3, "Fixed body port")
    duration, prime, double, factor, radial = map(_exact, (physical_duration_upper,
        anomaly_rate_prime_abs, anomaly_rate_double_prime_abs, torque_factor_upper, radius_log_derivative_abs))
    if (not isinstance(entry_spin, IntervalBox) or entry_spin.dimension != 3
            or not isinstance(anomaly_rate, IntervalBox) or anomaly_rate.dimension != 1
            or anomaly_rate.lower[0] <= 0 or min(duration, prime, double, factor, radial) < 0):
        raise ValueError("Use a spin box, positive complete-domain rate and nonnegative global derivative/time bounds")
    if isinstance(precision_bits, bool) or not isinstance(precision_bits, Integral) or precision_bits < 64:
        raise ValueError("Use at least 64 interval precision bits")
    import flint
    with _ARB_LOCK, flint.ctx.workprec(int(precision_bits)):
        ar = _ArbArithmetic(flint)
        inverse, spread = _inverse(inertia), jmax - jmin
        coefficients = _euler_coefficients(inertia, inverse)
        torque_norm = factor * spread / 2
        torque = ar.vector(IntervalBox((-torque_norm,) * 3, (torque_norm,) * 3))
        force = tuple(sum((ar.scalar(inverse[i][j]) * torque[j] for j in range(3)), ar.arb(0)) for i in range(3))
        center = tuple((lo + hi) / 2 for lo, hi in zip(entry_spin.lower, entry_spin.upper))
        initial_radius = tuple((hi - lo) / 2 for lo, hi in zip(entry_spin.lower, entry_spin.upper))
        seed = ar.box(_euler(ar, coefficients, ar.vector(entry_spin), force))
        widths = tuple(r + duration * max(abs(lo), abs(hi)) * Fraction(6, 5) + Fraction(1, 10 ** 14)
                       for r, lo, hi in zip(initial_radius, seed.lower, seed.upper))
        history = []
        for iteration in range(48):
            box = IntervalBox.from_center_radius(center, widths)
            omega = ar.vector(box)
            f = _euler(ar, coefficients, omega, force)
            fbox = ar.box(f)
            needed = tuple(r + duration * max(abs(lo), abs(hi))
                           for r, lo, hi in zip(initial_radius, fbox.lower, fbox.upper))
            margins = tuple(width - value for width, value in zip(widths, needed))
            history.append(dict(iteration=iteration, widths=widths, needed=needed, strict_margins=margins))
            if min(margins) > 0:
                break
            widths = tuple(max(width, value) * Fraction(6, 5) + Fraction(1, 10 ** 14)
                           for width, value in zip(widths, needed))
        else:
            raise ValueError("Componentwise Euler first-exit inclusion failed within 48 declared iterations")
        spin_norm, f_norm = _norm(ar, omega), _norm(ar, f)
        factor_dot = 3 * factor * radial * anomaly_rate.upper[0]
        torque_dot_norm = factor_dot * spread / 2 + factor * spread * (spin_norm + anomaly_rate.upper[0])
        torque_dot = ar.vector(IntervalBox((-torque_dot_norm,) * 3, (torque_dot_norm,) * 3))
        jacobian = _jacobian(ar, coefficients, omega)
        fd = tuple(sum((jacobian[i][j] * f[j] + ar.scalar(inverse[i][j]) * torque_dot[j]
                        for j in range(3)), ar.arb(0)) for i in range(3))
        (velocity, acceleration, jerk), port_polynomials = _port_jets(
            ar, coefficients, inverse, omega, torque, torque_dot, point)
        v, a, j = (_norm(ar, values) for values in (velocity, acceleration, jerk))
        nu = anomaly_rate.lower[0]
        x1, x2 = v / nu, (a + prime * v) / nu ** 2
        x3 = (j + 3 * prime * a + 2 * prime * prime * v) / nu ** 3 + double * v / nu ** 2
        p0 = _sqrt_upper(ar, sum(value * value for value in point))
        p1, p2, p3 = p0 + x1, p0 + 2 * x1 + x2, p0 + 3 * x1 + 3 * x2 + x3
        evidence = dict(method=RIGID_BODY_TERMINAL_METHOD, physical_duration_upper=duration, entry_spin=entry_spin,
            component_omega_box=box, component_omega_dot_box=fbox, component_omega_ddot_box=ar.box(fd),
            exact_inertia_inverse=inverse, expanded_port_polynomial_coefficients=port_polynomials,
            polynomial_coefficients=tuple({str(key): value for key, value in row.items()} for row in coefficients),
            strict_self_inclusion_history=history, torque_norm_upper=torque_norm,
            torque_derivative_norm_upper=torque_dot_norm, omega_norm_upper=spin_norm,
            omega_dot_norm_upper=f_norm, omega_ddot_norm_upper=_norm(ar, fd),
            body_port_velocity_box=ar.box(velocity), body_port_acceleration_box=ar.box(acceleration),
            body_port_jerk_box=ar.box(jerk), physical_port_derivative_norm_bounds=(v, a, j),
            inertial_anomaly_derivative_norm_bounds=(x1, x2, x3), lvlh_anomaly_derivative_norm_bounds=(p1, p2, p3),
            rotation_scope="Body-to-inertial rotation preserves norms; planar LVLH generator has operator norm one",
            torque_scope="Fixed full inertia; central spherical gravity-gradient torque only",
            precision_bits=int(precision_bits), sampled_maxima_used=False)
        return ComponentwiseTerminalJetBounds((p0, p1, p2, p3), evidence)
