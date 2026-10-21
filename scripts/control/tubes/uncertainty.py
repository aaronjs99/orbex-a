"""Componentwise uncertainty bounds for the physical anomaly-domain LVLH model.

Bounds cover the entire eccentricity/drag box and anomaly interval.  Forward
interval differentiation bounds the parameter Jacobian; the mean-value theorem
then bounds deviations from the explicitly supplied nominal parameters.  No
assumption that extrema occur at parameter corners is required.
"""

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


def _down(value):
    return float(np.nextafter(value, -np.inf))


def _up(value):
    return float(np.nextafter(value, np.inf))


@dataclass(frozen=True)
class _Interval:
    lo: float
    hi: float

    @staticmethod
    def cast(value):
        return value if isinstance(value, _Interval) else _Interval(float(value), float(value))

    def __add__(self, other):
        other = self.cast(other)
        if self.lo == self.hi == 0.0:
            return other
        if other.lo == other.hi == 0.0:
            return self
        return _Interval(_down(self.lo + other.lo), _up(self.hi + other.hi))

    __radd__ = __add__

    def __neg__(self):
        return _Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -self.cast(other)

    def __rsub__(self, other):
        return self.cast(other) + -self

    def __mul__(self, other):
        other = self.cast(other)
        if self.lo == self.hi == 0.0 or other.lo == other.hi == 0.0:
            return _Interval(0.0, 0.0)
        products = (self.lo * other.lo, self.lo * other.hi,
                    self.hi * other.lo, self.hi * other.hi)
        return _Interval(_down(min(products)), _up(max(products)))

    __rmul__ = __mul__

    def reciprocal(self):
        if self.lo <= 0.0 <= self.hi:
            raise ValueError("Orbital denominator interval contains zero; subdivide or restrict the domain")
        return _Interval(_down(1.0 / self.hi), _up(1.0 / self.lo))

    def __truediv__(self, other):
        return self * self.cast(other).reciprocal()

    def exp(self):
        return _Interval(max(0.0, _down(math.exp(self.lo))), _up(math.exp(self.hi)))

    def sin(self):
        if self.hi - self.lo >= 2.0 * math.pi:
            return _Interval(-1.0, 1.0)
        lower = min(math.sin(self.lo), math.sin(self.hi))
        upper = max(math.sin(self.lo), math.sin(self.hi))
        # Include all extrema in the interval, with outward-rounded endpoints.
        if math.ceil(_down((self.lo - math.pi / 2.0) / (2.0 * math.pi))) <= math.floor(
            _up((self.hi - math.pi / 2.0) / (2.0 * math.pi))
        ):
            upper = 1.0
        if math.ceil(_down((self.lo + math.pi / 2.0) / (2.0 * math.pi))) <= math.floor(
            _up((self.hi + math.pi / 2.0) / (2.0 * math.pi))
        ):
            lower = -1.0
        return _Interval(max(-1.0, _down(lower)), min(1.0, _up(upper)))

    def cos(self):
        # A direct cosine enclosure avoids widening via a rounded phase shift.
        if self.hi - self.lo >= 2.0 * math.pi:
            return _Interval(-1.0, 1.0)
        lower = min(math.cos(self.lo), math.cos(self.hi))
        upper = max(math.cos(self.lo), math.cos(self.hi))
        if math.ceil(_down(self.lo / (2.0 * math.pi))) <= math.floor(_up(self.hi / (2.0 * math.pi))):
            upper = 1.0
        if math.ceil(_down((self.lo - math.pi) / (2.0 * math.pi))) <= math.floor(
            _up((self.hi - math.pi) / (2.0 * math.pi))
        ):
            lower = -1.0
        return _Interval(max(-1.0, _down(lower)), min(1.0, _up(upper)))

    @property
    def magnitude(self):
        return max(abs(self.lo), abs(self.hi))


_ZERO = _Interval(0.0, 0.0)


@dataclass(frozen=True)
class _Jet:
    value: _Interval
    gradient: tuple

    @staticmethod
    def cast(value):
        return value if isinstance(value, _Jet) else _Jet(_Interval.cast(value), (_ZERO,) * 3)

    @staticmethod
    def variable(interval, index):
        gradient = [_ZERO] * 3
        gradient[index] = _Interval(1.0, 1.0)
        return _Jet(interval, tuple(gradient))

    def __add__(self, other):
        other = self.cast(other)
        return _Jet(self.value + other.value,
                    tuple(a + b for a, b in zip(self.gradient, other.gradient)))

    __radd__ = __add__

    def __neg__(self):
        return _Jet(-self.value, tuple(-value for value in self.gradient))

    def __sub__(self, other):
        return self + -self.cast(other)

    def __rsub__(self, other):
        return self.cast(other) + -self

    def __mul__(self, other):
        other = self.cast(other)
        return _Jet(self.value * other.value, tuple(
            a * other.value + self.value * b
            for a, b in zip(self.gradient, other.gradient)
        ))

    __rmul__ = __mul__

    def reciprocal(self):
        inverse = self.value.reciprocal()
        return _Jet(inverse, tuple(-entry * inverse * inverse for entry in self.gradient))

    def __truediv__(self, other):
        return self * self.cast(other).reciprocal()

    def exp(self):
        value = self.value.exp()
        return _Jet(value, tuple(value * entry for entry in self.gradient))


def _model(q, eccentricity, alpha, beta, semi_major_axis, theta0):
    """Acceleration matrix and affine term, with interval derivative entries."""
    phase = q - theta0
    cosine, sine = phase.cos(), phase.sin()
    exponential = (2.0 * alpha * q).exp()
    denominator = exponential + eccentricity * cosine
    if denominator.value.lo <= 0.0:
        raise ValueError("Orbital denominator must stay positive over the full uncertainty interval")
    k = (eccentricity * sine - 2.0 * alpha * exponential) / denominator
    scale = 1.0 + 4.0 * alpha * alpha
    c = scale * exponential / denominator
    radius = semi_major_axis * (1.0 - eccentricity * eccentricity) * scale / denominator
    gamma = beta - alpha
    zero = _Jet.cast(0.0)
    two = _Jet.cast(2.0)
    matrix = (
        (1.0 - c, gamma - 2.0 * k, zero, 2.0 * k - gamma, two, zero),
        (2.0 * k - gamma, 1.0 + 2.0 * c, zero, -two, 2.0 * k - gamma, zero),
        (zero, zero, -c, zero, zero, 2.0 * k - gamma),
    )
    disturbance = (gamma * radius, -gamma * radius * k, zero)
    return matrix, disturbance


def _domain(start_anom, end_anom, nominal_parameters, semi_major_axis, ranges,
            *, require_nominal_membership=True):
    values = (float(start_anom), float(end_anom), float(semi_major_axis))
    if not all(math.isfinite(value) for value in values) or values[2] <= 0.0:
        raise ValueError("Anomaly endpoints must be finite and semi_major_axis positive")
    if values[1] < values[0]:
        raise ValueError("end_anom must not precede start_anom")
    theta0 = float(nominal_parameters.get("theta0", 0.0))
    if not math.isfinite(theta0):
        raise ValueError("theta0 must be finite")
    boxes, radii = [], []
    for key, bounds in zip(("eccentricity", "alpha", "beta"), ranges):
        lo, hi = map(float, bounds)
        nominal = float(nominal_parameters[key])
        if not all(math.isfinite(value) for value in (lo, hi, nominal)) or lo > hi:
            raise ValueError(f"Invalid {key} uncertainty interval")
        if require_nominal_membership and not lo <= nominal <= hi:
            raise ValueError(f"Nominal {key} must belong to its uncertainty interval")
        boxes.append(_Interval(lo, hi))
        radii.append(0.0 if lo == hi else _up(max(nominal - lo, hi - nominal)))
    if boxes[0].lo < 0.0 or boxes[0].hi >= 1.0:
        raise ValueError("Eccentricity uncertainty must lie in [0, 1)")
    return _Interval(values[0], values[1]), boxes, radii, theta0


def uncertainty_envelope(
    start_anom: float,
    end_anom: float,
    nominal_parameters: Mapping[str, float],
    e_range,
    a_range,
    b_range,
    semi_major_axis: float,
    mu: float,
):
    """Return nonnegative ``(delta_A[3,6], delta_d[3])`` model-error bounds.

    At every common anomaly in the interval, ``abs(A(p)-A(p_nom))`` is bounded
    componentwise by ``delta_A``, and similarly for ``d``.  Here ``A`` denotes
    only its three acceleration rows.  ``mu`` cancels from these normalized
    anomaly-domain coefficients, but is checked as part of the physical domain.
    ``theta0`` is a fixed phase in ``nominal_parameters`` and defaults to zero.
    Zero-width parameter sets produce exact zero uncertainty.

    Arithmetic endpoints are rounded outward using IEEE-754 nextafter.  This is
    an analytic interval enclosure, not a formal machine-checked certificate of
    the host platform's transcendental library.
    """
    if not math.isfinite(float(mu)) or float(mu) <= 0.0:
        raise ValueError("mu must be positive and finite")
    q, boxes, radii, theta0 = _domain(
        start_anom, end_anom, nominal_parameters, semi_major_axis,
        (e_range, a_range, b_range),
    )
    matrix, disturbance = _model(
        q, *(_Jet.variable(box, index) for index, box in enumerate(boxes)),
        float(semi_major_axis), theta0,
    )

    def deviation(entry):
        total = _ZERO
        for derivative, radius in zip(entry.gradient, radii):
            total = total + _Interval.cast(derivative.magnitude) * radius
        return total.hi

    delta_a = np.asarray([[deviation(entry) for entry in row] for row in matrix])
    delta_d = np.asarray([deviation(entry) for entry in disturbance])
    if not np.all(np.isfinite(delta_a)) or not np.all(np.isfinite(delta_d)):
        raise ValueError("Non-finite uncertainty envelope; restrict or subdivide the domain")
    return delta_a, delta_d


def nominal_acceleration_bound(start_anom, end_anom, nominal_parameters, semi_major_axis):
    """Absolute acceleration-matrix envelope over an anomaly interval."""
    ranges = tuple((nominal_parameters[key], nominal_parameters[key])
                   for key in ("eccentricity", "alpha", "beta"))
    q, boxes, _, theta0 = _domain(
        start_anom, end_anom, nominal_parameters, semi_major_axis, ranges
    )
    matrix, _ = _model(q, *(_Jet.cast(box) for box in boxes),
                       float(semi_major_axis), theta0)
    result = np.asarray([[entry.value.magnitude for entry in row] for row in matrix])
    if not np.all(np.isfinite(result)):
        raise ValueError("Non-finite nominal acceleration envelope")
    return result


def coefficient_intervals(
    start_anom, end_anom, nominal_parameters, semi_major_axis,
    *, e_range=None, a_range=None, b_range=None,
):
    """Return ``(A_lo,A_hi,d_lo,d_hi)`` over time and optional parameter boxes.

    A contains the three acceleration rows (3 by 6), and d has three entries.
    All unspecified parameter intervals are the corresponding nominal point.
    The nominal point supplies omitted intervals and the fixed phase. It need
    not lie in explicitly supplied intervals, allowing subdivision contractors
    to enclose candidate boxes away from the controller's point estimate.
    """
    ranges = tuple(
        (nominal_parameters[key], nominal_parameters[key]) if bounds is None else bounds
        for key, bounds in zip(("eccentricity", "alpha", "beta"),
                               (e_range, a_range, b_range))
    )
    q, boxes, _, theta0 = _domain(
        start_anom, end_anom, nominal_parameters, semi_major_axis, ranges,
        require_nominal_membership=False,
    )
    matrix, disturbance = _model(q, *(_Jet.cast(box) for box in boxes),
                                 float(semi_major_axis), theta0)
    result = (
        np.asarray([[entry.value.lo for entry in row] for row in matrix]),
        np.asarray([[entry.value.hi for entry in row] for row in matrix]),
        np.asarray([entry.value.lo for entry in disturbance]),
        np.asarray([entry.value.hi for entry in disturbance]),
    )
    if not all(np.all(np.isfinite(value)) for value in result):
        raise ValueError("Non-finite coefficient intervals; restrict or subdivide the domain")
    return result
