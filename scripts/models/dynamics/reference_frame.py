"""reference_frame.py: Immutable commanded orbit coordinates; these are not the perturbed LVLH.

The reference rate defines dt/dq. Adaptive controller parameters never redefine
this clock. Floats in declarations denote their exact binary rational values.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from pathlib import Path

_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def exact(value):
    """Convert a numeric or string declaration to an exact ``Fraction``.

    Inputs: ``value`` is a Fraction, str, int, float or other Fraction-compatible
    number. A float converts to its exact binary rational value.
    Outputs: the equivalent ``Fraction``. Raises ValueError for a bool."""
    if isinstance(value, bool):
        raise ValueError("Boolean is not a physical declaration")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, str):
        return Fraction(value)
    if hasattr(value, "as_integer_ratio"):
        return Fraction(*value.as_integer_ratio())
    return Fraction(value)


def bounds(value):
    """Return the exact ``(lo, hi)`` endpoints of a scalar interval.

    Inputs: ``value`` is a scalar, giving ``lo == hi``, or a two-element tuple or
    list of endpoints.
    Outputs: tuple of two ``Fraction`` values. Raises ValueError when a sequence
    does not hold two endpoints or when ``lo > hi``."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("A scalar interval requires two endpoints")
        lo, hi = map(exact, value)
    else:
        lo = hi = exact(value)
    if lo > hi:
        raise ValueError("Interval endpoints are reversed")
    return lo, hi


def fingerprint(record):
    """Return the SHA-256 hex digest of the canonical JSON encoding of ``record``.

    The encoding uses sorted keys and compact separators. Raises ValueError when
    ``record`` contains NaN or infinity."""
    return hashlib.sha256(
        json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class FloatArithmetic:
    """Binary64 arithmetic namespace with the interface ``evaluate`` expects.

    Provides ``scalar``, ``sqrt``, ``exp``, ``expm1``, ``sin``, ``cos`` and
    ``square`` as float operations, mirroring the interval arithmetic used for
    outward evaluation."""

    scalar = staticmethod(float)
    sqrt = staticmethod(math.sqrt)
    exp = staticmethod(math.exp)
    expm1 = staticmethod(math.expm1)
    sin = staticmethod(math.sin)
    cos = staticmethod(math.cos)
    square = staticmethod(lambda value: value * value)


@dataclass(frozen=True)
class ReferenceFramePoint:
    """Reference frame quantities evaluated at one value of q.

    ``rotation`` is the 3 by 3 rotation (nested tuples of floats) mapping reference
    frame components to inertial components. ``rate_s_inv`` is dq/dt in 1/s.
    ``kappa`` is the value ``-alpha - 2k`` returned by ``evaluate``."""

    rotation: tuple
    rate_s_inv: float
    kappa: float


@dataclass(frozen=True)
class ReferenceClockFrame:
    """Exact declaration of the commanded reference orbit and its q clock.

    The independent variable q is the unwrapped anomaly in rad. Fields:
    ``semi_major_axis_m`` (m), ``eccentricity``, ``alpha`` (drag decay exponent
    per unit q), ``mu_m3_s2`` (m^3/s^2), ``phase`` (rad), ``epoch_orientation``
    (3 by 3 rotation at q = 0) and ``q_domain`` (``(lo, hi)`` rad interval on
    which queries are allowed). Numeric fields are converted to exact Fractions."""

    semi_major_axis_m: object = 7_000_000
    eccentricity: object = 0
    alpha: object = 0
    mu_m3_s2: object = "398600441800000"
    phase: object = 0
    epoch_orientation: tuple = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    q_domain: tuple = (0, 100)

    def __post_init__(self):
        """Convert the numeric fields to exact values and validate the declaration.

        State changes: replaces ``semi_major_axis_m``, ``eccentricity``, ``alpha``,
        ``mu_m3_s2`` and ``phase`` with Fractions, ``epoch_orientation`` with rows of
        Fractions and ``q_domain`` with a ``(lo, hi)`` Fraction pair.
        Outputs: raises ValueError when the semi-major axis or ``mu_m3_s2`` is not
        positive, the eccentricity is outside [0, 1), ``alpha`` is negative, the
        orientation is not 3 by 3, not exactly orthonormal or has determinant not
        equal to 1, or ``q_domain`` is not a valid interval."""
        for name in ("semi_major_axis_m", "eccentricity", "alpha", "mu_m3_s2", "phase"):
            object.__setattr__(self, name, exact(getattr(self, name)))
        if (
            self.semi_major_axis_m <= 0
            or self.mu_m3_s2 <= 0
            or not 0 <= self.eccentricity < 1
            or self.alpha < 0
        ):
            raise ValueError("Invalid reference ellipse/rate declaration")
        orientation = tuple(tuple(map(exact, row)) for row in self.epoch_orientation)
        if len(orientation) != 3 or any(len(row) != 3 for row in orientation):
            raise ValueError("Epoch orientation must be 3 by 3")
        if any(
            sum(orientation[k][i] * orientation[k][j] for k in range(3)) != int(i == j)
            for i in range(3)
            for j in range(3)
        ):
            raise ValueError("Supply an exactly proper rational epoch rotation")
        determinant = sum(
            orientation[0][i]
            * (
                orientation[1][(i + 1) % 3] * orientation[2][(i + 2) % 3]
                - orientation[1][(i + 2) % 3] * orientation[2][(i + 1) % 3]
            )
            for i in range(3)
        )
        if determinant != 1:
            raise ValueError("Epoch orientation must have determinant +1")
        object.__setattr__(self, "epoch_orientation", orientation)
        object.__setattr__(self, "q_domain", bounds(self.q_domain))

    def as_dict(self):
        """Return a JSON-serializable record of the declaration.

        The record holds the model name, the SHA-256 of this source file, the frame
        name and every declared field with numbers rendered as strings."""
        return dict(
            model="commanded-reference-orbit-clock-v1",
            source_sha256=_SOURCE_SHA256,
            frame="reference-orbit-frame-v1",
            **{
                name: str(getattr(self, name))
                for name in (
                    "semi_major_axis_m",
                    "eccentricity",
                    "alpha",
                    "mu_m3_s2",
                    "phase",
                )
            },
            epoch_orientation=tuple(
                tuple(map(str, row)) for row in self.epoch_orientation
            ),
            q_domain=tuple(map(str, self.q_domain)),
        )

    @cached_property
    def fingerprint(self):
        """Return the SHA-256 hex digest of ``as_dict()``."""
        return fingerprint(self.as_dict())

    def check_domain(self, q_interval):
        """Return the exact bounds of ``q_interval`` after checking the q domain.

        Inputs: ``q_interval`` is a scalar or ``(lo, hi)`` pair of q values in rad.
        Outputs: ``(lo, hi)`` as Fractions. Raises ValueError when the interval is
        malformed or lies outside ``q_domain``."""
        lo, hi = bounds(q_interval)
        if lo < self.q_domain[0] or hi > self.q_domain[1]:
            raise ValueError("Reference clock query is outside its finite q domain")
        return lo, hi

    def evaluate(self, q, arithmetic):
        """Same algebra for binary64 evaluation and outward Arb evaluation."""
        s = arithmetic.scalar
        e, alpha, mu, axis, phase = map(
            s,
            (
                self.eccentricity,
                self.alpha,
                self.mu_m3_s2,
                self.semi_major_axis_m,
                self.phase,
            ),
        )
        exponential = arithmetic.exp(2 * alpha * q)
        denominator = exponential + e * arithmetic.cos(q - phase)
        if not denominator > 0:
            raise ValueError("Reference orbital denominator is not proved positive")
        h2 = mu * axis * (1 - e * e)
        radius = h2 * (1 + 4 * alpha * alpha) / (mu * denominator)
        k = (e * arithmetic.sin(q - phase) - 2 * alpha * exponential) / denominator
        rate = arithmetic.sqrt(h2) * arithmetic.exp(-alpha * q) / (radius * radius)
        if not rate > 0:
            raise ValueError("Reference rate is not proved positive")
        cosine, sine, zero, one = arithmetic.cos(q), arithmetic.sin(q), s(0), s(1)
        rz = ((cosine, -sine, zero), (sine, cosine, zero), (zero, zero, one))
        rotation = tuple(
            tuple(
                sum(s(self.epoch_orientation[i][k]) * rz[k][j] for k in range(3))
                for j in range(3)
            )
            for i in range(3)
        )
        return rotation, rate, -alpha - 2 * k

    def point(self, q):
        """Evaluate the reference frame at a single q in binary64.

        Inputs: ``q`` is the unwrapped anomaly in rad inside ``q_domain``.
        Outputs: ``ReferenceFramePoint``. Raises ValueError when ``q`` is outside
        ``q_domain`` or the orbital denominator or rate is not positive."""
        self.check_domain((q, q))
        rotation, rate, kappa = self.evaluate(float(q), FloatArithmetic())
        return ReferenceFramePoint(rotation, rate, kappa)

    def interval_bounds(self, q_interval, precision_bits=128):
        """Return exact outward boxes for rotation entries, rate and kappa."""
        import flint
        from orbexa.common.intervals import _ARB_LOCK, _Arithmetic

        interval = self.check_domain(q_interval)
        with _ARB_LOCK, flint.ctx.workprec(precision_bits):
            ar = _Arithmetic(flint)
            rotation, rate, kappa = self.evaluate(ar.interval(interval), ar)
            return {
                "rotation": ar.box(tuple(v for row in rotation for v in row)),
                "rate_s_inv": ar.box((rate,)),
                "kappa": ar.box((kappa,)),
                "clock_fingerprint": self.fingerprint,
            }

    def elapsed_bounds(
        self,
        q_start,
        q_end,
        *,
        max_step="0.002",
        max_segments=10000,
        precision_bits=128,
    ):
        """Return outward bounds on the physical time elapsed between two q values.

        Sums, over equal q pieces no wider than ``max_step``, the bounds on
        width / (dq/dt) evaluated on each piece with interval arithmetic.
        Inputs: ``q_start`` and ``q_end`` are anomalies in rad inside ``q_domain``;
        ``max_step`` is the largest piece width in rad; ``max_segments`` is the
        largest allowed number of pieces; ``precision_bits`` is the working precision.
        Outputs: ``IntervalBox`` with one component, in s. Raises ValueError when a
        query is outside ``q_domain``, ``max_step`` is not positive, ``max_segments``
        is a bool or below 1, or more than ``max_segments`` pieces are needed."""
        import flint
        from orbexa.common.intervals import _ARB_LOCK, IntervalBox, _Arithmetic

        start, end, step = exact(q_start), exact(q_end), exact(max_step)
        self.check_domain((start, end))
        if step <= 0 or isinstance(max_segments, bool) or max_segments < 1:
            raise ValueError(
                "Clock quadrature requires finite positive step and segment budget"
            )
        ratio = (end - start) / step
        pieces = max(1, (ratio.numerator + ratio.denominator - 1) // ratio.denominator)
        if pieces > max_segments:
            raise ValueError("Clock quadrature segment budget exhausted")
        width, lower, upper = (end - start) / pieces, exact(0), exact(0)
        with _ARB_LOCK, flint.ctx.workprec(precision_bits):
            ar = _Arithmetic(flint)
            for index in range(pieces):
                q = ar.interval((start + index * width, start + (index + 1) * width))
                increment = ar.box((ar.scalar(width) / self.evaluate(q, ar)[1],))
                lower += increment.lower[0]
                upper += increment.upper[0]
        return IntervalBox((lower,), (upper,))

    def to_inertial(self, relative_states, q, target_state):
        """Convert relative states in the reference frame to inertial states.

        Inputs: ``relative_states`` is a (6,) or (N, 6) array of relative position (m)
        and relative velocity per unit q (m/rad) in the reference frame; ``q`` is the
        unwrapped anomaly in rad; ``target_state`` is the (6,) inertial target position
        (m) and velocity (m/s).
        Outputs: (N, 6) array of inertial position (m) and velocity (m/s). Raises
        ValueError when an input is non-finite or has the wrong shape, or when ``q`` is
        outside ``q_domain``."""
        import numpy as np

        relative = np.atleast_2d(np.asarray(relative_states, dtype=float))
        target = np.asarray(target_state, dtype=float)
        if (
            relative.shape[1] != 6
            or target.shape != (6,)
            or not np.all(np.isfinite(relative))
            or not np.all(np.isfinite(target))
        ):
            raise ValueError("Finite relative/target states have six components")
        point = self.point(q)
        rotation = np.asarray(point.rotation)
        p, w = relative[:, :3], relative[:, 3:]
        zp = np.column_stack((-p[:, 1], p[:, 0], np.zeros(len(p))))
        return np.concatenate(
            (
                target[:3] + p @ rotation.T,
                target[3:] + point.rate_s_inv * (w + zp) @ rotation.T,
            ),
            axis=1,
        )
