"""Immutable commanded orbit coordinates; these are not the perturbed LVLH.

The reference rate defines dt/dq. Adaptive controller parameters never redefine
this clock. Floats in declarations denote their exact binary rational values.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
import hashlib
import json
import math
from pathlib import Path

_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def exact(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a physical declaration')
    if isinstance(value, Fraction):
        return value
    if isinstance(value, str):
        return Fraction(value)
    if hasattr(value, 'as_integer_ratio'):
        return Fraction(*value.as_integer_ratio())
    return Fraction(value)


def bounds(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError('A scalar interval requires two endpoints')
        lo, hi = map(exact, value)
    else:
        lo = hi = exact(value)
    if lo > hi:
        raise ValueError('Interval endpoints are reversed')
    return lo, hi


def fingerprint(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


class FloatArithmetic:
    scalar = staticmethod(float)
    sqrt = staticmethod(math.sqrt)
    exp = staticmethod(math.exp)
    expm1 = staticmethod(math.expm1)
    sin = staticmethod(math.sin)
    cos = staticmethod(math.cos)
    square = staticmethod(lambda value: value*value)


@dataclass(frozen=True)
class ReferenceFramePoint:
    rotation: tuple
    rate_s_inv: float
    kappa: float


@dataclass(frozen=True)
class ReferenceClockFrame:
    semi_major_axis_m: object = 7_000_000
    eccentricity: object = 0
    alpha: object = 0
    mu_m3_s2: object = '398600441800000'
    phase: object = 0
    epoch_orientation: tuple = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    q_domain: tuple = (0, 100)

    def __post_init__(self):
        for name in ('semi_major_axis_m', 'eccentricity', 'alpha', 'mu_m3_s2', 'phase'):
            object.__setattr__(self, name, exact(getattr(self, name)))
        if (self.semi_major_axis_m <= 0 or self.mu_m3_s2 <= 0 or
                not 0 <= self.eccentricity < 1 or self.alpha < 0):
            raise ValueError('Invalid reference ellipse/rate declaration')
        orientation = tuple(tuple(map(exact, row)) for row in self.epoch_orientation)
        if len(orientation) != 3 or any(len(row) != 3 for row in orientation):
            raise ValueError('Epoch orientation must be 3 by 3')
        if any(sum(orientation[k][i]*orientation[k][j] for k in range(3)) != int(i == j)
               for i in range(3) for j in range(3)):
            raise ValueError('Supply an exactly proper rational epoch rotation')
        determinant = sum(orientation[0][i]*(orientation[1][(i+1)%3]*orientation[2][(i+2)%3]
                          - orientation[1][(i+2)%3]*orientation[2][(i+1)%3]) for i in range(3))
        if determinant != 1:
            raise ValueError('Epoch orientation must have determinant +1')
        object.__setattr__(self, 'epoch_orientation', orientation)
        object.__setattr__(self, 'q_domain', bounds(self.q_domain))

    def as_dict(self):
        return dict(model='commanded-reference-orbit-clock-v1',
            source_sha256=_SOURCE_SHA256,
            frame='reference-orbit-frame-v1',
            **{name: str(getattr(self, name)) for name in
               ('semi_major_axis_m', 'eccentricity', 'alpha', 'mu_m3_s2', 'phase')},
            epoch_orientation=tuple(tuple(map(str, row)) for row in self.epoch_orientation),
            q_domain=tuple(map(str, self.q_domain)))

    @cached_property
    def fingerprint(self):
        return fingerprint(self.as_dict())

    def check_domain(self, q_interval):
        lo, hi = bounds(q_interval)
        if lo < self.q_domain[0] or hi > self.q_domain[1]:
            raise ValueError('Reference clock query is outside its finite q domain')
        return lo, hi

    def evaluate(self, q, arithmetic):
        """Same algebra for binary64 evaluation and outward Arb evaluation."""
        s = arithmetic.scalar
        e, alpha, mu, axis, phase = map(s, (self.eccentricity, self.alpha,
            self.mu_m3_s2, self.semi_major_axis_m, self.phase))
        exponential = arithmetic.exp(2*alpha*q)
        denominator = exponential + e*arithmetic.cos(q-phase)
        if not denominator > 0:
            raise ValueError('Reference orbital denominator is not proved positive')
        h2 = mu*axis*(1-e*e)
        radius = h2*(1+4*alpha*alpha)/(mu*denominator)
        k = (e*arithmetic.sin(q-phase)-2*alpha*exponential)/denominator
        rate = arithmetic.sqrt(h2)*arithmetic.exp(-alpha*q)/(radius*radius)
        if not rate > 0:
            raise ValueError('Reference rate is not proved positive')
        cosine, sine, zero, one = arithmetic.cos(q), arithmetic.sin(q), s(0), s(1)
        rz = ((cosine, -sine, zero), (sine, cosine, zero), (zero, zero, one))
        rotation = tuple(tuple(sum(s(self.epoch_orientation[i][k])*rz[k][j]
                              for k in range(3)) for j in range(3)) for i in range(3))
        return rotation, rate, -alpha-2*k

    def point(self, q):
        self.check_domain((q, q))
        rotation, rate, kappa = self.evaluate(float(q), FloatArithmetic())
        return ReferenceFramePoint(rotation, rate, kappa)

    def interval_bounds(self, q_interval, precision_bits=128):
        """Return exact outward boxes for rotation entries, rate and kappa."""
        from orbexa.control.certification.reachability import _ARB_LOCK
        from orbexa.control.certification.nonlinear import _Arithmetic
        import flint
        interval = self.check_domain(q_interval)
        with _ARB_LOCK, flint.ctx.workprec(precision_bits):
            ar = _Arithmetic(flint)
            rotation, rate, kappa = self.evaluate(ar.interval(interval), ar)
            return dict(rotation=ar.box(tuple(v for row in rotation for v in row)),
                        rate_s_inv=ar.box((rate,)), kappa=ar.box((kappa,)),
                        clock_fingerprint=self.fingerprint)

    def elapsed_bounds(self, q_start, q_end, *, max_step='0.002', max_segments=10000, precision_bits=128):
        from orbexa.control.certification.reachability import IntervalBox, _ARB_LOCK
        from orbexa.control.certification.nonlinear import _Arithmetic
        import flint
        start, end, step = exact(q_start), exact(q_end), exact(max_step)
        self.check_domain((start, end))
        if step <= 0 or isinstance(max_segments, bool) or max_segments < 1:
            raise ValueError('Clock quadrature requires finite positive step and segment budget')
        ratio = (end-start)/step
        pieces = max(1, (ratio.numerator+ratio.denominator-1)//ratio.denominator)
        if pieces > max_segments:
            raise ValueError('Clock quadrature segment budget exhausted')
        width, lower, upper = (end-start)/pieces, exact(0), exact(0)
        with _ARB_LOCK, flint.ctx.workprec(precision_bits):
            ar = _Arithmetic(flint)
            for index in range(pieces):
                q = ar.interval((start+index*width, start+(index+1)*width))
                increment = ar.box((ar.scalar(width)/self.evaluate(q, ar)[1],))
                lower += increment.lower[0]
                upper += increment.upper[0]
        return IntervalBox((lower,), (upper,))

    def to_reference(self, cartesian_states, q, target_state):
        import numpy as np
        point = self.point(q)
        states = np.atleast_2d(np.asarray(cartesian_states, dtype=float))
        target = np.asarray(target_state, dtype=float)
        if states.shape[1] != 6 or target.shape != (6,) or not np.all(np.isfinite(states)) or not np.all(np.isfinite(target)):
            raise ValueError('Finite inertial states have six components')
        rotation = np.asarray(point.rotation)
        p = (states[:, :3]-target[:3]) @ rotation
        relative_v = (states[:, 3:]-target[3:]) @ rotation / point.rate_s_inv
        zp = np.column_stack((-p[:, 1], p[:, 0], np.zeros(len(p))))
        return np.concatenate((p, relative_v-zp), axis=1)

    def to_inertial(self, relative_states, q, target_state):
        import numpy as np
        relative = np.atleast_2d(np.asarray(relative_states, dtype=float))
        target = np.asarray(target_state, dtype=float)
        if relative.shape[1] != 6 or target.shape != (6,) or not np.all(np.isfinite(relative)) or not np.all(np.isfinite(target)):
            raise ValueError('Finite relative/target states have six components')
        point = self.point(q)
        rotation = np.asarray(point.rotation)
        p, w = relative[:, :3], relative[:, 3:]
        zp = np.column_stack((-p[:, 1], p[:, 0], np.zeros(len(p))))
        return np.concatenate((target[:3]+p@rotation.T,
            target[3:]+point.rate_s_inv*(w+zp)@rotation.T), axis=1)
