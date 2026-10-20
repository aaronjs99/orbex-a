"""Bounded-error identification from three positions and fixed actuator cells.

For position p in the declared anomaly clock, write the exact relative model as

    p'' + (2 Z + kappa I) p' + kappa Z p
        = H + sigma_target d_target + sigma_chaser d_chaser + M u + b + epsilon,

where Z p = (-p_y, p_x, 0), H is normalized differential gravity minus Z^2 p,
and M,b are the same fixed actuator parameters throughout a context. A tent
kernel vanishing at three-position arc endpoints removes initial velocity.
Integration by parts puts explicit Coriolis/clock terms on bounded positions.

This module proves only the conditional integral and parameter-set algebra.
Callers must provide sound whole-segment coefficient/state/error enclosures,
the actual delivered command, measurement-event boxes, and a context binding
the model, clock, sensing, source, and fixed parameter/actuator epoch. Source
identifiers record provenance; they do not certify those physical obligations.
No stochastic independence, parameter excitation, or useful contraction is
assumed. A feasible outer relaxation does not prove a physical trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from uuid import uuid4

from orbexa.core.reference_frame import bounds, exact, fingerprint


def _add(a, b):
    return a[0]+b[0], a[1]+b[1]


def _negative(a):
    return -a[1], -a[0]


def _multiply(a, b):
    products = tuple(x*y for x in a for y in b)
    return min(products), max(products)


def _scale(a, coefficient):
    coefficient = exact(coefficient)
    return _multiply(a, (coefficient, coefficient))


def _vector(values, count=3):
    result = tuple(bounds(value) for value in values)
    if len(result) != count:
        raise ValueError(f'Expected {count} bounded components')
    return result


def _identifier(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('A nonempty source/context/parameter identity is required')
    return value


def _serial_intervals(values):
    return tuple(tuple(map(str, pair)) for pair in values)


@dataclass(frozen=True)
class IntegralCoefficientSegment:
    """Conditional bounds on one complete physical hold, in a single clock.

    Position and H/d_target/d_chaser are three scalar intervals each. kappa and
    its anomaly derivative are scalar intervals. ``additive_error_abs`` covers
    every remaining acceleration error, including delivered-input rounding.
    The exact command is the input before the persistent affine map M u+b.
    None of these boxes may be obtained by merely sampling the hold endpoints.
    """
    start: Fraction
    end: Fraction
    position: tuple
    centered_gravity: tuple
    target_drag: tuple
    chaser_drag: tuple
    kappa: tuple
    kappa_prime: tuple
    command: tuple
    additive_error_abs: tuple
    source_id: str
    context_id: str

    def __post_init__(self):
        for name in ('start', 'end'):
            object.__setattr__(self, name, exact(getattr(self, name)))
        for name in ('position', 'centered_gravity', 'target_drag', 'chaser_drag'):
            object.__setattr__(self, name, _vector(getattr(self, name)))
        for name in ('kappa', 'kappa_prime'):
            object.__setattr__(self, name, bounds(getattr(self, name)))
        for name in ('command', 'additive_error_abs'):
            values = tuple(map(exact, getattr(self, name)))
            if len(values) != 3:
                raise ValueError('Three command/error components are required')
            object.__setattr__(self, name, values)
        if self.end <= self.start or any(value < 0 for value in self.additive_error_abs):
            raise ValueError('A positive hold and nonnegative error bounds are required')
        _identifier(self.source_id)
        _identifier(self.context_id)

    @property
    def fingerprint(self):
        return fingerprint(dict(context_id=self.context_id, source_id=self.source_id,
            start=str(self.start), end=str(self.end),
            **{name: _serial_intervals(getattr(self, name)) for name in
                ('position', 'centered_gravity', 'target_drag', 'chaser_drag')},
            **{name: tuple(map(str, getattr(self, name))) for name in
                ('kappa', 'kappa_prime', 'command', 'additive_error_abs')}))

    def primitives(self):
        """Enclose Q=kappa Zp-kappa' p-H, R=-2Zp-kappa p, d_t,d_c,u,error."""
        zero = (Fraction(0), Fraction(0))
        zp = (_negative(self.position[1]), self.position[0], zero)
        q = tuple(_add(_multiply(self.kappa, zp[i]), _negative(_add(
            _multiply(self.kappa_prime, self.position[i]), self.centered_gravity[i])))
            for i in range(3))
        r = tuple(_negative(_add(_scale(zp[i], 2),
            _multiply(self.kappa, self.position[i]))) for i in range(3))
        return (q+r+self.target_drag+self.chaser_drag
            +tuple((value, value) for value in self.command)
            +tuple((value, value) for value in self.additive_error_abs))


@dataclass(frozen=True)
class IntegralPositionEvent:
    """Exact prefix snapshot with a bounded position at its actual event time."""
    stream_id: str
    context_id: str
    q: Fraction
    count: int
    lower_zero: tuple
    upper_zero: tuple
    lower_first: tuple
    upper_first: tuple
    position: tuple
    source_id: str
    chain_id: str

    def __post_init__(self):
        for name in ('stream_id', 'context_id', 'source_id', 'chain_id'):
            _identifier(getattr(self, name))
        object.__setattr__(self, 'q', exact(self.q))
        object.__setattr__(self, 'position', _vector(self.position))
        for name in ('lower_zero', 'upper_zero', 'lower_first', 'upper_first'):
            values = tuple(map(exact, getattr(self, name)))
            if len(values) != 18:
                raise ValueError('A complete exact integral-prefix snapshot is required')
            object.__setattr__(self, name, values)
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 0:
            raise ValueError('A nonnegative integral segment count is required')


@dataclass(frozen=True)
class IntervalAffineRow:
    """Necessary observation in sum(interval_coefficient_j * fixed_parameter_j)."""
    parameter_names: tuple
    coefficients: tuple
    observed: tuple
    source_ids: tuple
    context_id: str

    def __post_init__(self):
        names = tuple(_identifier(name) for name in self.parameter_names)
        object.__setattr__(self, 'parameter_names', names)
        object.__setattr__(self, 'coefficients', _vector(self.coefficients, len(names)))
        object.__setattr__(self, 'observed', bounds(self.observed))
        sources = tuple(_identifier(name) for name in self.source_ids)
        object.__setattr__(self, 'source_ids', sources)
        if not names or len(set(names)) != len(names) or not sources:
            raise ValueError('Distinct physical/nuisance parameters and provenance are required')
        _identifier(self.context_id)

    @property
    def fingerprint(self):
        return fingerprint(dict(parameter_names=self.parameter_names,
            coefficients=_serial_intervals(self.coefficients),
            observed=tuple(map(str, self.observed)), source_ids=self.source_ids,
            context_id=self.context_id))


class IntegralMomentLedger:
    """Stream 72 exact scalar moments; retain no segment or observation history.

    Callers retain three snapshots for each wanted arc. Separate exact lower
    and upper *bound-function* moments preserve shared prefix cancellation.
    Their first moments need not themselves be ordered when q is negative.
    The recovered tent weights are nonnegative over each complete subarc.
    Fraction bit complexity depends on the supplied enclosures and clock;
    bounded record count does not imply a hardware execution deadline.
    """
    width = 18

    def __init__(self, start, *, context_id):
        self.q, self.context_id = exact(start), _identifier(context_id)
        self.stream_id, self.count = uuid4().hex, 0
        for name in ('lower_zero', 'upper_zero', 'lower_first', 'upper_first'):
            setattr(self, name, [Fraction(0)]*self.width)
        self.chain_id = fingerprint(dict(context_id=self.context_id,
            start=str(self.q), method='exact-tent-position-moments-v1'))

    def append(self, segment):
        if (not isinstance(segment, IntegralCoefficientSegment)
                or segment.start != self.q or segment.context_id != self.context_id):
            raise ValueError('A continuous hold in the same physical context is required')
        zero = segment.end-segment.start
        first = (segment.end**2-segment.start**2)/2
        for i, (lo, hi) in enumerate(segment.primitives()):
            self.lower_zero[i] += zero*lo
            self.upper_zero[i] += zero*hi
            self.lower_first[i] += first*lo
            self.upper_first[i] += first*hi
        self.chain_id = fingerprint(dict(previous=self.chain_id, segment=segment.fingerprint))
        self.q, self.count = segment.end, self.count+1

    def event(self, position, *, source_id):
        return IntegralPositionEvent(self.stream_id, self.context_id, self.q, self.count,
            tuple(self.lower_zero), tuple(self.upper_zero), tuple(self.lower_first),
            tuple(self.upper_first), position, source_id, self.chain_id)

    @staticmethod
    def _weighted(left, right, *, slope, offset):
        if min(slope*left.q+offset, slope*right.q+offset) < 0:
            raise ValueError('The affine kernel must be nonnegative on the complete subarc')
        values = []
        for i in range(IntegralMomentLedger.width):
            lo = (slope*(right.lower_first[i]-left.lower_first[i])
                +offset*(right.lower_zero[i]-left.lower_zero[i]))
            hi = (slope*(right.upper_first[i]-left.upper_first[i])
                +offset*(right.upper_zero[i]-left.upper_zero[i]))
            if lo > hi:
                raise ValueError('Inconsistent scalar bound-function prefix moments')
            values.append((lo, hi))
        return tuple(values)

    @staticmethod
    def rows(left, middle, right, *, agent_id):
        """Use any three ordered snapshots from one fixed physical context.

        The tent has height one at the middle event. Endpoint position weights
        are (1/L, -1/L-1/R, 1/R); its mass is (L+R)/2. No velocity measurement
        or independence between the three position errors is needed.
        """
        events = (left, middle, right)
        _identifier(agent_id)
        if (any(not isinstance(item, IntegralPositionEvent) for item in events)
                or len({item.stream_id for item in events}) != 1
                or len({item.context_id for item in events}) != 1
                or not left.q < middle.q < right.q
                or not left.count < middle.count < right.count):
            raise ValueError('Ordered position events from one integral stream are required')
        length_left, length_right = middle.q-left.q, right.q-middle.q
        a = IntegralMomentLedger._weighted(left, middle,
            slope=1/length_left, offset=-left.q/length_left)
        b = IntegralMomentLedger._weighted(middle, right,
            slope=-1/length_right, offset=right.q/length_right)
        weighted = tuple(_add(x, y) for x, y in zip(a, b))
        positions = (1/length_left, -1/length_left-1/length_right, 1/length_right)
        names = ('sigma_target', 'sigma_chaser')+tuple(
            f'{agent_id}.M{i}{j}' for i in range(3) for j in range(3))+tuple(
            f'{agent_id}.b{i}' for i in range(3))
        sources = tuple(item.source_id for item in events)+tuple(item.chain_id for item in events)
        result = []
        for axis in range(3):
            i = 3+axis
            derivative = _add(_scale((middle.lower_zero[i]-left.lower_zero[i],
                middle.upper_zero[i]-left.upper_zero[i]), 1/length_left),
                _scale((right.lower_zero[i]-middle.lower_zero[i],
                right.upper_zero[i]-middle.upper_zero[i]), -1/length_right))
            observed = _add(weighted[axis], derivative)
            for coefficient, event in zip(positions, events):
                observed = _add(observed, _scale(event.position[axis], coefficient))
            error = weighted[15+axis][1]
            observed = _add(observed, (-error, error))
            coefficients = [weighted[6+axis], weighted[9+axis]]+[(Fraction(0), Fraction(0))]*12
            for column in range(3):
                value = weighted[12+column]
                if value[0] != value[1]:
                    raise ValueError('The delivered command history must be exact')
                coefficients[2+3*axis+column] = value
            mass = (length_left+length_right)/2
            coefficients[11+axis] = (mass, mass)
            result.append(IntervalAffineRow(names, tuple(coefficients), observed,
                sources, left.context_id))
        return tuple(result)


@dataclass(frozen=True)
class ExactSupportCertificate:
    """A dual residual upper bound; numerical optimizer status is irrelevant."""
    parameter_set_id: str
    candidate_box: tuple
    objective: tuple
    weights: tuple
    residual: tuple
    upper: Fraction


class IntegralParameterSet:
    """Retain finite necessary cuts with one shared physical/nuisance prior.

    Interval rows produce a conservative linear outer relaxation. A numerical
    backend may propose nonnegative dual weights, but every support bound is
    recomputed with exact rational arithmetic and the remaining box residual.
    A strictly negative bound for the zero objective proves a cell empty.
    Solver failure or lack of such a certificate leaves the cell untouched.
    """
    def __init__(self, parameter_names, prior, *, context_id, maximum_rows=64):
        self.parameter_names = tuple(_identifier(name) for name in parameter_names)
        self.prior = _vector(prior, len(self.parameter_names))
        self.context_id = _identifier(context_id)
        if not self.parameter_names or len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError('Distinct physical/nuisance parameter coordinates are required')
        if isinstance(maximum_rows, bool) or not isinstance(maximum_rows, int) or maximum_rows < 1:
            raise ValueError('A finite positive retained-row budget is required')
        self.maximum_rows, self._rows = maximum_rows, []

    @property
    def rows(self):
        return tuple(self._rows)

    @property
    def fingerprint(self):
        return fingerprint(dict(context_id=self.context_id,
            parameter_names=self.parameter_names, prior=_serial_intervals(self.prior),
            rows=tuple(row.fingerprint for row in self._rows)))

    def add_rows(self, rows):
        """Append an entire arc atomically, or retain all previous cuts unchanged."""
        rows = tuple(rows)
        if any(not isinstance(row, IntervalAffineRow) or row.context_id != self.context_id
                or any(name not in self.parameter_names for name in row.parameter_names)
                for row in rows):
            raise ValueError('All rows must share the original physical parameter context')
        if len(self._rows)+len(rows) > self.maximum_rows:
            return False
        self._rows.extend(rows)
        return True

    def _box(self, box):
        box = self.prior if box is None else _vector(box, len(self.prior))
        if any(lo < a or hi > b for (lo, hi), (a, b) in zip(box, self.prior)):
            raise ValueError('Candidate cells must stay within the original physical prior')
        return box

    def outer_inequalities(self, box=None):
        """Return exact (normal, upper) pairs containing every compatible theta."""
        box = self._box(box)
        result = []
        for row in self._rows:
            coefficients = [(Fraction(0), Fraction(0))]*len(self.parameter_names)
            for name, value in zip(row.parameter_names, row.coefficients):
                coefficients[self.parameter_names.index(name)] = value
            center = tuple((lo+hi)/2 for lo, hi in coefficients)
            remainder = sum((hi-lo)/2*max(abs(a), abs(b))
                for (lo, hi), (a, b) in zip(coefficients, box))
            result.append((center, row.observed[1]+remainder))
            result.append((tuple(-value for value in center), -row.observed[0]+remainder))
        return tuple(result)

    def support_certificate(self, objective, weights, *, box=None):
        """For A theta<=b, return lambda*b+support_B(f-lambda*A)."""
        box = self._box(box)
        objective, weights = tuple(map(exact, objective)), tuple(map(exact, weights))
        rows = self.outer_inequalities(box)
        if (len(objective) != len(self.parameter_names) or len(weights) != len(rows)
                or any(value < 0 for value in weights)):
            raise ValueError('One nonnegative exact weight per retained inequality is required')
        residual = tuple(value-sum(weight*row[0][j] for weight, row in zip(weights, rows))
            for j, value in enumerate(objective))
        upper = sum(weight*row[1] for weight, row in zip(weights, rows))+sum(
            max(value*a, value*b) for value, (a, b) in zip(residual, box))
        return ExactSupportCertificate(self.fingerprint, box, objective, weights, residual, upper)

    def verify_support_certificate(self, certificate):
        if not isinstance(certificate, ExactSupportCertificate):
            raise TypeError('An exact support certificate is required')
        rebuilt = self.support_certificate(certificate.objective, certificate.weights,
            box=certificate.candidate_box)
        if rebuilt != certificate:
            raise ValueError('Support certificate is stale, altered, or from a different parameter set')
        return rebuilt.upper

    def proves_infeasible(self, certificate):
        upper = self.verify_support_certificate(certificate)
        return all(value == 0 for value in certificate.objective) and upper < 0
