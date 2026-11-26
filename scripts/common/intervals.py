"""intervals.py: exact rational interval boxes and Arb ball arithmetic shared by every layer.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from pathlib import Path
from threading import RLock
from typing import Any

_ARB_LOCK = RLock()
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
    """Return ``(lower, upper)`` as exact Fractions from two bounds.

    Inputs: ``values`` holds the two bounds and ``name`` labels error messages.
    Raises ValueError unless there are exactly two bounds with lower <= upper.
    """
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
    if (
        Fraction.from_float(rounded) < value
        if upward
        else Fraction.from_float(rounded) > value
    ):
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
        """Store the endpoints as exact Fractions.

        Raises ValueError unless the endpoints are nonempty, equally sized and
        ordered componentwise.
        """
        lo, hi = tuple(map(_exact, self.lower)), tuple(map(_exact, self.upper))
        if not lo or len(lo) != len(hi) or any(a > b for a, b in zip(lo, hi)):
            raise ValueError("A box requires equally sized nonempty ordered endpoints")
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)

    @classmethod
    def point(cls, values: Iterable[Any]) -> IntervalBox:
        """Return the degenerate box whose lower and upper endpoints are ``values``."""
        values = tuple(values)
        return cls(values, values)

    @classmethod
    def from_center_radius(cls, center, radius) -> IntervalBox:
        """Return the box from ``center - radius`` to ``center + radius``, exactly.

        Raises ValueError when the dimensions differ or any radius is negative.
        """
        center, radius = tuple(map(_exact, center)), tuple(map(_exact, radius))
        if len(center) != len(radius) or any(r < 0 for r in radius):
            raise ValueError("Center and nonnegative radius dimensions must agree")
        return cls(
            tuple(c - r for c, r in zip(center, radius)),
            tuple(c + r for c, r in zip(center, radius)),
        )

    @property
    def dimension(self) -> int:
        """Number of components."""
        return len(self.lower)

    def contains(self, other: IntervalBox) -> bool:
        """Return True when ``other`` has this dimension and lies inside this box."""
        return self.dimension == other.dimension and all(
            a <= c and d <= b
            for a, b, c, d in zip(self.lower, self.upper, other.lower, other.upper)
        )

    def hull(self, other: IntervalBox) -> IntervalBox:
        """Return the smallest box containing this box and ``other``.

        Raises ValueError when the dimensions differ.
        """
        if self.dimension != other.dimension:
            raise ValueError("Cannot hull different box dimensions")
        return IntervalBox(
            tuple(map(min, self.lower, other.lower)),
            tuple(map(max, self.upper, other.upper)),
        )

    def slice(self, start: int, end: int) -> IntervalBox:
        """Return the sub-box of components ``start`` up to ``end``."""
        return IntervalBox(self.lower[start:end], self.upper[start:end])

    def float_bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return float ``(lower, upper)`` tuples, rounded outward."""
        return (
            tuple(_float_outward(v, False) for v in self.lower),
            tuple(_float_outward(v, True) for v in self.upper),
        )

    def as_dict(self) -> dict:
        """Export the box as a new dict.

        Outputs: keys ``lower`` and ``upper`` (outward float bounds) and
        ``exact_lower`` and ``exact_upper`` (rational strings). The derived
        values are cached on the box and recomputed whenever the exact endpoints
        no longer match the cache key.
        """
        # Repeated certificate fingerprints export the same immutable boxes.
        # Cache only derived immutable tuples, with copied exact primitive
        # endpoints as the guard. Even a caller bypassing the frozen dataclass
        # or mutating a Fraction cannot hide changed bounds behind this cache.
        key = (
            tuple((v.numerator, v.denominator) for v in self.lower),
            tuple((v.numerator, v.denominator) for v in self.upper),
        )
        cached = getattr(self, "_export_cache", None)
        if cached is None or cached[0] != key:
            lo, hi = self.float_bounds()
            values = (lo, hi, tuple(map(str, self.lower)), tuple(map(str, self.upper)))
            cached = (key, values)
            object.__setattr__(self, "_export_cache", cached)
        lo, hi, exact_lo, exact_hi = cached[1]
        # Callers own their dictionary; changing it never changes a later export.
        return {
            "lower": lo,
            "upper": hi,
            "exact_lower": exact_lo,
            "exact_upper": exact_hi,
        }


class _ProofUnresolved(Exception):
    """Raised when Arb cannot return a finite enclosure, so a proof stays unresolved."""


class _ArbArithmetic:
    """Conversions between exact rationals, Arb balls and interval boxes."""

    def __init__(self, flint):
        """Keep the ``arb`` ball type of the supplied python-flint module."""
        self.arb = flint.arb

    def scalar(self, value):
        """Return the Arb ball for an exact value as numerator over denominator."""
        value = _exact(value)
        return self.arb(value.numerator) / self.arb(value.denominator)

    def interval(self, bounds):
        """Return an Arb ball covering ``(lo, hi)``, one ball when they are equal."""
        lo, hi = bounds
        return self.scalar(lo).union(self.scalar(hi)) if lo != hi else self.scalar(lo)

    def vector(self, box):
        """Return a tuple of Arb balls, one per component of ``box``."""
        return tuple(self.interval(bounds) for bounds in zip(box.lower, box.upper))

    @staticmethod
    def exact_point(ball):
        """Return the exact rational ``mantissa * 2**exponent`` of an Arb ball."""
        mantissa, exponent = ball.man_exp()
        exponent = int(exponent)
        return Fraction(int(mantissa)) * (Fraction(2) ** exponent)

    def box(self, balls):
        """Return the exact IntervalBox whose endpoints are the ball bounds.

        Raises _ProofUnresolved when any ball is nonfinite.
        """
        if not all(ball.is_finite() for ball in balls):
            raise _ProofUnresolved("Arb returned a nonfinite enclosure")
        return IntervalBox(
            tuple(self.exact_point(v.lower()) for v in balls),
            tuple(self.exact_point(v.upper()) for v in balls),
        )

    def inflate(self, balls, config):
        """Widen each ball radius by ``radius * inflation_fraction + padding``.

        Inputs: ``config`` supplies ``inflation_fraction`` and ``absolute_padding``.
        The added radius is rounded up.
        Outputs: a tuple of balls.
        """
        fraction, padding = (
            self.scalar(config.inflation_fraction),
            self.scalar(config.absolute_padding),
        )
        return tuple(
            v + self.arb(0, (v.rad() * fraction + padding).upper()) for v in balls
        )


class _Arithmetic(_ArbArithmetic):
    """Arb arithmetic with the elementary functions used by the flow proofs."""

    sqrt = staticmethod(lambda value: value.sqrt())
    exp = staticmethod(lambda value: value.exp())
    expm1 = staticmethod(lambda value: value.expm1())
    sin = staticmethod(lambda value: value.sin())
    cos = staticmethod(lambda value: value.cos())

    def square(self, value):
        """Return the interval square of a ball, with low end 0 if it contains 0."""
        # flint 0.8's generic power uses log for negative balls, even **2.
        box = self.box((value,))
        lo, hi = box.lower[0], box.upper[0]
        lower = 0 if lo <= 0 <= hi else min(lo * lo, hi * hi)
        return self.interval((lower, max(lo * lo, hi * hi)))
