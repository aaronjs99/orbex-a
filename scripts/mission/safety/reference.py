"""reference.py: Exact C2 polynomial routes and Bernstein reference enclosures.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial, gcd, perm
from typing import Any

from orbexa.common.intervals import IntervalBox, _exact
from orbexa.control.fallback.polynomial import (
    PolynomialReferenceSource,
    PolynomialReferenceSpan,
)
from orbexa.control.fallback.refuge import (
    ReferenceBounds,
    ReferenceJet,
    _absbox,
    _identity,
    _scope,
    _vec,
    exact_record,
    fingerprint,
)


def _derivative(coefficients, order):
    """Return the coefficients of the order-th derivative of a power-basis polynomial.

    Order zero returns the input. A polynomial differentiated past its degree yields
    (Fraction(0),).
    """
    if type(order) is int and order == 0:
        return coefficients
    if type(order) is int and order > 0:
        return tuple(
            value * perm(i, order) for i, value in enumerate(coefficients) if i >= order
        ) or (Fraction(0),)
    return tuple(
        value * Fraction(factorial(i), factorial(i - order))
        for i, value in enumerate(coefficients)
        if i >= order
    ) or (Fraction(0),)


def _polynomial(coefficients, value):
    """Evaluate ascending-power coefficients at value by Horner's rule.

    A single Fraction constant is returned as a new Fraction.
    """
    iterator = reversed(coefficients)
    total = next(iterator, Fraction(0))
    if len(coefficients) == 1 and type(total) is Fraction:
        # Keep a returned constant independent of the source Fraction object.
        return Fraction(total.numerator, total.denominator)
    for coefficient in iterator:
        total = total * value + coefficient
    return total


def _fraction_scalars(value):
    """Return (numerator, denominator) of a normalized exact Fraction.

    Raises ValueError for a non-Fraction or a rational that is not normalized.
    """
    if type(value) is not Fraction:
        raise ValueError("Polynomial source scalars must remain exact Fractions")
    numerator, denominator = value.numerator, value.denominator
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or denominator <= 0
        or gcd(numerator, denominator) != 1
    ):
        raise ValueError("Polynomial source rationals must remain normalized")
    return numerator, denominator


def _coefficient_scalars(rows):
    """Return (numerator, denominator) pairs for three tuple rows of 1 to 6 Fractions.

    Raises ValueError for any other structure.
    """
    if (
        type(rows) is not tuple
        or len(rows) != 3
        or any(type(row) is not tuple or not 1 <= len(row) <= 6 for row in rows)
    ):
        raise ValueError("Three immutable finite polynomial rows are required")
    return tuple(tuple(_fraction_scalars(value) for value in row) for row in rows)


def _bernstein(coefficients, lower, upper):
    """Exact range enclosure on [lower,upper] by Bernstein convex hull."""
    degree = len(coefficients) - 1
    if degree == 0:
        value = coefficients[0]
        if type(value) is Fraction:
            value = Fraction(value.numerator, value.denominator)
        return value, value
    power = tuple(
        sum(
            coefficients[i] * comb(i, j) * lower ** (i - j) * (upper - lower) ** j
            for i in range(j, degree + 1)
        )
        for j in range(degree + 1)
    )
    bernstein = tuple(
        sum(power[j] * Fraction(comb(k, j), comb(degree, j)) for j in range(k + 1))
        for k in range(degree + 1)
    )
    return min(bernstein), max(bernstein)


@dataclass(frozen=True)
class PolynomialReferencePiece:
    """Three axis polynomials of the position on one finite reference-clock interval.

    Coefficients are ascending powers of (q - start), at most six per axis; start, end
    and q are unwrapped-anomaly reference-clock values.
    """

    start: Any
    end: Any
    coefficients: Any

    def __post_init__(self):
        """Convert the scope and coefficients to exact values and validate them.

        Raises ValueError unless there are three nonempty rows of at most six
        coefficients.
        """
        start, end = _scope(self.start, self.end)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        value = tuple(tuple(map(_exact, row)) for row in self.coefficients)
        if len(value) != 3 or any(not row or len(row) > 6 for row in value):
            raise ValueError("Three polynomials of degree at most five are required")
        object.__setattr__(self, "coefficients", value)

    def jet(self, q, order):
        """Return the exact order-th derivative of each axis polynomial at q.

        Raises ValueError if q lies outside [start, end].
        """
        q = _exact(q)
        if not self.start <= q <= self.end:
            raise ValueError("Polynomial query exceeds its finite piece")
        offset = q - self.start
        return tuple(
            _polynomial(_derivative(row, order), offset) for row in self.coefficients
        )

    def jets(self, q):
        """Evaluate all three exact jets with one shared rational clock offset."""
        q = _exact(q)
        if not self.start <= q <= self.end:
            raise ValueError("Polynomial query exceeds its finite piece")
        offset = q - self.start
        return tuple(
            tuple(
                _polynomial(_derivative(row, order), offset)
                for row in self.coefficients
            )
            for order in range(3)
        )

    def enclose(self, start, end, order):
        """Return an IntervalBox enclosing the order-th derivative of each axis.

        The range over [start, end] is bounded exactly by the Bernstein convex hull.
        Raises ValueError unless self.start <= start <= end <= self.end.
        """
        if not self.start <= start <= end <= self.end:
            raise ValueError("Polynomial range exceeds its finite piece")
        pairs = tuple(
            _bernstein(_derivative(row, order), start - self.start, end - self.start)
            for row in self.coefficients
        )
        return IntervalBox(tuple(lo for lo, _ in pairs), tuple(hi for _, hi in pairs))


_PIECE_JET_METHOD = PolynomialReferencePiece.jet


_PIECE_JETS_METHOD = PolynomialReferencePiece.jets


_DERIVATIVE_METHOD = _derivative


_POLYNOMIAL_METHOD = _polynomial


class C2RefugeReference:
    """Exact continuous polynomial route with a final constant holding piece."""

    def __init__(self, pieces, *, frame, clock_id, reference_id=None):
        """Store a finite route of pieces joined at exact common C2 jets.

        Inputs: pieces, PolynomialReferencePiece items; frame and clock_id identities;
            reference_id, which defaults to a fingerprint of the pieces, frame and
            clock. Raises ValueError if pieces are empty or of another type, if
            neighboring pieces do not meet with equal jets of orders 0 to 2, or if an
            identity is invalid.
        """
        self.pieces = tuple(pieces)
        if not self.pieces or any(
            not isinstance(piece, PolynomialReferencePiece) for piece in self.pieces
        ):
            raise ValueError("A finite polynomial route is required")
        for old, new in zip(self.pieces, self.pieces[1:]):
            if old.end != new.start or any(
                old.jet(old.end, order) != new.jet(new.start, order)
                for order in range(3)
            ):
                raise ValueError("Reference pieces must join at exact common C2 jets")
        self.frame, self.clock_id = (
            _identity(frame, "reference frame"),
            _identity(clock_id, "clock identity"),
        )
        self.reference_id = reference_id or fingerprint(
            {"pieces": exact_record(self.pieces), "frame": frame, "clock_id": clock_id}
        )
        _identity(self.reference_id, "reference identity")

    @classmethod
    def retreat(
        cls,
        *,
        start,
        retreat_end,
        hold_end,
        initial_state,
        refuge_position,
        initial_acceleration=(0, 0, 0),
        frame,
        clock_id,
        reference_id=None,
    ):
        """Build a quintic retreat piece followed by a constant hold at the refuge.

        The quintic starts from initial_state (position then velocity in the frame) and
        initial_acceleration at start, and reaches refuge_position at retreat_end with
        zero velocity and zero acceleration. The constant piece holds that position
        until hold_end. Raises ValueError for an invalid scope or vector size.
        """
        start, retreat_end = _scope(start, retreat_end)
        _, hold_end = _scope(retreat_end, hold_end)
        x, goal = (
            _vec(initial_state, 6, "reference initial state"),
            _vec(refuge_position, 3, "refuge position"),
        )
        acceleration = _vec(initial_acceleration, 3, "initial reference acceleration")
        duration, rows = retreat_end - start, []
        for p, v, a, g in zip(x[:3], x[3:], acceleration, goal):
            c0, c1, c2 = p, v, a / 2
            dp, dv, da = (
                g - c0 - c1 * duration - c2 * duration**2,
                -c1 - 2 * c2 * duration,
                -2 * c2,
            )
            rows.append(
                (
                    c0,
                    c1,
                    c2,
                    10 * dp / duration**3 - 4 * dv / duration**2 + da / (2 * duration),
                    -15 * dp / duration**4 + 7 * dv / duration**3 - da / duration**2,
                    6 * dp / duration**5
                    - 3 * dv / duration**4
                    + da / (2 * duration**3),
                )
            )
        return cls(
            (
                PolynomialReferencePiece(start, retreat_end, rows),
                PolynomialReferencePiece(
                    retreat_end, hold_end, tuple((g,) for g in goal)
                ),
            ),
            frame=frame,
            clock_id=clock_id,
            reference_id=reference_id,
        )

    def sample(self, q):
        """Return the exact ReferenceJet of the route at q.

        The piece containing q is used (the last piece at its end). Per-order jet calls
        replace the combined evaluation when the piece or helper functions were
        replaced. Raises ValueError if q is outside the route.
        """
        q = _exact(q)
        piece = next((p for p in self.pieces if p.start <= q < p.end), None)
        if piece is None and q == self.pieces[-1].end:
            piece = self.pieces[-1]
        if piece is None:
            raise ValueError("Reference sample exceeds its finite route")
        known = (
            type(piece) is PolynomialReferencePiece
            and getattr(piece.jet, "__func__", piece.jet) is _PIECE_JET_METHOD
            and getattr(piece.jets, "__func__", piece.jets) is _PIECE_JETS_METHOD
            and _derivative is _DERIVATIVE_METHOD
            and _polynomial is _POLYNOMIAL_METHOD
        )
        jets = (
            piece.jets(q) if known else tuple(piece.jet(q, order) for order in range(3))
        )
        return ReferenceJet(q, *jets, self.reference_id, self.frame, self.clock_id)

    def polynomial_source(self):
        """Reuse an immutable source only while every actual exact field agrees.

        The key copies field values rather than retaining Piece identities, so
        even replacing fields on a frozen piece cannot alias the old guard.
        Mutable containers never qualify for this cache.
        """
        if (
            type(self.pieces) is not tuple
            or not self.pieces
            or any(type(piece) is not PolynomialReferencePiece for piece in self.pieces)
        ):
            raise ValueError(
                "Polynomial source caching requires immutable typed reference pieces"
            )
        pieces = tuple(
            (
                _fraction_scalars(piece.start),
                _fraction_scalars(piece.end),
                _coefficient_scalars(piece.coefficients),
            )
            for piece in self.pieces
        )
        key = (self.reference_id, self.frame, self.clock_id, pieces)
        cached = getattr(self, "_polynomial_source_cache", None)
        if (
            type(cached) is tuple
            and len(cached) == 2
            and cached[0] == key
            and type(cached[1]) is PolynomialReferenceSource
        ):
            source = cached[1]
            try:
                unchanged = (
                    type(source.pieces) is tuple
                    and all(
                        type(piece) is PolynomialReferenceSpan
                        for piece in source.pieces
                    )
                    and (
                        source.reference_id,
                        source.frame,
                        source.clock_id,
                        tuple(
                            (
                                _fraction_scalars(piece.start),
                                _fraction_scalars(piece.end),
                                _coefficient_scalars(piece.coefficients),
                            )
                            for piece in source.pieces
                        ),
                    )
                    == key
                )
            except (AttributeError, TypeError, ValueError):
                unchanged = False
            if unchanged:
                return source
        # Copy every rational from native integers; the reusable source must
        # not alias a source/cache Fraction forced to change after admission.
        source = PolynomialReferenceSource(
            tuple(
                PolynomialReferenceSpan(
                    Fraction(*start),
                    Fraction(*end),
                    tuple(tuple(Fraction(*value) for value in row) for row in rows),
                )
                for start, end, rows in pieces
            ),
            self.reference_id,
            self.frame,
            self.clock_id,
        )
        self._polynomial_source_cache = (key, source)
        return source

    def bounds(self, start, end):
        """Return ReferenceBounds enclosing the route over [start, end].

        Orders 0 to 2 give position, velocity and acceleration boxes; order 3 gives the
        absolute one-sided jerk bound. Each is the hull of the Bernstein enclosures of
        the overlapping pieces. Raises ValueError if the query leaves the route.
        """
        start, end = _scope(start, end)
        if start < self.pieces[0].start or end > self.pieces[-1].end:
            raise ValueError("Reference bound query exceeds its finite route")
        ranges = []
        for order in range(4):
            value = None
            for piece in self.pieces:
                lower, upper = max(start, piece.start), min(end, piece.end)
                if lower < upper:
                    local = piece.enclose(lower, upper, order)
                    value = local if value is None else value.hull(local)
            ranges.append(value)
        return ReferenceBounds(
            start,
            end,
            *ranges[:3],
            _absbox(ranges[3]),
            self.reference_id,
            self.frame,
            self.clock_id,
            "exact-quintic-Bernstein:" + fingerprint(self.pieces),
            (
                "Exact rational polynomial coefficients and exact C2 boundary identities",
                "Bernstein convex hull encloses every position, velocity, acceleration and one-sided jerk value",
            ),
            proof_kind="analytic-global",
        )

    def windows(self, maximum_width, maximum_windows=10000):
        """Split every piece into equal windows no wider than maximum_width.

        Returns a tuple of (start, end) pairs. Raises ValueError if maximum_width is not
        positive or the window count would exceed maximum_windows.
        """
        width = _exact(maximum_width)
        if width <= 0:
            raise ValueError("Positive finite reference bound width is required")
        output = []
        for piece in self.pieces:
            ratio = (piece.end - piece.start) / width
            count = max(
                1, (ratio.numerator + ratio.denominator - 1) // ratio.denominator
            )
            if len(output) + count > maximum_windows:
                raise ValueError("Finite reference-window budget exhausted")
            step = (piece.end - piece.start) / count
            output.extend(
                (piece.start + i * step, piece.start + (i + 1) * step)
                for i in range(count)
            )
        return tuple(output)
