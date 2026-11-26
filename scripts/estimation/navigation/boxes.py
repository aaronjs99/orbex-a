"""boxes.py: Exact navigation box intersection and numerical representatives.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from orbexa.common.intervals import IntervalBox


def intersect(left, right):
    """Return the componentwise intersection of two boxes, or None if it is empty.

    Inputs: left, right: IntervalBox values of equal dimension.
    Outputs: IntervalBox, or None when any lower bound exceeds its upper bound.
    Raises ValueError when the dimensions differ.
    """
    if left.dimension != right.dimension:
        raise ValueError("Cannot intersect unlike state dimensions")
    lo = tuple(max(a, b) for a, b in zip(left.lower, right.lower))
    hi = tuple(min(a, b) for a, b in zip(left.upper, right.upper))
    return None if any(a > b for a, b in zip(lo, hi)) else IntervalBox(lo, hi)


def midpoint(box):
    """Return the float midpoint (lower + upper) / 2 of each box component as a tuple.

    The result is a numerical representative, not an enclosure.
    """
    return tuple(float((a + b) / 2) for a, b in zip(box.lower, box.upper))


def _product(a, b):
    """Return (min, max) of the four endpoint products of two (low, high) pairs."""
    values = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return min(values), max(values)
