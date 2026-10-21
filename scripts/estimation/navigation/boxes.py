"""Exact navigation box intersection and numerical representatives."""
from __future__ import annotations

from orbexa.control.certification.reachability import IntervalBox


def intersect(left, right):
    if left.dimension != right.dimension:
        raise ValueError('Cannot intersect unlike state dimensions')
    lo = tuple(max(a, b) for a, b in zip(left.lower, right.lower))
    hi = tuple(min(a, b) for a, b in zip(left.upper, right.upper))
    return None if any(a > b for a, b in zip(lo, hi)) else IntervalBox(lo, hi)

def midpoint(box):
    return tuple(float((a+b)/2) for a, b in zip(box.lower, box.upper))

def _product(a, b):
    values = (a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1])
    return min(values), max(values)
