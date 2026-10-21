"""Exact occupied-set supports when a body's attitude is any proper rotation."""
from __future__ import annotations

from fractions import Fraction
from itertools import product

from orbexa.control.certification.attitude import _sqrt_upper
from orbexa.control.certification.reachability import IntervalBox, _exact


_DIRECTIONS = tuple(tuple(map(Fraction, row)) for row in product((-1, 0, 1), repeat=3)
    if any(row) and next(value for value in row if value) == 1)


def all_rotation_component_bounds(vertices, center):
    """Enclose every rotation of the complete expanded vertex hull.

    Orthogonality gives |n^T R v| <= ||n||_2 ||v||_2. The squared
    radii and directions are exact; square roots round outward. Convexity
    extends each vertex bound to its entire solid component. Thirteen fixed
    axes and diagonals avoid assuming an observed attitude or a sampled pose.
    """
    points = tuple(tuple(map(_exact, vertex)) for vertex in vertices)
    if not points or any(len(point) != 3 for point in points) or center.dimension != 3:
        raise ValueError('All-rotation supports need complete 3D vertices and a 3D center set')
    radius_squared = max(sum(value*value for value in point) for point in points)
    radius = _sqrt_upper(radius_squared)
    occupied = IntervalBox(tuple(value-radius for value in center.lower),
        tuple(value+radius for value in center.upper))
    projections = []
    for direction in _DIRECTIONS:
        support = _sqrt_upper(radius_squared*sum(value*value for value in direction))
        products = tuple((d*lo, d*hi) for d, lo, hi in zip(direction, center.lower, center.upper))
        lower = sum(min(a, b) for a, b in products)-support
        upper = sum(max(a, b) for a, b in products)+support
        projections.append((direction, IntervalBox((lower,), (upper,))))
    return occupied, tuple(projections)
