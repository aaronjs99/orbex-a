"""Stress-free digital bond formation with carried uncertain rest geometry."""
from __future__ import annotations

from fractions import Fraction

from orbexa.control.reachability import IntervalBox
from orbexa.simulation.contact_rotation import intersect
from orbexa.simulation.rigid_interval import (
    quaternion_product, quaternion_conjugate, subtract, square,
)


def initial_strain_boxes(model, time, combined, ar, supplied=None):
    _, diag = model.evaluate(ar.scalar(time), ar.vector(combined))
    result = []
    for k, row in enumerate(diag['rows']):
        if model.rest_quaternions[k] is None:
            result.append(IntervalBox.point((0, 0, 0, 0, 0, 0, 1)))
            continue
        bodies = diag['body']
        relative = quaternion_product(quaternion_conjugate(bodies[0][6:10]), bodies[k+1][6:10])
        strain = quaternion_product(relative, quaternion_conjugate(model.rest_quaternions[k]))
        if ar.box((strain[3],)).upper[0] < 0:
            strain = tuple(-value for value in strain)
        source = ar.box(subtract(row['e'], model.rest_positions[k])+strain)
        if supplied is not None and supplied[k] is not None:
            if supplied[k].dimension != 7:
                raise ValueError('A carried bond strain needs three displacement and four quaternion coordinates')
            source = intersect(source, supplied[k])
        result.append(source)
    joined = IntervalBox((*combined.lower, *(value for box in result for value in box.lower)),
                         (*combined.upper, *(value for box in result for value in box.upper)))
    return joined, tuple(result)


def potential_box(model, diag):
    ar, options = model.ar, model.o
    value = ar.scalar(0)
    for row in diag['rows']:
        value += ar.scalar(Fraction(options.compression_stiffness_n_m)/2)*square(row['compression'], ar)
        value += row['activation']/2*(ar.scalar(options.bond_stiffness_n_m)*
            sum(square(x, ar) for x in row['deformation'])+ar.scalar(options.bond_rotational_stiffness_nm_rad)*
            sum(square(x, ar) for x in row['strain_angle']))
    box = ar.box((value,))
    return IntervalBox((max(Fraction(0), box.lower[0]),), (max(Fraction(0), box.upper[0]),))
