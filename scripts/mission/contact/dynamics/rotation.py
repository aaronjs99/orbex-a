"""rotation.py: Measured SO(3) supports retained alongside contact quaternion coordinates.

The matrix entries are redundant interval states with Rdot=R[omega]. They
preserve the exact physical orientation and its sensor support without
repeatedly losing that support through independent quaternion components.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from fractions import Fraction

from orbexa.common.intervals import IntervalBox
from orbexa.proofs.numerics.rigid_interval import (
    add,
    matvec,
    normalized_quaternion,
    quaternion_rotation,
    subtract,
    transpose,
)


def intersect(left, right):
    """Return the componentwise intersection of two interval boxes.

    The result is not checked for emptiness; a coordinate with lower above
    upper is returned as computed."""
    return IntervalBox(
        tuple(max(a, b) for a, b in zip(left.lower, right.lower)),
        tuple(min(a, b) for a, b in zip(left.upper, right.upper)),
    )


def initial_rotation_boxes(fleet, physical, ar, *, observation=None, support=None):
    """Return one nine-entry rotation matrix box per body at the initial state.

    Each box encloses the quaternion rotation of `physical`. If `observation`
    is given, the measured rotation widened by the target or chaser pose error
    (rad) from `support` is intersected with it."""
    states = (
        None
        if observation is None
        else (observation["target"],)
        + tuple(observation["chasers"][member.id] for member in fleet.members)
    )
    result = []
    for k in range(fleet.body_count):
        start = 6 + 14 * k + 6
        raw = quaternion_rotation(ar.vector(physical.slice(start, start + 4)))
        source = ar.box(tuple(value for row in raw for value in row))
        if states is not None:
            measured = quaternion_rotation(
                normalized_quaternion(
                    tuple(map(ar.scalar, states[k].quaternion_body_to_eci)), ar
                )
            )
            center = ar.box(tuple(value for row in measured for value in row))
            error = Fraction(
                support.target_pose_error_rad
                if k == 0
                else support.chaser_pose_error_rad
            )
            measured = IntervalBox(
                tuple(lo - error for lo in center.lower),
                tuple(hi + error for hi in center.upper),
            )
            source = intersect(source, measured)
        result.append(source)
    return tuple(result)


def rotation_derivative(rotation, spin):
    """Return the flattened row-major product R[spin] for rotation R.

    Inputs: `rotation` is a 3x3 nested sequence and `spin` a body-axis
    angular velocity (rad/s). Outputs: nine entries of the time derivative
    of R."""
    x, y, z = spin
    skew = ((0, -z, y), (z, 0, -x), (-y, x, 0))
    return tuple(
        sum(rotation[i][k] * skew[k][j] for k in range(3))
        for i in range(3)
        for j in range(3)
    )


def auxiliary_initial(fleet, physical, ar, *, rotations=None, gaps=None):
    """Append rotation and gap boxes to the physical initial box.

    Inputs: `rotations` and `gaps`, if given, are carried supports that are
    intersected with the rotation boxes and per-chaser gap boxes (m) derived
    from `physical`.
    Outputs: `(combined, rotation_boxes, gap_boxes)`, where `combined` is the
    physical box extended by the rotation entries and then the gaps.
    Raises ValueError if `rotations` does not give nine entries for every body."""
    source_rotations = initial_rotation_boxes(fleet, physical, ar)
    if rotations is not None:
        if len(rotations) != fleet.body_count or any(
            box.dimension != 9 for box in rotations
        ):
            raise ValueError(
                "Contact rotation supports must name every body with nine matrix entries"
            )
        source_rotations = tuple(
            intersect(a, b) for a, b in zip(source_rotations, rotations)
        )
    matrices = tuple(
        tuple(tuple(ar.vector(box)[3 * i : 3 * i + 3]) for i in range(3))
        for box in source_rotations
    )
    values, source_gaps = ar.vector(physical), []
    for k, member in enumerate(fleet.members):
        j = k + 1
        body = values[6 + 14 * j : 6 + 14 * (j + 1)]
        target = values[6:20]
        tip = values[fleet.tip_slice.start + 6 * k : fleet.tip_slice.start + 6 * k + 3]
        offset = matvec(matrices[j], add(tuple(map(ar.scalar, fleet.mounts[k])), tip))
        relative = matvec(
            transpose(matrices[0]), subtract(add(body[:3], offset), target[:3])
        )
        gap = ar.box(
            (
                sum(
                    ar.scalar(normal) * (value - ar.scalar(anchor))
                    for normal, value, anchor in zip(
                        member.normal_body, relative, member.surface_anchor_body_m
                    )
                ),
            )
        )
        if gaps is not None and gaps[k] is not None:
            gap = intersect(gap, gaps[k])
        source_gaps.append(gap)
    auxiliary = (*source_rotations, *source_gaps)
    combined = IntervalBox(
        (*physical.lower, *(value for box in auxiliary for value in box.lower)),
        (*physical.upper, *(value for box in auxiliary for value in box.upper)),
    )
    return combined, source_rotations, tuple(source_gaps)
