"""Redundant adhesive-wrench graph with measured same-event initialization.

B=k*displacement+c*relative_velocity and M=kr*log(Q)+cr*relative_spin
remain part of the same physical strain graph. Carrying their derivatives
retains measured wrench information through each held interval.
"""
from __future__ import annotations

from fractions import Fraction

from orbexa.control.reachability import IntervalBox
from orbexa.simulation.contact_rotation import intersect
from orbexa.simulation.rigid_interval import add, subtract, scale, cross, matvec, transpose


def initial_wrench_boxes(model, time, combined, ar, supplied=None):
    """Derive the current full graph, then intersect carried/measured supports."""
    _, diag = model.evaluate(ar.scalar(time), ar.vector(combined))
    if supplied is not None and len(supplied) != model.fleet.chaser_count:
        raise ValueError('Adhesive wrench supports must name every contact member')
    result = []
    for k, row in enumerate(diag['rows']):
        if model.rest_quaternions[k] is None:
            result.append(IntervalBox.point((0,)*6))
            continue
        # A new stress-free bond still has nonzero damping wrench D*u,
        # Drot*spin. Only its displacement/rotation strain starts at zero.
        source = ar.box(scale(-1, row['bond_force_unscaled'])+scale(-1, row['couple_unscaled']))
        if supplied is not None and supplied[k] is not None:
            if supplied[k].dimension != 6:
                raise ValueError('An adhesive wrench needs three force and three couple coordinates')
            source = intersect(source, supplied[k])
        result.append(source)
    joined = IntervalBox((*combined.lower, *(value for box in result for value in box.lower)),
                         (*combined.upper, *(value for box in result for value in box.upper)))
    return joined, tuple(result)


def wrench_derivatives(model, diag, derivative):
    """Exact graph derivatives, enclosed on the positive principal log branch."""
    ar, fleet, options = model.ar, model.fleet, model.o
    body, tips, rotations = diag['body'], diag['tips'], diag['rotations']
    wt, alpha_t = body[0][10:13], tuple(derivative[16:19])
    result = []
    for k, row in enumerate(diag['rows']):
        if model.rest_quaternions[k] is None:
            result.extend(ar.scalar(0) for _ in range(6))
            continue
        j, start = k+1, 6+14*(k+1)
        relative = tuple(tuple(sum(rotations[0][m][i]*rotations[j][m][n] for m in range(3))
            for n in range(3)) for i in range(3))
        wj, alpha_j = body[j][10:13], tuple(derivative[start+10:start+13])
        lever = add(model.mounts[k], tips[k][:3])
        rotational = add(cross(alpha_j, lever), add(cross(wj, cross(wj, lever)),
            add(scale(ar.scalar(2), cross(wj, tips[k][3:])), model.extension[k])))
        # Local translations use an accelerating target-COM origin, so this
        # is already aj-at in ECI axes; no independent gravity subtraction.
        acceleration = tuple(derivative[start+3:start+6])
        rho = add(row['e'], model.anchors[k])
        u_dot = subtract(add(matvec(transpose(rotations[0]), acceleration), matvec(relative, rotational)),
            add(scale(ar.scalar(2), cross(wt, row['u'])),
                add(cross(alpha_t, rho), cross(wt, cross(wt, rho)))))
        spin_dot = subtract(subtract(matvec(relative, alpha_j), alpha_t), cross(wt, row['spin']))
        theta, spin = row['strain_angle'], row['spin']
        # J_left(theta)^-1 = I - [theta]/2 + a*[theta]^2.
        # For x=|theta|/2 in [0,pi/2), tan(x)>=x and
        # x*cot(x)>=cos(x)>=1-x^2/2 give 0<=a<=1/8.
        # This includes the identity limit without division by |theta|.
        theta_dot = add(subtract(spin, scale(ar.scalar(Fraction(1, 2)), cross(theta, spin))),
            scale(ar.interval((0, Fraction(1, 8))), cross(theta, cross(theta, spin))))
        result.extend(add(scale(ar.scalar(options.bond_stiffness_n_m), row['u']),
            scale(ar.scalar(options.bond_damping_n_s_m), u_dot)))
        result.extend(add(scale(ar.scalar(options.bond_rotational_stiffness_nm_rad), theta_dot),
            scale(ar.scalar(options.bond_rotational_damping_nm_s_rad), spin_dot)))
    return tuple(result)
