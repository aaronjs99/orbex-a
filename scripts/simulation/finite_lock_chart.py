"""Exact positioner motion in a relative-to-initial attitude chart.

The immutable initial rotations and positioner pose/twist retain their complete
input uncertainty. Unit initial increments describe a coordinate identity, not
a reset or restriction of the initial physical attitudes.
"""
from orbexa.simulation.rigid_interval import add, subtract, scale, cross, matvec, transpose, quaternion_rotation


def _minus_identity(matrix, ar):
    return tuple(tuple(value-ar.scalar(int(i == j)) for j, value in enumerate(row))
                 for i, row in enumerate(matrix))


def positioner_motion(model, time, values, diag, index):
    """Return the same finite profile's extension, speed and acceleration."""
    ar, fleet = model.ar, model.fleet
    j = index+1
    initial_rotation = model.initial_relative_rotations[j]
    inverse_initial = transpose(initial_rotation)
    increment_target = quaternion_rotation(values[13:17])
    increment_body = quaternion_rotation(values[13+4*j:17+4*j])
    inverse_increment = transpose(increment_body)
    initial_extension, initial_speed = model.initial_positioners[index]
    lever = matvec(initial_rotation, add(tuple(map(ar.scalar, fleet.mounts[index])), initial_extension))
    initial_speed_reference = matvec(initial_rotation, initial_speed)
    target_difference = _minus_identity(increment_target, ar)
    body_difference = _minus_identity(inverse_increment, ar)
    anchor = model.tip_anchors[index]
    anchor_speed = model.tip_anchor_velocity[index]
    sigma_target, sigma_body = model.relative_spin[0], model.relative_spin[j]
    duration = ar.scalar(model.o.lock_duration_s)
    f = time-time*time/(2*duration)
    rotated_anchor_speed = matvec(increment_target, anchor_speed)
    target_anchor_increment = add(matvec(target_difference, anchor), scale(f, rotated_anchor_speed))
    displacement_increment = add(scale(f, subtract(model.relative_velocity[0], model.relative_velocity[j])),
                                 target_anchor_increment)
    # This is the INCREMENT of d_f. The exact initial identity is
    # d_f(0)=A_j*xdot0 + sigma_j x (A_j*ell0).
    derivative_increment = add(cross(sigma_target, target_anchor_increment),
                               matvec(target_difference, anchor_speed))
    extension = add(initial_extension, add(matvec(inverse_initial, matvec(body_difference, lever)),
        matvec(inverse_initial, matvec(inverse_increment, displacement_increment))))
    extension_f = add(initial_speed, add(matvec(inverse_initial, matvec(body_difference, initial_speed_reference)),
        matvec(inverse_initial, matvec(inverse_increment, subtract(derivative_increment,
            cross(sigma_body, displacement_increment))))))
    anchor_second = add(cross(sigma_target, cross(sigma_target,
        add(matvec(increment_target, anchor), scale(f, rotated_anchor_speed)))),
        scale(ar.scalar(2), cross(sigma_target, rotated_anchor_speed)))
    second_reference = add(subtract(subtract(anchor_second,
        scale(ar.scalar(2), cross(sigma_body, initial_speed_reference))),
        cross(sigma_body, cross(sigma_body, lever))),
        add(scale(ar.scalar(-2), cross(sigma_body, derivative_increment)),
            cross(sigma_body, cross(sigma_body, displacement_increment))))
    extension_ff = matvec(inverse_initial, matvec(inverse_increment, second_reference))
    return extension, scale(diag['g'], extension_f), add(scale(diag['gd'], extension_f),
        scale(diag['g']*diag['g'], extension_ff))
