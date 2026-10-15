"""Explicit bus-patch matching; finite-body route policies live separately."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def assign_composite_docking_targets(target, initial_states):
    """Match initial directions to declared accessible patches without fixtures.

    This is geometric point assignment only. The composite deployment policy
    additionally screens finite-bus paths, tube reserves and fleet separation.
    """
    states = np.asarray(initial_states, dtype=float)
    if states.size == 0:
        return []
    if states.ndim != 2 or states.shape[1] != 6 or not np.all(np.isfinite(states)):
        raise ValueError("Composite assignment requires finite six-state initial conditions")
    patches = tuple(target.geometry.attachment_patches)
    if len(patches) < len(states):
        raise ValueError("The spacecraft has too few explicitly declared attachment patches")
    rotation = np.asarray(target.rotation_matrix_at(0.), dtype=float)
    starts = states[:, :3] @ rotation
    goals = np.asarray([np.asarray(patch.point_body_m) + target.docking_standoff * np.asarray(patch.normal_body) for patch in patches])
    costs = np.linalg.norm(starts[:, None, :] - goals[None, :, :], axis=-1)
    for index, goal in enumerate(goals):
        if target.geometry.signed_distance(goal) <= 0:
            costs[:, index] = np.inf
    try:
        rows, cols = linear_sum_assignment(costs)
    except ValueError as error:
        raise ValueError("No complete geometric patch matching was found; no physical unreachability is inferred") from error
    result = []
    for row, col in sorted(zip(rows, cols)):
        patch = patches[col]
        result.append(dict(body_position=goals[col].copy(), body_normal=np.asarray(patch.normal_body).copy(),
            azimuth=float(np.arctan2(patch.normal_body[1], patch.normal_body[0])), surface="attachment_patch",
            standoff=float(target.docking_standoff), candidate_index=int(col), attachment_patch_id=patch.id,
            attachment_component_id=patch.component_id, assignment_scope="geometric_point_matching_only"))
    return result
