# /***********************************************************
# *                                                         *
# * Copyright (c) 2026                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu, Brett T. Lopez                *
# * Contact: {aaronjs, btlopez}@ucla.edu                    *
# *                                                         *
# ***********************************************************/

"""Cylindrical side-point assignment from each chaser approach direction."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from orbexa.core.target_geometry import TumblingCylinderTarget


def assign_docking_targets(
    target: TumblingCylinderTarget, initial_states: Sequence[np.ndarray]
) -> List[Dict[str, Any]]:
    count = len(initial_states)
    if count == 0:
        return []

    approach_angles = [target.approach_azimuth(state) for state in initial_states]
    approach_axials = [
        target.approach_axial_coordinate(state) for state in initial_states
    ]
    if count == 1:
        start_azimuth = approach_angles[0]
        candidate_angles = [start_azimuth]
        candidate_geometry = [target.side_docking_geometry(start_azimuth)]
        assignment = [(0, 0)]
    else:
        best_cost = np.inf
        best_angles: List[float] = []
        best_assignment: List[Tuple[int, int]] = []
        start_options = [0.0]
        start_options.extend(
            angle - 2.0 * np.pi * offset / count
            for angle in approach_angles
            for offset in range(count)
        )
        for start_azimuth in start_options:
            angles = [
                float(
                    np.mod(
                        start_azimuth + 2.0 * np.pi * idx / count,
                        2.0 * np.pi,
                    )
                )
                for idx in range(count)
            ]
            costs = np.zeros((count, count), dtype=float)
            for state_idx, approach_angle in enumerate(approach_angles):
                for point_idx, candidate_angle in enumerate(angles):
                    costs[state_idx, point_idx] = abs(
                        np.mod(
                            approach_angle - candidate_angle + np.pi,
                            2.0 * np.pi,
                        )
                        - np.pi
                    )
            rows, cols = linear_sum_assignment(costs)
            total_cost = float(costs[rows, cols].sum())
            if total_cost < best_cost - 1.0e-12:
                best_cost = total_cost
                best_angles = angles
                best_assignment = sorted(
                    zip(rows.tolist(), cols.tolist()), key=lambda item: item[0]
                )
        candidate_angles = best_angles
        candidate_geometry = [
            target.side_docking_geometry(angle) for angle in candidate_angles
        ]
        assignment = best_assignment

    assigned: List[Dict[str, Any]] = []
    for state_idx, point_idx in assignment:
        _, normal = candidate_geometry[point_idx]
        point, normal = target.side_docking_geometry(
            candidate_angles[point_idx], approach_axials[state_idx]
        )
        assigned.append(
            {
                "body_position": point,
                "body_normal": normal,
                "azimuth": candidate_angles[point_idx],
                "surface": "cylinder_side",
                "standoff": target.docking_standoff,
                "candidate_index": int(point_idx),
            }
        )
    return assigned
