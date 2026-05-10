#!/usr/bin/env python3
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

"""Paper-level rendezvous and docking constraint helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from orbexa.utils.math_utils import tait_bryan_to_rotation_matrix


@dataclass(frozen=True)
class CylinderConstraint:
    """Rotating cylindrical target exclusion model from the paper."""

    radius: float
    half_length: float
    orientation: np.ndarray
    tube_radius: float = 0.0

    def margins(self, position: np.ndarray) -> Tuple[float, float]:
        """Return radial and axial margins; safe when either value is nonnegative."""
        p_body = target_frame_position(position, self.orientation)
        radial_margin = (
            p_body[0] ** 2
            + p_body[1] ** 2
            - (self.radius + self.tube_radius) ** 2
        )
        axial_margin = abs(p_body[2]) - (self.half_length + self.tube_radius)
        return float(radial_margin), float(axial_margin)

    def is_satisfied(self, position: np.ndarray) -> bool:
        radial_margin, axial_margin = self.margins(position)
        return max(radial_margin, axial_margin) >= 0.0


def target_frame_position(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Express an LVLH-frame position in the rotating target body frame."""
    rotation = tait_bryan_to_rotation_matrix(np.asarray(orientation, dtype=float))
    return rotation.T @ np.asarray(position, dtype=float)


def rotating_docking_point(
    docking_point_body: np.ndarray, orientation: np.ndarray
) -> np.ndarray:
    """Map a body-fixed docking point to the LVLH frame."""
    rotation = tait_bryan_to_rotation_matrix(np.asarray(orientation, dtype=float))
    return rotation @ np.asarray(docking_point_body, dtype=float)


def rendezvous_margin(
    position: np.ndarray, target_radius: float, tube_radius: float = 0.0
) -> float:
    """Positive margin means the chaser is outside the rendezvous exclusion sphere."""
    return float(np.linalg.norm(position) - target_radius - tube_radius)


def collision_params_from_target_config(
    target_config, operation: str = "rendezvous"
) -> Dict[str, object]:
    """Build solver-ready target collision parameters from a TargetConfig."""
    return target_config.collision_params(operation=operation)
