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
from typing import Dict, Optional, Tuple

import numpy as np

from orbexa.utils.geometry.algebra import tait_bryan_to_rotation_matrix


@dataclass(frozen=True)
class CylinderConstraint:
    """Rotating cylindrical target exclusion model from the paper."""

    radius: float
    half_length: float
    orientation: np.ndarray
    tube_radius: float = 0.0

    def margins(self, position: np.ndarray) -> Tuple[float, float]:
        """Return length-valued radial/axial margins in metres.

        The maximum is an exclusion margin, with safety when nonnegative. It
        equals penetration depth inside the finite cylinder; outside a rim it
        is a conservative distance rather than the Euclidean signed distance.
        """
        p_body = target_frame_position(position, self.orientation)
        radial_margin = np.hypot(p_body[0], p_body[1]) - (self.radius + self.tube_radius)
        axial_margin = abs(p_body[2]) - (self.half_length + self.tube_radius)
        return float(radial_margin), float(axial_margin)

    def is_satisfied(self, position: np.ndarray) -> bool:
        radial_margin, axial_margin = self.margins(position)
        return max(radial_margin, axial_margin) >= 0.0


def target_frame_position(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Express an LVLH-frame position in the rotating target body frame."""
    value = np.asarray(orientation, dtype=float)
    if value.shape == (3, 3):
        if (not np.all(np.isfinite(value)) or
                not np.allclose(value.T @ value, np.eye(3), atol=1e-10, rtol=0) or
                not np.isclose(np.linalg.det(value), 1.0, atol=1e-10, rtol=0)):
            raise ValueError("Target rotation matrix must be finite and proper")
        rotation = value
    else:
        rotation = tait_bryan_to_rotation_matrix(value)
    return rotation.T @ np.asarray(position, dtype=float)


def rotating_docking_point(
    docking_point_body: np.ndarray, orientation: np.ndarray
) -> np.ndarray:
    """Map a body-fixed docking point to the LVLH frame."""
    rotation = tait_bryan_to_rotation_matrix(np.asarray(orientation, dtype=float))
    return rotation @ np.asarray(docking_point_body, dtype=float)


def rotating_body_point_velocity(
    body_point: np.ndarray, orientation: np.ndarray, orientation_rate: np.ndarray
) -> np.ndarray:
    """Derivative of ``R(roll, pitch, yaw) @ body_point`` for body-fixed points."""
    roll, pitch, yaw = np.asarray(orientation, dtype=float)
    roll_rate, pitch_rate, yaw_rate = np.asarray(orientation_rate, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    d_r_d_roll = np.array(
        [
            [0.0, 0.0, 0.0],
            [-sr * sy + cr * sp * cy, -sr * cy - cr * sp * sy, -cr * cp],
            [cr * sy + sr * sp * cy, cr * cy - sr * sp * sy, -sr * cp],
        ],
        dtype=float,
    )
    d_r_d_pitch = np.array(
        [
            [-sp * cy, sp * sy, cp],
            [sr * cp * cy, -sr * cp * sy, sr * sp],
            [-cr * cp * cy, cr * cp * sy, -cr * sp],
        ],
        dtype=float,
    )
    d_r_d_yaw = np.array(
        [
            [-cp * sy, -cp * cy, 0.0],
            [cr * cy - sr * sp * sy, -cr * sy - sr * sp * cy, 0.0],
            [sr * cy + cr * sp * sy, -sr * sy + cr * sp * cy, 0.0],
        ],
        dtype=float,
    )
    rotation_dot = (
        roll_rate * d_r_d_roll
        + pitch_rate * d_r_d_pitch
        + yaw_rate * d_r_d_yaw
    )
    return rotation_dot @ np.asarray(body_point, dtype=float)


def rotating_body_point_acceleration(
    body_point: np.ndarray,
    orientation: np.ndarray,
    orientation_rate: np.ndarray,
) -> np.ndarray:
    """Second derivative of a body-fixed point for constant Euler angle rates.

    This follows the repository's ``Rx(roll) @ Ry(pitch) @ Rz(yaw)`` convention.
    Rates are derivatives with respect to the dynamics' independent variable,
    which is true anomaly in the ADTMPC mission.
    """
    roll, pitch, yaw = np.asarray(orientation, dtype=float)
    roll_rate, pitch_rate, yaw_rate = np.asarray(orientation_rate, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    gx = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    gy = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=float)
    gz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    rotation_ddot = (
        roll_rate**2 * gx @ gx @ rx @ ry @ rz
        + pitch_rate**2 * rx @ gy @ gy @ ry @ rz
        + yaw_rate**2 * rx @ ry @ gz @ gz @ rz
        + 2.0 * roll_rate * pitch_rate * gx @ rx @ gy @ ry @ rz
        + 2.0 * roll_rate * yaw_rate * gx @ rx @ ry @ gz @ rz
        + 2.0 * pitch_rate * yaw_rate * rx @ gy @ ry @ gz @ rz
    )
    return rotation_ddot @ np.asarray(body_point, dtype=float)


def rendezvous_margin(
    position: np.ndarray, target_radius: float, tube_radius: float = 0.0
) -> float:
    """Positive margin means the chaser is outside the rendezvous exclusion sphere."""
    return float(np.linalg.norm(position) - target_radius - tube_radius)


def project_box_halfspace(
    control: np.ndarray,
    normal: np.ndarray,
    offset: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Optional[np.ndarray]:
    """Nearest bounded input satisfying ``normal @ input >= offset``.

    A scalar multiplier solves the Euclidean projection onto a box intersected
    with one halfspace. Return ``None`` when the bounded halfspace is infeasible.
    """
    control, normal, lower, upper = (
        np.asarray(value, dtype=float) for value in (control, normal, lower, upper)
    )
    if (control.ndim != 1 or any(value.shape != control.shape for value in (normal, lower, upper))
            or not np.all(np.isfinite(control)) or not np.all(np.isfinite(normal))
            or not np.isfinite(offset) or np.any(np.isnan(lower)) or np.any(np.isnan(upper))
            or np.any(np.isposinf(lower)) or np.any(np.isneginf(upper))
            or np.any(lower > upper)):
        raise ValueError("Halfspace projection requires finite control/normal/offset and a valid input box")
    bounded = np.clip(control, lower, upper)
    if float(normal @ bounded) >= offset:
        return bounded
    original_normal, original_offset = normal, offset
    scale = float(np.max(np.abs(normal)))
    if scale == 0.0:
        return None
    normal, offset = normal / scale, offset / scale
    if not np.isfinite(offset):
        return None
    positive, negative = normal > 0, normal < 0
    maximizer = bounded.copy()
    maximizer[positive], maximizer[negative] = upper[positive], lower[negative]
    maximum = float(normal @ maximizer)
    if maximum < offset:
        return None
    high = max(1.0, (offset - float(normal @ bounded)) / float(normal @ normal))
    for _ in range(128):
        if float(normal @ np.clip(control + high * normal, lower, upper)) >= offset:
            break
        high *= 2.0
        if not np.isfinite(high):
            return None
    else:
        return None
    low = 0.0
    for _ in range(60):
        midpoint = 0.5 * (low + high)
        if float(normal @ np.clip(control + midpoint * normal, lower, upper)) < offset:
            low = midpoint
        else:
            high = midpoint
    projected = np.clip(control + high * normal, lower, upper)
    tolerance = 8.0 * np.finfo(float).eps * max(1.0, abs(original_offset))
    if (not np.all(np.isfinite(projected)) or np.any(projected < lower)
            or np.any(projected > upper)
            or float(original_normal @ projected) < original_offset - tolerance):
        return None
    return projected


def collision_params_from_target_config(
    target_config, operation: str = "rendezvous"
) -> Dict[str, object]:
    """Build solver-ready target collision parameters from a TargetConfig."""
    return target_config.collision_params(operation=operation)
