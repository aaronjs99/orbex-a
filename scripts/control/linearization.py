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

"""Affine constraint approximations for QP-ready MPC experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from orbexa.control.constraints import CylinderConstraint
from orbexa.utils.math_utils import tait_bryan_to_rotation_matrix


@dataclass(frozen=True)
class AffineConstraint:
    """Half-space ``normal @ x + offset >= 0``."""

    normal: np.ndarray
    offset: float
    source: str

    def margin(self, state: np.ndarray) -> float:
        return float(self.normal @ np.asarray(state, dtype=float) + self.offset)


def linearize_rendezvous_constraint(
    position: np.ndarray,
    *,
    target_radius: float,
    tube_radius: float = 0.0,
) -> AffineConstraint:
    """Linearize ``||p|| - radius >= 0`` at ``position``."""
    p = np.asarray(position, dtype=float)
    norm = float(np.linalg.norm(p))
    if norm <= 1.0e-12:
        raise ValueError("Cannot linearize rendezvous constraint at the origin")
    normal = p / norm
    radius = target_radius + tube_radius
    offset = norm - radius - normal @ p
    return AffineConstraint(normal=normal, offset=float(offset), source="rendezvous")


def linearize_cylinder_constraint(
    position: np.ndarray,
    cylinder: CylinderConstraint,
    *,
    active: Literal["auto", "radial", "axial"] = "auto",
) -> AffineConstraint:
    """
    Linearize the active member of the docking union constraint.

    The nonlinear path remains authoritative; this approximation exists for
    sequential convex/QP experiments.
    """
    p = np.asarray(position, dtype=float)
    rotation = tait_bryan_to_rotation_matrix(cylinder.orientation)
    body_from_lvlh = rotation.T
    p_body = body_from_lvlh @ p
    radial_margin, axial_margin = cylinder.margins(p)

    chosen = active
    if chosen == "auto":
        chosen = "radial" if radial_margin >= axial_margin else "axial"

    if chosen == "radial":
        body_normal = np.array([2.0 * p_body[0], 2.0 * p_body[1], 0.0])
        normal = rotation @ body_normal
        offset = radial_margin - normal @ p
        return AffineConstraint(normal=normal, offset=float(offset), source="cylinder_radial")

    z_sign = 1.0 if p_body[2] >= 0.0 else -1.0
    body_normal = np.array([0.0, 0.0, z_sign])
    normal = rotation @ body_normal
    offset = axial_margin - normal @ p
    return AffineConstraint(normal=normal, offset=float(offset), source="cylinder_axial")
