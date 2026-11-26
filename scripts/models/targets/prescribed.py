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

"""prescribed.py: Prescribed cylindrical target geometry and anomaly-domain kinematics.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from orbexa.models.geometry.constraints import (
    CylinderConstraint,
    rotating_body_point_acceleration,
    rotating_body_point_velocity,
    rotating_docking_point,
    target_frame_position,
)

if TYPE_CHECKING:
    from orbexa.models.configuration.model import SimulationConfig


class TargetGeometry(Protocol):
    """Target operations used by fleet, reference and safety services.

    An injected implementation must supply collision data supported by the
    selected coupled solver; this interface does not add a new physical model.
    """

    docking_standoff: float
    rendezvous_radius: float
    bounding_sphere_radius: float

    def orientation_at(self, anom: float) -> np.ndarray:
        """Return the target orientation at anomaly ``anom``."""
        ...

    def angular_velocity_at(self, anom: float) -> np.ndarray:
        """Return the target angular velocity at anomaly ``anom``."""
        ...

    def rotation_matrix_at(self, anom: float) -> np.ndarray:
        """Return the target rotation matrix at anomaly ``anom``."""
        ...

    def docking_state(self, docking_point_body: np.ndarray, anom: float) -> np.ndarray:
        """Return the docking-point state for a body-frame point at ``anom``."""
        ...

    def docking_acceleration(
        self, docking_point_body: np.ndarray, anom: float
    ) -> np.ndarray:
        """Return the docking-point acceleration for a body-frame point at ``anom``."""
        ...

    def collision_params(self, operation: str, anom: float) -> dict[str, Any]:
        """Return the collision parameters for ``operation`` at ``anom``."""
        ...

    def assign_docking_targets(
        self, initial_states: Sequence[np.ndarray]
    ) -> list[dict[str, Any]]:
        """Return the docking-target assignments for the initial states."""
        ...

    def as_dict(self) -> dict[str, Any]:
        """Return the target declaration as a dictionary."""
        ...


@dataclass(frozen=True)
class TumblingCylinderTarget:
    """Tumbling cylindrical target geometry and docking-point kinematics."""

    radius: float
    half_length: float
    initial_orientation: np.ndarray
    angular_velocity: np.ndarray
    tolerance: float = 0.0
    docking_standoff: float = 0.12

    def __post_init__(self) -> None:
        """Validate the fields and store orientation and rate as float copies.

        Raises ValueError when radius or half_length is not finite and positive, when
        tolerance or docking_standoff is not finite and nonnegative, or when
        initial_orientation or angular_velocity is not a finite vector of length three.
        """
        for name in ("radius", "half_length"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Target {name} must be finite and positive")
        for name in ("tolerance", "docking_standoff"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"Target {name} must be finite and nonnegative")
        for name in ("initial_orientation", "angular_velocity"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"Target {name} must be a finite vector of length three"
                )
            object.__setattr__(self, name, value.copy())

    @classmethod
    def from_config(cls, config: SimulationConfig) -> TumblingCylinderTarget:
        """Build the target from a simulation configuration.

        Raises ValueError when the target shape is not "cylinder", the center is not a
        finite zero vector of length three, or the assignment strategy is not
        "cylinder_side_approach_hungarian".
        """
        if config.target.shape != "cylinder":
            raise ValueError(
                "The default target requires shape='cylinder'; inject an explicit supported target"
            )
        center = np.asarray(config.target.center, dtype=float)
        if (
            center.shape != (3,)
            or not np.all(np.isfinite(center))
            or np.any(center != 0.0)
        ):
            raise ValueError(
                "The default cylindrical target requires a finite zero LVLH center of length three"
            )
        if config.target.assignment_strategy != "cylinder_side_approach_hungarian":
            raise ValueError(
                f"Unsupported default target assignment_strategy: {config.target.assignment_strategy}"
            )
        return cls(
            radius=float(config.target.radius),
            half_length=float(config.target.half_length),
            initial_orientation=np.asarray(
                config.target.initial_orientation, dtype=float
            ),
            angular_velocity=np.asarray(
                config.target.initial_angular_velocity, dtype=float
            ),
            tolerance=float(config.target.tolerance),
            docking_standoff=float(config.target.docking_standoff),
        )

    @property
    def height(self) -> float:
        """Return the cylinder height, m, as twice the half length."""
        return 2.0 * self.half_length

    @property
    def bounding_sphere_radius(self) -> float:
        """Return the radius, m, of the sphere bounding the cylinder."""
        return float(np.sqrt(self.radius**2 + self.half_length**2))

    @property
    def rendezvous_radius(self) -> float:
        """Return the bounding-sphere radius, m, scaled by one plus the tolerance."""
        return self.bounding_sphere_radius * (1.0 + self.tolerance)

    def orientation_at(self, anom: float) -> np.ndarray:
        """Return the Tait-Bryan orientation at anomaly ``anom``.

        Linear in ``anom``: the initial orientation plus the angular velocity times
        ``anom``.
        """
        return self.initial_orientation + self.angular_velocity * float(anom)

    def angular_velocity_at(self, anom: float) -> np.ndarray:
        """Return the constant angular velocity as a float array."""
        return np.asarray(self.angular_velocity, dtype=float)

    def rotation_matrix_at(self, anom: float) -> np.ndarray:
        """Return the rotation matrix of the orientation at anomaly ``anom``."""
        from orbexa.common.geometry.algebra import tait_bryan_to_rotation_matrix

        return tait_bryan_to_rotation_matrix(self.orientation_at(anom))

    def approach_azimuth(self, initial_state: np.ndarray) -> float:
        """Return the body-frame azimuth, rad in [0, 2*pi), of an initial position.

        Returns 0.0 when the radial (x, y) distance of the body-frame position is at
        most 1e-9 m.

        Inputs:
            initial_state: State whose first three entries are the LVLH position, m.
        """
        position_body = self.approach_position_body(initial_state)
        radial_norm = float(np.linalg.norm(position_body[:2]))
        if radial_norm <= 1.0e-9:
            return 0.0
        return float(
            np.mod(np.arctan2(position_body[1], position_body[0]), 2.0 * np.pi)
        )

    def approach_position_body(self, initial_state: np.ndarray) -> np.ndarray:
        """Return the initial LVLH position, m, expressed in the target body frame.

        Inputs:
            initial_state: State whose first three entries are the LVLH position, m.
        """
        return target_frame_position(
            np.asarray(initial_state[:3], dtype=float), self.initial_orientation
        )

    def approach_axial_coordinate(self, initial_state: np.ndarray) -> float:
        """Return the body-frame axial position, m, limited to 0.8 of the half length.

        Inputs:
            initial_state: State whose first three entries are the LVLH position, m.
        """
        position_body = self.approach_position_body(initial_state)
        axial_limit = 0.8 * self.half_length
        return float(np.clip(position_body[2], -axial_limit, axial_limit))

    def side_docking_geometry(
        self, azimuth: float, axial: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a side docking point and its outward normal in the body frame.

        The normal is the unit radial direction at ``azimuth``. The point lies at the
        radius plus docking standoff, with axial position clipped to 0.8 of the half
        length.

        Inputs:
            azimuth: Body-frame azimuth, rad.
            axial: Requested axial position, m.

        Outputs:
            Tuple of the docking point (m) and the unit normal.
        """
        normal = np.array([np.cos(azimuth), np.sin(azimuth), 0.0], dtype=float)
        point = normal * (self.radius + self.docking_standoff)
        point[2] = float(
            np.clip(axial, -0.8 * self.half_length, 0.8 * self.half_length)
        )
        return point, normal

    def docking_points_body(
        self, count: int, start_azimuth: float = 0.0
    ) -> list[np.ndarray]:
        """Return ``count`` side docking points evenly spaced in azimuth.

        Points lie at zero axial position in the body frame, m; the first is at
        ``start_azimuth`` (rad).
        """
        points = []
        for idx in range(count):
            angle = start_azimuth + 2.0 * np.pi * idx / max(count, 1)
            point, _ = self.side_docking_geometry(angle)
            points.append(point)
        return points

    def assign_docking_targets(
        self, initial_states: Sequence[np.ndarray]
    ) -> list[dict[str, Any]]:
        """Return the docking-target assignments from ``targets.docking``."""
        from orbexa.models.targets.docking import assign_docking_targets

        return assign_docking_targets(self, initial_states)

    def docking_state(self, docking_point_body: np.ndarray, anom: float) -> np.ndarray:
        """Return the LVLH position and velocity of a body-fixed docking point.

        Outputs:
            Six values: position, then its derivative with respect to ``anom``.
        """
        orientation = self.orientation_at(anom)
        position = rotating_docking_point(docking_point_body, orientation)
        velocity = rotating_body_point_velocity(
            docking_point_body,
            orientation,
            self.angular_velocity_at(anom),
        )
        return np.concatenate((position, velocity))

    def docking_acceleration(
        self, docking_point_body: np.ndarray, anom: float
    ) -> np.ndarray:
        """Return the second derivative of a docking point with respect to ``anom``."""
        return rotating_body_point_acceleration(
            docking_point_body,
            self.orientation_at(anom),
            self.angular_velocity_at(anom),
        )

    def collision_params(self, operation: str, anom: float) -> dict[str, Any]:
        """Return the cylinder collision parameters for ``operation`` at ``anom``.

        Radius, half length and height are scaled by one plus the tolerance. The active
        safety model is "bounding_sphere" for "rendezvous" and "rotating_cylinder_union"
        otherwise; the tube radius is zero.
        """
        radius = self.radius * (1.0 + self.tolerance)
        half_length = self.half_length * (1.0 + self.tolerance)
        return {
            "operation": operation,
            "shape": "cylinder",
            "active_safety_model": (
                "bounding_sphere"
                if operation == "rendezvous"
                else "rotating_cylinder_union"
            ),
            "center": [0.0, 0.0, 0.0],
            "orientation": self.orientation_at(anom).tolist(),
            "angular_velocity": self.angular_velocity_at(anom).tolist(),
            "target_radius": radius,
            "target_height": self.height * (1.0 + self.tolerance),
            "target_half_length": half_length,
            "rendezvous_radius": self.rendezvous_radius,
            "bounding_sphere_radius": self.bounding_sphere_radius,
            "docking_standoff": self.docking_standoff,
            "tube_radius": 0.0,
        }

    def cylinder_constraint(
        self, anom: float, tube_radius: float = 0.0
    ) -> CylinderConstraint:
        """Return the ``CylinderConstraint`` at anomaly ``anom``.

        Radius and half length are scaled by one plus the tolerance.

        Inputs:
            anom: Anomaly at which the orientation is evaluated.
            tube_radius: Tube radius, m.
        """
        return CylinderConstraint(
            radius=self.radius * (1.0 + self.tolerance),
            half_length=self.half_length * (1.0 + self.tolerance),
            orientation=self.orientation_at(anom),
            tube_radius=tube_radius,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the target geometry and kinematics as a dictionary."""
        return {
            "shape": "cylinder",
            "radius": float(self.radius),
            "height": float(self.height),
            "half_length": float(self.half_length),
            "bounding_sphere_radius": float(self.bounding_sphere_radius),
            "rendezvous_sphere_radius": float(self.rendezvous_radius),
            "initial_orientation": self.initial_orientation.tolist(),
            "angular_velocity": self.angular_velocity.tolist(),
            "tolerance": float(self.tolerance),
            "docking_standoff": float(self.docking_standoff),
            "docking_clearance": float(self.docking_standoff),
            "assignment_strategy": "cylinder_side_approach_hungarian",
        }
