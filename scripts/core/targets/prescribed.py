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

"""Prescribed cylindrical target geometry and anomaly-domain kinematics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from orbexa.control.mpc.constraints import CylinderConstraint, rotating_body_point_acceleration, rotating_body_point_velocity, rotating_docking_point, target_frame_position

if TYPE_CHECKING:
    from orbexa.core.configuration.model import SimulationConfig


class TargetGeometry(Protocol):
    """Target operations used by fleet, reference and safety services.

    An injected implementation must supply collision data supported by the
    selected coupled solver; this interface does not add a new physical model.
    """

    docking_standoff: float
    rendezvous_radius: float
    bounding_sphere_radius: float

    def orientation_at(self, anom: float) -> np.ndarray: ...
    def angular_velocity_at(self, anom: float) -> np.ndarray: ...
    def rotation_matrix_at(self, anom: float) -> np.ndarray: ...
    def docking_state(self, docking_point_body: np.ndarray, anom: float) -> np.ndarray: ...
    def docking_acceleration(self, docking_point_body: np.ndarray, anom: float) -> np.ndarray: ...
    def collision_params(self, operation: str, anom: float) -> Dict[str, Any]: ...
    def assign_docking_targets(self, initial_states: Sequence[np.ndarray]) -> List[Dict[str, Any]]: ...
    def as_dict(self) -> Dict[str, Any]: ...


class CylinderTargetGeometry(TargetGeometry, Protocol):
    """Explicit cylinder-only extension used by the paper baseline."""

    radius: float
    half_length: float
    tolerance: float

    def cylinder_constraint(self, anom: float, tube_radius: float = 0.0) -> CylinderConstraint: ...


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
        for name in ("radius", "half_length"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Target {name} must be finite and positive")
        for name in ("tolerance", "docking_standoff"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"Target {name} must be finite and nonnegative")
        for name in ("initial_orientation", "angular_velocity"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"Target {name} must be a finite vector of length three")
            object.__setattr__(self, name, value.copy())

    @classmethod
    def from_config(cls, config: SimulationConfig) -> "TumblingCylinderTarget":
        if config.target.shape != "cylinder":
            raise ValueError("The default target requires shape='cylinder'; inject an explicit supported target")
        center = np.asarray(config.target.center, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)) or np.any(center != 0.0):
            raise ValueError("The default cylindrical target requires a finite zero LVLH center of length three")
        if config.target.assignment_strategy != "cylinder_side_approach_hungarian":
            raise ValueError(f"Unsupported default target assignment_strategy: {config.target.assignment_strategy}")
        return cls(
            radius=float(config.target.radius),
            half_length=float(config.target.half_length),
            initial_orientation=np.asarray(config.target.initial_orientation, dtype=float),
            angular_velocity=np.asarray(config.target.initial_angular_velocity, dtype=float),
            tolerance=float(config.target.tolerance),
            docking_standoff=float(config.target.docking_standoff),
        )

    @property
    def height(self) -> float:
        return 2.0 * self.half_length

    @property
    def bounding_sphere_radius(self) -> float:
        return float(np.sqrt(self.radius**2 + self.half_length**2))

    @property
    def rendezvous_radius(self) -> float:
        return self.bounding_sphere_radius * (1.0 + self.tolerance)

    def orientation_at(self, anom: float) -> np.ndarray:
        return self.initial_orientation + self.angular_velocity * float(anom)

    def angular_velocity_at(self, anom: float) -> np.ndarray:
        return np.asarray(self.angular_velocity, dtype=float)

    def rotation_matrix_at(self, anom: float) -> np.ndarray:
        from orbexa.utils.math_utils import tait_bryan_to_rotation_matrix
        return tait_bryan_to_rotation_matrix(self.orientation_at(anom))

    def approach_azimuth(self, initial_state: np.ndarray) -> float:
        position_body = self.approach_position_body(initial_state)
        radial_norm = float(np.linalg.norm(position_body[:2]))
        if radial_norm <= 1.0e-9:
            return 0.0
        return float(np.mod(np.arctan2(position_body[1], position_body[0]), 2.0 * np.pi))

    def approach_position_body(self, initial_state: np.ndarray) -> np.ndarray:
        return target_frame_position(
            np.asarray(initial_state[:3], dtype=float), self.initial_orientation
        )

    def approach_axial_coordinate(self, initial_state: np.ndarray) -> float:
        position_body = self.approach_position_body(initial_state)
        axial_limit = 0.8 * self.half_length
        return float(np.clip(position_body[2], -axial_limit, axial_limit))

    def side_docking_geometry(
        self, azimuth: float, axial: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        normal = np.array([np.cos(azimuth), np.sin(azimuth), 0.0], dtype=float)
        point = normal * (self.radius + self.docking_standoff)
        point[2] = float(np.clip(axial, -0.8 * self.half_length, 0.8 * self.half_length))
        return point, normal

    def docking_points_body(self, count: int, start_azimuth: float = 0.0) -> List[np.ndarray]:
        points = []
        for idx in range(count):
            angle = start_azimuth + 2.0 * np.pi * idx / max(count, 1)
            point, _ = self.side_docking_geometry(angle)
            points.append(point)
        return points

    def assign_docking_targets(
        self, initial_states: Sequence[np.ndarray]
    ) -> List[Dict[str, Any]]:
        from orbexa.planning.rendezvous.cylinder import assign_docking_targets
        return assign_docking_targets(self, initial_states)
    def docking_state(self, docking_point_body: np.ndarray, anom: float) -> np.ndarray:
        orientation = self.orientation_at(anom)
        position = rotating_docking_point(docking_point_body, orientation)
        velocity = rotating_body_point_velocity(
            docking_point_body,
            orientation,
            self.angular_velocity_at(anom),
        )
        return np.concatenate((position, velocity))

    def docking_acceleration(self, docking_point_body: np.ndarray, anom: float) -> np.ndarray:
        return rotating_body_point_acceleration(
            docking_point_body,
            self.orientation_at(anom),
            self.angular_velocity_at(anom),
        )

    def collision_params(self, operation: str, anom: float) -> Dict[str, Any]:
        radius = self.radius * (1.0 + self.tolerance)
        half_length = self.half_length * (1.0 + self.tolerance)
        return {
            "operation": operation,
            "shape": "cylinder",
            "active_safety_model": (
                "bounding_sphere" if operation == "rendezvous" else "rotating_cylinder_union"
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

    def cylinder_constraint(self, anom: float, tube_radius: float = 0.0) -> CylinderConstraint:
        return CylinderConstraint(
            radius=self.radius * (1.0 + self.tolerance),
            half_length=self.half_length * (1.0 + self.tolerance),
            orientation=self.orientation_at(anom),
            tube_radius=tube_radius,
        )

    def as_dict(self) -> Dict[str, Any]:
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
