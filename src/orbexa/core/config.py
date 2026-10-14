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

"""
ORBEX-A Configuration Module.

This module defines the `SimulationConfig` class, which encapsulates all
simulation parameters. It replaces global variables to ensure modularity
and support dependency injection.
"""

import math
from pathlib import Path
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from orbexa.utils.io_utils import load_config

logger = logging.getLogger(__name__)


def _as_square_weight_matrix(value: Any, size: int) -> np.ndarray:
    """Normalize scalar/vector/matrix YAML weights into a square matrix."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.eye(size) * float(arr)
    if arr.ndim == 1:
        if arr.shape[0] != size:
            raise ValueError(f"Expected {size} weights, got {arr.shape[0]}")
        return np.diag(arr)
    if arr.shape == (size, size):
        return arr
    raise ValueError(f"Expected scalar, length-{size}, or {size}x{size} weight matrix")


@dataclass
class OrbitConfig:
    """Orbital mechanics configuration."""

    mu: float
    """Standard gravitational parameter for Earth (m^3/s^2)."""

    semi_major_axis: float
    """Semi-major axis (meters)."""

    eccentricity: float
    """Orbital eccentricity."""

    time_periapsis: float
    """Time of periapsis passage."""

    initial_conditions: np.ndarray
    """Physical LVLH state [x, y, z, dx/dq, dy/dq, dz/dq]."""

    target_drag: float = 0.0
    """Target quadratic drag constant alpha."""

    chaser_drag: float = 0.0
    """Chaser quadratic drag constant beta."""

    central_body_radius: float = 0.0
    """Reference surface radius in meters; zero disables surface admissibility."""

    def __post_init__(self) -> None:
        if not (np.isfinite(self.mu) and self.mu > 0 and
                np.isfinite(self.semi_major_axis) and self.semi_major_axis > 0 and
                0 <= self.eccentricity < 1):
            raise ValueError("The reference orbit must be a finite bound ellipse")
        if self.central_body_radius < 0 or not np.isfinite(self.central_body_radius):
            raise ValueError("central_body_radius must be finite and nonnegative")
        if self.semi_major_axis * (1 - self.eccentricity) <= self.central_body_radius:
            raise ValueError("Reference-orbit perigee intersects the central body")
        if min(self.target_drag, self.chaser_drag) < 0:
            raise ValueError("Drag coefficients must be nonnegative")

    @property
    def mean_motion(self) -> float:
        """Mean motion (rad/s) derived from mu and semi-major axis."""
        return np.sqrt(self.mu / self.semi_major_axis**3)

    @property
    def specific_angular_momentum(self) -> float:
        """Specific angular momentum for the configured Keplerian orbit."""
        return float(
            np.sqrt(self.mu * self.semi_major_axis * (1.0 - self.eccentricity**2))
        )

    def dynamics_params(self) -> Dict[str, float]:
        """Runtime parameters for the paper's relative dynamics model."""
        return {
            "mean_motion": self.mean_motion,
            "eccentricity": self.eccentricity,
            "time_periapsis": self.time_periapsis,
            "mu": self.mu,
            "semi_major_axis": self.semi_major_axis,
            "specific_angular_momentum": self.specific_angular_momentum,
            "alpha": self.target_drag,
            "beta": self.chaser_drag,
        }


@dataclass
class MPCConfig:
    """Model Predictive Control configuration."""

    horizon_steps: Dict[str, int]
    actuation_steps: Dict[str, int]

    state_bounds: List[Dict[str, Any]]
    input_bounds: List[Dict[str, float]]
    goal_bounds: Optional[List[float]] = None
    """Optional shell radii for standalone Chaser goal allocation, in meters."""
    min_chaser_separation: float = 0.35

    Q: np.ndarray = field(default_factory=lambda: np.eye(6))
    R: np.ndarray = field(default_factory=lambda: np.eye(3) * 1e-4)
    terminal_cost_enabled: bool = True
    reference_preview_enabled: bool = True
    docking_position_tolerance: float = 0.002
    docking_velocity_tolerance: float = 0.08
    terminal_dwell_updates: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.terminal_cost_enabled, bool):
            raise ValueError("mpc.terminal_cost_enabled must be a boolean")
        if not isinstance(self.reference_preview_enabled, bool):
            raise ValueError("mpc.reference_preview_enabled must be a boolean")
        if (not isinstance(self.terminal_dwell_updates, int) or
                isinstance(self.terminal_dwell_updates, bool) or self.terminal_dwell_updates <= 0):
            raise ValueError("mpc.terminal_dwell_updates must be a positive integer")
        if any(not np.isfinite(value) or value <= 0 for value in
               (self.docking_position_tolerance, self.docking_velocity_tolerance)):
            raise ValueError("Docking tolerances must be finite and positive")

    def bounds(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        return self.state_bounds, self.input_bounds


@dataclass
class TargetConfig:
    """Target spacecraft configuration."""

    initial_orientation: np.ndarray
    initial_angular_velocity: np.ndarray
    """Prescribed Euler angle derivatives per anomaly, not body rates in rad/s."""
    inertia: np.ndarray
    shape: str
    center: np.ndarray
    limits: Dict[str, float]
    tolerance: float
    docking_standoff: float
    assignment_strategy: str
    stop_time: float
    discretize_dockers: bool

    observation_error: Dict[str, float]

    @property
    def radius(self) -> float:
        return float(self.limits.get("target_radius", self.limits.get("r_T", 0.0)))

    @property
    def height(self) -> float:
        if "target_height" in self.limits:
            return float(self.limits["target_height"])
        if "target_half_length" in self.limits:
            return 2.0 * float(self.limits["target_half_length"])
        if "half_length" in self.limits:
            return 2.0 * float(self.limits["half_length"])
        if "l_T" in self.limits:
            return 2.0 * float(self.limits["l_T"])
        return 0.0

    @property
    def half_length(self) -> float:
        if "target_half_length" in self.limits:
            return float(self.limits["target_half_length"])
        if "half_length" in self.limits:
            return float(self.limits["half_length"])
        if "l_T" in self.limits:
            return float(self.limits["l_T"])
        return self.height / 2.0

    @property
    def bounding_sphere_radius(self) -> float:
        return float(np.sqrt(self.radius**2 + self.half_length**2))

    @property
    def rendezvous_radius(self) -> float:
        return self.bounding_sphere_radius * (1.0 + self.tolerance)

    def collision_params(self, operation: str = "rendezvous") -> Dict[str, Any]:
        """Translate target geometry into solver constraint parameters."""
        radius = self.radius * (1.0 + self.tolerance)
        half_length = self.half_length * (1.0 + self.tolerance)
        return {
            "operation": operation,
            "shape": self.shape,
            "center": self.center,
            "orientation": self.initial_orientation,
            "target_radius": radius,
            "target_height": self.height * (1.0 + self.tolerance),
            "target_half_length": half_length,
            "rendezvous_radius": self.rendezvous_radius,
            "docking_standoff": self.docking_standoff,
            "assignment_strategy": self.assignment_strategy,
            "tube_radius": 0.0,
        }


@dataclass
class TubeMPCConfig:
    """Tube MPC settings."""

    run_tube: bool
    bandwidth_0: List[float]
    boundary_layer_0: List[float]
    sliding_gains: List[float]

    # Adaptation ranges
    bandwidth_range: Dict[str, float]
    feedback_substeps: int = 32
    bandwidth_rate_limit: float = 1500.0
    bandwidth_weight: float = 1.0e-7
    tube_weight: float = 10.0
    numerical_reserve: float = 1.0e-5
    membership_error_bound: float = 1.0e-7
    membership_box_budget: int = 32
    membership_update_period: int = 5

    def __post_init__(self) -> None:
        for name in ("bandwidth_0", "sliding_gains", "boundary_layer_0"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"tube.{name} must contain three finite values")
            if np.any(value <= 0):
                raise ValueError(f"tube.{name} must be positive")
        low, high = self.bandwidth_range["lower"], self.bandwidth_range["upper"]
        if not 0 < low <= high or self.feedback_substeps < 1:
            raise ValueError("Invalid tube bandwidth range or feedback sampling")
        if any(not np.isfinite(value) or value < 0 for value in
               (self.numerical_reserve, self.membership_error_bound, self.bandwidth_rate_limit,
                self.bandwidth_weight, self.tube_weight)):
            raise ValueError("Tube numerical reserves, limits and weights must be finite and nonnegative")
        if self.membership_box_budget < 1 or self.membership_update_period < 1:
            raise ValueError("Membership budget and update period must be positive")

    def runtime_params(
        self,
        *,
        enabled: bool,
        eccentricity_range: Tuple[float, float],
        target_drag_range: Tuple[float, float],
        chaser_drag_range: Tuple[float, float],
    ) -> Optional[Dict[str, Any]]:
        if not enabled:
            return None
        return {
            "enabled": True,
            "lambda_gain": self.sliding_gains,
            "alpha": self.bandwidth_0,
            "phi": self.boundary_layer_0,
            "eccentricity_range": eccentricity_range,
            "aRange": target_drag_range,
            "bRange": chaser_drag_range,
            "tighten_inputs": True,
        }


@dataclass
class SimulationConfig:
    """Main simulation configuration."""

    seed: int
    anom_step: float
    total_anom: float
    num_update_steps: int
    decoupled_mode: bool

    orbit: OrbitConfig
    mpc: MPCConfig
    target: TargetConfig
    tube: TubeMPCConfig

    num_chasers: int

    @property
    def iter_anomaly(self) -> float:
        return self.num_update_steps * self.anom_step

    def dynamics_params(self) -> Dict[str, float]:
        params = self.orbit.dynamics_params()
        params["anom_step"] = self.anom_step
        return params

    def default_adaptation_params(self) -> Dict[str, Any]:
        """Paper-scale default feasible sets for SMID when YAML is silent."""
        return {
            "eccentricity": [0.0, max(0.6, float(self.orbit.eccentricity))],
            "alpha": [0.0, max(5.0e-7, float(self.orbit.target_drag))],
            "beta": [float(self.orbit.chaser_drag), float(self.orbit.chaser_drag)],
        }

    @classmethod
    def load(cls, path: Optional[str] = None) -> "SimulationConfig":
        """Load configuration from a YAML file."""
        logger.debug(f"Loading configuration from {path}")
        if path is None:
            path = Path(__file__).resolve().parents[3] / "config" / "default.yaml"
            if not path.is_file():
                raise FileNotFoundError("Default scenario is unavailable; supply a scenario YAML with --config PATH")
        data = load_config(path)
        if data is None:
            logger.error(f"Failed to load configuration from {path}")
            raise ValueError(f"Failed to load configuration from {path}")

        # Helper to convert list lists to numpy arrays
        def to_numpy(val):
            if isinstance(val, list):
                return np.array(val)
            return val

        # Parse sections
        orbit_data = data["orbit"]
        orbit_config = OrbitConfig(
            mu=float(orbit_data["mu"]),
            semi_major_axis=float(orbit_data["semi_major_axis"]),
            eccentricity=float(orbit_data["eccentricity"]),
            time_periapsis=float(orbit_data["time_periapsis"]),
            initial_conditions=to_numpy(orbit_data["initial_conditions"]),
            target_drag=float(orbit_data.get("target_drag", orbit_data.get("alpha", 0.0))),
            chaser_drag=float(orbit_data.get("chaser_drag", orbit_data.get("beta", 0.0))),
            central_body_radius=float(orbit_data.get("central_body_radius", 0.0)),
        )

        mpc_data = data["mpc"]
        mpc_config = MPCConfig(
            horizon_steps=mpc_data["horizon_steps"],
            actuation_steps=mpc_data["actuation_steps"],
            state_bounds=mpc_data["state_bounds"],
            input_bounds=mpc_data["input_bounds"],
            goal_bounds=mpc_data.get("goal_bounds"),
            min_chaser_separation=float(mpc_data.get("min_chaser_separation", 0.35)),
            Q=_as_square_weight_matrix(mpc_data.get("Q", np.eye(6).tolist()), 6),
            R=_as_square_weight_matrix(
                mpc_data.get("R", (np.eye(3) * 1e-4).tolist()), 3
            ),
            terminal_cost_enabled=mpc_data.get("terminal_cost_enabled", True),
            reference_preview_enabled=mpc_data.get("reference_preview_enabled", True),
            docking_position_tolerance=float(mpc_data.get("docking_position_tolerance", 0.002)),
            docking_velocity_tolerance=float(mpc_data.get("docking_velocity_tolerance", 0.08)),
            terminal_dwell_updates=int(mpc_data.get("terminal_dwell_updates", 5)),
        )

        target_data = data["target"]
        target_config = TargetConfig(
            initial_orientation=to_numpy(target_data["initial_orientation"]),
            initial_angular_velocity=to_numpy(target_data["initial_angular_velocity"]),
            inertia=to_numpy(target_data["inertia"]),
            shape=target_data["shape"],
            center=to_numpy(target_data["center"]),
            limits=target_data["limits"],
            tolerance=float(target_data.get("tolerance", 0.0)),
            docking_standoff=float(target_data.get("docking_standoff", 0.12)),
            assignment_strategy=target_data.get(
                "assignment_strategy", "cylinder_side_approach_hungarian"
            ),
            stop_time=target_data["stop_time"],
            discretize_dockers=target_data["discretize_dockers"],
            observation_error=target_data["observation_error"],
        )

        tube_data = data["tube"]
        tube_config = TubeMPCConfig(
            run_tube=tube_data["run_tube"],
            bandwidth_0=tube_data["bandwidth_0"],
            boundary_layer_0=tube_data["boundary_layer_0"],
            sliding_gains=tube_data["sliding_gains"],
            bandwidth_range=tube_data["bandwidth_range"],
            feedback_substeps=int(tube_data.get("feedback_substeps", 32)),
            bandwidth_rate_limit=float(tube_data.get("bandwidth_rate_limit", 1500.0)),
            bandwidth_weight=float(tube_data.get("bandwidth_weight", 1.0e-7)),
            tube_weight=float(tube_data.get("tube_weight", 10.0)),
            numerical_reserve=float(tube_data.get("numerical_reserve", 1.0e-5)),
            membership_error_bound=float(tube_data.get("membership_error_bound", 1.0e-7)),
            membership_box_budget=int(tube_data.get("membership_box_budget", 32)),
            membership_update_period=int(tube_data.get("membership_update_period", 5)),
        )

        config = cls(
            seed=data["seed"],
            anom_step=data["anom_step"],
            total_anom=data["total_anom"],
            num_update_steps=data["num_update_steps"],
            decoupled_mode=data["decoupled_mode"],
            num_chasers=data["num_chasers"],
            orbit=orbit_config,
            mpc=mpc_config,
            target=target_config,
            tube=tube_config,
        )
        logger.info(f"Configuration loaded successfully from {path}")
        return config
