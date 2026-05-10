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
    """Initial relative state vector [x, y, z, vx, vy, vz]."""

    target_drag: float = 0.0
    """Target quadratic drag constant alpha."""

    chaser_drag: float = 0.0
    """Chaser quadratic drag constant beta."""

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
    force_bounds: List[Dict[str, float]]
    goal_bounds: List[float]
    min_chaser_separation: float = 0.35

    Q: np.ndarray = field(default_factory=lambda: np.eye(6))
    R: np.ndarray = field(default_factory=lambda: np.eye(3) * 1e-4)

    def bounds(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        return self.state_bounds, self.input_bounds


@dataclass
class TargetConfig:
    """Target spacecraft configuration."""

    initial_orientation: np.ndarray
    initial_angular_velocity: np.ndarray
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
    convergence_rate: List[float]
    sliding_gains: List[float]

    # Adaptation ranges
    bandwidth_range: Dict[str, float]

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
    def load(cls, path: str = "config/default.yaml") -> "SimulationConfig":
        """Load configuration from a YAML file."""
        logger.debug(f"Loading configuration from {path}")
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
        )

        mpc_data = data["mpc"]
        mpc_config = MPCConfig(
            horizon_steps=mpc_data["horizon_steps"],
            actuation_steps=mpc_data["actuation_steps"],
            state_bounds=mpc_data["state_bounds"],
            input_bounds=mpc_data["input_bounds"],
            force_bounds=mpc_data["force_bounds"],
            goal_bounds=mpc_data["goal_bounds"],
            min_chaser_separation=float(mpc_data.get("min_chaser_separation", 0.35)),
            Q=_as_square_weight_matrix(mpc_data.get("Q", np.eye(6).tolist()), 6),
            R=_as_square_weight_matrix(
                mpc_data.get("R", (np.eye(3) * 1e-4).tolist()), 3
            ),
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
            convergence_rate=tube_data["convergence_rate"],
            sliding_gains=tube_data["sliding_gains"],
            bandwidth_range=tube_data["bandwidth_range"],
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
