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
from typing import Dict, List, Any, Optional

from orbexa.utils.io_utils import load_config

logger = logging.getLogger(__name__)


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

    @property
    def mean_motion(self) -> float:
        """Mean motion (rad/s) derived from mu and semi-major axis."""
        return np.sqrt(self.mu / self.semi_major_axis**3)


@dataclass
class MPCConfig:
    """Model Predictive Control configuration."""

    horizon_steps: Dict[str, int]
    actuation_steps: Dict[str, int]

    state_bounds: List[Dict[str, Any]]
    input_bounds: List[Dict[str, float]]
    force_bounds: List[Dict[str, float]]
    goal_bounds: List[float]

    Q: np.ndarray = field(default_factory=lambda: np.eye(6))
    R: np.ndarray = field(default_factory=lambda: np.eye(3) * 1e-4)


@dataclass
class TargetConfig:
    """Target spacecraft configuration."""

    initial_orientation: np.ndarray
    initial_angular_velocity: np.ndarray
    inertia: np.ndarray
    shape: str
    center: np.ndarray
    limits: Dict[str, float]
    stop_time: float
    discretize_dockers: bool

    observation_error: Dict[str, float]


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
        )

        mpc_data = data["mpc"]
        mpc_config = MPCConfig(
            horizon_steps=mpc_data["horizon_steps"],
            actuation_steps=mpc_data["actuation_steps"],
            state_bounds=mpc_data["state_bounds"],
            input_bounds=mpc_data["input_bounds"],
            force_bounds=mpc_data["force_bounds"],
            goal_bounds=mpc_data["goal_bounds"],
            Q=to_numpy(mpc_data.get("Q", np.eye(6).tolist())),
            R=to_numpy(mpc_data.get("R", (np.eye(3) * 1e-4).tolist())),
        )

        target_data = data["target"]
        target_config = TargetConfig(
            initial_orientation=to_numpy(target_data["initial_orientation"]),
            initial_angular_velocity=to_numpy(target_data["initial_angular_velocity"]),
            inertia=to_numpy(target_data["inertia"]),
            shape=target_data["shape"],
            center=to_numpy(target_data["center"]),
            limits=target_data["limits"],
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
