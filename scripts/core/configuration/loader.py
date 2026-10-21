"""YAML decoding into structured configuration, loaded on demand."""

import logging
from typing import Optional
import numpy as np
from orbexa.layout import default_config_path
from orbexa.utils.io.configuration import load_config
from orbexa.core.configuration.types import OrbitConfig, MPCConfig, TargetConfig, TubeMPCConfig, MissionPolicyConfig, AdaptationConfig, SolverConfig, _as_square_weight_matrix

logger = logging.getLogger(__name__)


def load_simulation_config(path: Optional[str] = None, *, config_type=None):
    """Decode a scenario into the supplied simulation configuration class."""
    from orbexa.core.configuration.model import SimulationConfig
    cls = config_type or SimulationConfig
    logger.debug(f"Loading configuration from {path}")
    if path is None:
        path = default_config_path()
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
        feedback_substeps=tube_data.get("feedback_substeps", 32),
        bandwidth_rate_limit=float(tube_data.get("bandwidth_rate_limit", 1500.0)),
        bandwidth_weight=float(tube_data.get("bandwidth_weight", 1.0e-7)),
        tube_weight=float(tube_data.get("tube_weight", 10.0)),
        numerical_reserve=float(tube_data.get("numerical_reserve", 1.0e-5)),
        membership_error_bound=float(tube_data.get("membership_error_bound", 1.0e-7)),
        membership_box_budget=tube_data.get("membership_box_budget", 32),
        membership_update_period=tube_data.get("membership_update_period", 5),
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
        mission=MissionPolicyConfig(**data.get("mission", {})),
        adaptation=AdaptationConfig(**data.get("adaptation", {})),
        solver=SolverConfig(**data.get("solver", {})),
    )
    logger.info(f"Configuration loaded successfully from {path}")
    return config
