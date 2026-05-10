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
MPC Problem Builder

Constructs solver-agnostic MPCProblem objects from ORBEX-A mission parameters.
This bridges the gap between high-level mission specs and the generic solver interface.
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass, field

from orbexa.solvers.base import MPCProblem


def _square_weight_matrix(matrix: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        return np.diag(arr)
    if arr.ndim == 0:
        return np.eye(size) * float(arr)
    return arr


@dataclass
class ORBEXAProblemConfig:
    """
    Configuration for building ORBEX-A MPC problems.

    This holds all the mission-specific parameters that get translated
    into a generic MPCProblem for the solver.
    """

    # Time parameters
    start_anom: float = 0.0
    anom_step: float = 0.001
    num_steps: int = 80

    # States and goals
    initial_state: np.ndarray = field(default_factory=lambda: np.zeros(6))
    final_state: np.ndarray = field(default_factory=lambda: np.zeros(6))
    control_0: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Dynamics (can be matrices OR callables)
    dynamics_matrix: Union[np.ndarray, Callable, Any] = None
    input_matrix: Optional[np.ndarray] = None
    state_cost_matrix: np.ndarray = field(default_factory=lambda: np.eye(6))
    input_cost_matrix: np.ndarray = field(default_factory=lambda: np.eye(3) * 1e-4)
    disturbance: Union[np.ndarray, Callable] = None  # Disturbance

    # Constraints
    state_bounds: Optional[List[Dict]] = None
    input_bounds: Optional[List[Dict]] = None

    # Orbital mechanics params (for time-varying dynamics)
    eccentricity: float = 0.0
    time_periapsis: float = 0.0
    mean_motion: float = 0.001

    # Tube MPC settings
    tube_mpc_enabled: bool = False
    tube_params: Dict[str, Any] = field(default_factory=dict)

    # Anomaly scaling
    use_anomaly_scaling: bool = False

    # Target tracking
    target_params: Optional[Dict[str, Any]] = None
    affine_constraints: List[Any] = field(default_factory=list)


def build_mpc_problem(config: ORBEXAProblemConfig) -> MPCProblem:
    """
    Build a generic MPCProblem from ORBEX-A configuration.

    Args:
        config: ORBEX-A problem configuration

    Returns:
        MPCProblem ready for any solver backend
    """
    if hasattr(config.dynamics_matrix, "as_tuple"):
        model = config.dynamics_matrix
        config.dynamics_matrix = model.A
        config.input_matrix = model.B
        config.state_cost_matrix = model.Q
        config.input_cost_matrix = model.R
        config.disturbance = model.d
        config.state_bounds = config.state_bounds or model.state_bounds
        config.input_bounds = config.input_bounds or model.input_bounds

    # Handle time-varying A matrix
    if callable(config.dynamics_matrix):
        # Evaluate at start time for initial problem setup
        # Solvers that support time-varying dynamics will use the callable
        A_eval = config.dynamics_matrix(config.start_anom, config.time_periapsis)
        if hasattr(A_eval, "__iter__") and not isinstance(A_eval, np.ndarray):
            A_eval = np.array(A_eval)
    else:
        A_eval = (
            config.dynamics_matrix
            if config.dynamics_matrix is not None
            else np.zeros((6, 6))
        )

    # Handle B matrix
    B_raw = config.input_matrix
    if callable(B_raw):
        # Evaluate B. Arguments ignored if constant, or use start_anom if needed.
        # B_func typically takes *args, **kwargs and returns constant B.
        B = B_raw(config.start_anom, config.time_periapsis)
        if not isinstance(B, np.ndarray):
            B = np.array(B)
    elif B_raw is not None:
        B = B_raw
    else:
        B = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Build extra params for solver-specific features
    extra_params = {
        "dynamics_callable": (
            config.dynamics_matrix if callable(config.dynamics_matrix) else None
        ),
        "disturbance_callable": (
            config.disturbance if callable(config.disturbance) else None
        ),
        "disturbance_vector": (
            None
            if callable(config.disturbance) or config.disturbance is None
            else np.asarray(config.disturbance, dtype=float)
        ),
        "start_anom": config.start_anom,
        "time_periapsis": config.time_periapsis,
        "eccentricity": config.eccentricity,
        "mean_motion": config.mean_motion,
        "u_0": config.control_0,
        "use_anomaly_scaling": config.use_anomaly_scaling,
    }

    # Calculate initial true anomaly q_0 for solver prediction
    if config.use_anomaly_scaling:
        ecc = config.eccentricity
        n_motion = config.mean_motion
        t_p = config.time_periapsis
        t = config.start_anom

        # Calculate initial true anomaly q_0 for solver prediction
        # M = n*(t - tp). Solve Kepler for E (approx), then q.
        mean_anomaly_0 = n_motion * (t - t_p)
        eccentric_anomaly_0 = mean_anomaly_0  # Initial guess
        for _ in range(5):
            eccentric_anomaly_0 = mean_anomaly_0 + ecc * np.sin(eccentric_anomaly_0)

        # q = 2 * atan(sqrt((1+e)/(1-e)) * tan(E/2))
        q_0 = 2 * np.arctan(
            np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(eccentric_anomaly_0 / 2)
        )
        extra_params["q_0"] = q_0

    # Add tube MPC if enabled
    if config.tube_mpc_enabled:
        extra_params["tube_mpc"] = config.tube_params

    # Add target params if present
    if config.target_params:
        extra_params["target_params"] = config.target_params

    if config.affine_constraints:
        extra_params["affine_constraints"] = config.affine_constraints

    if config.target_params and config.target_params.get("pairwise_constraints"):
        extra_params["pairwise_constraints"] = config.target_params[
            "pairwise_constraints"
        ]

    return MPCProblem(
        dynamics_matrix=A_eval,
        input_matrix=B,
        state_cost_matrix=_square_weight_matrix(
            config.state_cost_matrix, len(config.initial_state)
        ),
        input_cost_matrix=_square_weight_matrix(config.input_cost_matrix, B.shape[1]),
        initial_state=config.initial_state,
        final_state=config.final_state,
        num_steps=config.num_steps,
        anom_step=config.anom_step,
        state_bounds=config.state_bounds,
        input_bounds=config.input_bounds,
        dynamics_type="continuous",
        extra_params=extra_params,
    )


def build_from_dynamics(
    A_func: Optional[Callable] = None,
    input_matrix: Optional[np.ndarray] = None,
    state_cost_matrix: Optional[np.ndarray] = None,
    input_cost_matrix: Optional[np.ndarray] = None,
    d_func: Optional[Callable] = None,
    initial_state: Optional[np.ndarray] = None,
    final_state: Optional[np.ndarray] = None,
    start_anom: float = 0.0,
    num_steps: int = 1,
    anom_step: Optional[float] = None,
    time_periapsis: float = 0.0,
    eccentricity: float = 0.0,
    state_bounds: Optional[List[Dict]] = None,
    input_bounds: Optional[List[Dict]] = None,
    **kwargs,
) -> MPCProblem:
    """
    Convenience function to build MPCProblem directly from dynamics functions.

    This is the primary interface for the refactored mpc.py to use.
    """
    if A_func is None:
        A_func = kwargs.pop("A", None)
    input_matrix = input_matrix if input_matrix is not None else kwargs.pop("B", None)
    state_cost_matrix = (
        state_cost_matrix if state_cost_matrix is not None else kwargs.pop("Q", None)
    )
    input_cost_matrix = (
        input_cost_matrix if input_cost_matrix is not None else kwargs.pop("R", None)
    )
    if d_func is None:
        d_func = kwargs.pop("d", None)
    initial_state = (
        initial_state if initial_state is not None else kwargs.pop("x_0", None)
    )
    final_state = final_state if final_state is not None else kwargs.pop("x_f", None)
    if anom_step is None:
        anom_step = kwargs.pop("dt", None)
    time_periapsis = kwargs.pop("t_periapsis", time_periapsis)

    if A_func is None or input_matrix is None or d_func is None:
        raise ValueError("A_func, input_matrix, and d_func are required")
    if initial_state is None or final_state is None or anom_step is None:
        raise ValueError("initial_state, final_state, and anom_step are required")

    state_cost_matrix = (
        np.eye(len(initial_state)) if state_cost_matrix is None else state_cost_matrix
    )
    input_cost_matrix = (
        np.eye(input_matrix.shape[1]) if input_cost_matrix is None else input_cost_matrix
    )

    config = ORBEXAProblemConfig(
        start_anom=start_anom,
        anom_step=anom_step,
        num_steps=num_steps,
        initial_state=initial_state,
        final_state=final_state,
        dynamics_matrix=A_func,
        input_matrix=input_matrix,
        state_cost_matrix=state_cost_matrix,
        input_cost_matrix=input_cost_matrix,
        disturbance=d_func,
        state_bounds=state_bounds,
        input_bounds=input_bounds,
        eccentricity=eccentricity,
        time_periapsis=time_periapsis,
        mean_motion=kwargs.get("mean_motion", 0.001),
    )

    # Handle optional tube MPC
    if "tube_mpc" in kwargs:
        config.tube_mpc_enabled = True
        config.tube_params = kwargs["tube_mpc"]

    if "use_anomaly_scaling" in kwargs:
        config.use_anomaly_scaling = kwargs["use_anomaly_scaling"]

    if "target_params" in kwargs:
        config.target_params = kwargs["target_params"]

    if "affine_constraints" in kwargs:
        config.affine_constraints = kwargs["affine_constraints"]

    if "pairwise_constraints" in kwargs:
        if config.target_params is None:
            config.target_params = {}
        config.target_params["pairwise_constraints"] = kwargs["pairwise_constraints"]

    return build_mpc_problem(config)
