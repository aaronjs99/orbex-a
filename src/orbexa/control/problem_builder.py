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


@dataclass
class ORBEXAProblemConfig:
    """
    Configuration for building ORBEX-A MPC problems.

    This holds all the mission-specific parameters that get translated
    into a generic MPCProblem for the solver.
    """

    # Time parameters
    t_start: float = 0.0
    dt: float = 0.001
    num_steps: int = 80

    # States and goals
    x_0: np.ndarray = field(default_factory=lambda: np.zeros(6))
    x_f: np.ndarray = field(default_factory=lambda: np.zeros(6))
    u_0: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Dynamics (can be matrices OR callables)
    A: Union[np.ndarray, Callable] = None
    B: np.ndarray = None
    Q: np.ndarray = field(default_factory=lambda: np.eye(6))
    R: np.ndarray = field(default_factory=lambda: np.eye(3) * 1e-4)
    d: Union[np.ndarray, Callable] = None  # Disturbance

    # Constraints
    state_bounds: Optional[List[Dict]] = None
    input_bounds: Optional[List[Dict]] = None

    # Orbital mechanics params (for time-varying dynamics)
    eccentricity: float = 0.0
    t_periapsis: float = 0.0
    mean_motion: float = 0.001

    # Tube MPC settings
    tube_mpc_enabled: bool = False
    tube_params: Dict[str, Any] = field(default_factory=dict)

    # Target tracking
    target_params: Optional[Dict[str, Any]] = None


def build_mpc_problem(config: ORBEXAProblemConfig) -> MPCProblem:
    """
    Build a generic MPCProblem from ORBEX-A configuration.

    Args:
        config: ORBEX-A problem configuration

    Returns:
        MPCProblem ready for any solver backend
    """
    # Handle time-varying A matrix
    if callable(config.A):
        # Evaluate at start time for initial problem setup
        # Solvers that support time-varying dynamics will use the callable
        A_eval = config.A(config.t_start, config.t_periapsis)
        if hasattr(A_eval, "__iter__") and not isinstance(A_eval, np.ndarray):
            A_eval = np.array(A_eval)
    else:
        A_eval = config.A if config.A is not None else np.zeros((6, 6))

    # Handle B matrix
    # Handle B matrix
    B_raw = config.B
    if callable(B_raw):
        # Evaluate B. Arguments ignored if constant, or use t_start if needed.
        # B_func typically takes *args, **kwargs and returns constant B.
        B = B_raw(config.t_start, config.t_periapsis)
        if not isinstance(B, np.ndarray):
            B = np.array(B)
    elif B_raw is not None:
        B = B_raw
    else:
        B = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Build extra params for solver-specific features
    extra_params = {
        "dynamics_callable": config.A if callable(config.A) else None,
        "disturbance_callable": config.d if callable(config.d) else None,
        "t_start": config.t_start,
        "t_periapsis": config.t_periapsis,
        "eccentricity": config.eccentricity,
        "mean_motion": config.mean_motion,
        "u_0": config.u_0,
    }

    # Add tube MPC if enabled
    if config.tube_mpc_enabled:
        extra_params["tube_mpc"] = config.tube_params

    # Add target params if present
    if config.target_params:
        extra_params["target_params"] = config.target_params

    return MPCProblem(
        A=A_eval,
        B=B,
        Q=config.Q,
        R=config.R,
        x_0=config.x_0,
        x_f=config.x_f,
        num_steps=config.num_steps,
        dt=config.dt,
        state_bounds=config.state_bounds,
        input_bounds=config.input_bounds,
        dynamics_type="continuous",
        extra_params=extra_params,
    )


def build_from_dynamics(
    A_func: Callable,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    d_func: Callable,
    x_0: np.ndarray,
    x_f: np.ndarray,
    t_start: float,
    num_steps: int,
    dt: float,
    t_periapsis: float = 0.0,
    eccentricity: float = 0.0,
    state_bounds: Optional[List[Dict]] = None,
    input_bounds: Optional[List[Dict]] = None,
    **kwargs,
) -> MPCProblem:
    """
    Convenience function to build MPCProblem directly from dynamics functions.

    This is the primary interface for the refactored mpc.py to use.
    """
    config = ORBEXAProblemConfig(
        t_start=t_start,
        dt=dt,
        num_steps=num_steps,
        x_0=x_0,
        x_f=x_f,
        A=A_func,
        B=B,
        Q=Q,
        R=R,
        d=d_func,
        state_bounds=state_bounds,
        input_bounds=input_bounds,
        eccentricity=eccentricity,
        t_periapsis=t_periapsis,
    )

    # Handle optional tube MPC
    if "tube_mpc" in kwargs:
        config.tube_mpc_enabled = True
        config.tube_params = kwargs["tube_mpc"]

    return build_mpc_problem(config)
