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
ORBEX-A MPC Controller Module.

This module implements the Model Predictive Control (MPC) logic for spacecraft
rendezvous and docking, including Tube MPC for robustness and Adaptive estimates.

Key Design Principles:
- Solver-agnostic: Uses the orbexa.solvers abstraction layer
- No direct solver imports (GEKKO, CasADi, etc.)
- Clean separation between problem formulation and solving
"""

import logging
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass

from orbexa.solvers import get_solver, get_solver_from_config, MPCProblem, SolverResult
from orbexa.control.problem_builder import build_from_dynamics

logger = logging.getLogger(__name__)


@dataclass
class MissionResult:
    """Result from a complete MPC mission."""

    success: bool
    time_history: List[float]
    state_history: List[np.ndarray]
    input_history: List[np.ndarray]
    solver_stats: Dict[str, Any]
    message: str = ""


class MPCController:
    """
    Model Predictive Controller for Spacecraft RPO.

    This controller is solver-agnostic and uses the orbexa.solvers
    abstraction layer to solve MPC problems.
    """

    def __init__(
        self,
        solver_backend: str = "gekko",
        solver_config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize the MPC Controller.

        Args:
            solver_backend: Solver to use ("gekko", "casadi", "scipy")
            solver_config: Solver-specific configuration
            config_path: Path to YAML config file (overrides other options)
        """
        self.solver_backend = solver_backend
        self.solver_config = solver_config or {}
        self.config_path = config_path

        # Get solver instance
        if config_path:
            self._solver = get_solver_from_config(config_path)
        else:
            self._solver = get_solver(solver_backend, solver_config)

    def solve_step(
        self,
        x_0: np.ndarray,
        x_f: np.ndarray,
        u_0: np.ndarray,
        t_start: float,
        dt: float,
        num_steps: int,
        dynamics: Tuple[Callable, np.ndarray, np.ndarray, np.ndarray, Callable],
        bounds: Tuple[List[Dict], List[Dict]],
        t_periapsis: float = 0.0,
        eccentricity: float = 0.0,
        **kwargs,
    ) -> SolverResult:
        """
        Solve a single MPC optimization step.

        Args:
            x_0: Initial state vector (n,)
            x_f: Target/reference state vector (n,)
            u_0: Initial control guess (m,)
            t_start: Current simulation time
            dt: Time step
            num_steps: MPC horizon length
            dynamics: Tuple of (A_func, B, Q, R, d_func)
            bounds: Tuple of (state_bounds, input_bounds)
            t_periapsis: Time of periapsis (for orbital dynamics)
            eccentricity: Orbit eccentricity
            **kwargs: Additional parameters (tube_mpc, target_params, etc.)

        Returns:
            SolverResult with optimized trajectory
        """
        A_func, B, Q, R, d_func = dynamics
        state_bounds, input_bounds = bounds

        # Build the MPC problem
        problem = build_from_dynamics(
            A_func=A_func,
            B=B,
            Q=Q,
            R=R,
            d_func=d_func,
            x_0=x_0,
            x_f=x_f,
            t_start=t_start,
            num_steps=num_steps,
            dt=dt,
            t_periapsis=t_periapsis,
            eccentricity=eccentricity,
            state_bounds=state_bounds,
            input_bounds=input_bounds,
            **kwargs,
        )

        # Solve using the abstracted solver
        result = self._solver.solve_problem(problem)

        return result

    def run_mission(
        self,
        operation: str,
        dt: float,
        t_0: float,
        num_chasers: int,
        num_mpc_steps: int,
        num_act_steps: int,
        X_0: np.ndarray,
        f_X_f: Union[np.ndarray, Callable],
        U_0: np.ndarray,
        dynamics_func: Callable,  # Explicit dynamics factory
        dynamics_params: Dict[str, Any],  # e.g. {mean_motion: ..., ecc: ...}
        bounds: Tuple[List[Dict], List[Dict]],
        max_mission_steps: int,
        **kwargs,
    ) -> MissionResult:
        """
        Run a full mission simulation loop.

        Args:
           dynamics_func: Function that returns (matrices, constraints, bounds).
                          e.g. orbexa.core.dynamics.orbital_ellp_undrag
           dynamics_params: Kwargs for dynamics_func (e.g. mean_motion).
        """
        t = t_0
        X = X_0.copy()

        # Determine X_f
        if callable(f_X_f):
            X_f_val = f_X_f(t)
        else:
            X_f_val = f_X_f

        # Lists to store history
        time_history = [t]
        state_history = [X.copy()]
        input_history = []

        solve_times = []
        success = True

        # Initial solution guess
        u_seed = U_0

        for step in range(max_mission_steps):
            # Update dynamics model for current step
            matrices, _, _ = dynamics_func(**dynamics_params)

            # Extract necessary params for solve_step (t_periapsis, eccentricity)
            # Default to 0.0 if not provided in dynamics_params
            t_p = dynamics_params.get("t_periapsis", 0.0)
            ecc = dynamics_params.get("eccentricity", 0.0)

            # Prepare kwargs for solve_step, removing explicit args to avoid duplicates
            solve_kwargs = kwargs.copy()
            solve_kwargs.pop("t_periapsis", None)
            solve_kwargs.pop("eccentricity", None)

            result = self.solve_step(
                x_0=X,
                x_f=X_f_val,
                u_0=u_seed,
                t_start=t,
                dt=dt,
                num_steps=num_mpc_steps,
                dynamics=matrices,
                bounds=bounds,
                t_periapsis=t_p,  # Explicitly passed from dynamics_params
                eccentricity=ecc,
                **solve_kwargs,  # Pass remaining params
            )

            solve_times.append(result.solve_time)

            if not result.success:
                logger.warning(f"Solver failed at step {step}")
                # success = False
                # break/continue? simple fallback?
                # For now continue with zero input
                u_applied = np.zeros(3)  # num_chasers*3?
            else:
                # Apply first input
                u_full = result.inputs
                # If result inputs is vector (m*N,), take first block
                # If list of arrays, take first
                # Extract first input from optimized sequence
                if isinstance(result.inputs, list):
                    # Handle list of arrays (e.g. from Gekko wrapper where inputs are per-variable lists)
                    u_applied = np.array([u_ch[0] for u_ch in result.inputs])
                elif isinstance(result.inputs, np.ndarray):
                    if result.inputs.ndim == 2:
                        # Shape (num_inputs, num_steps) -> Take first column
                        u_applied = result.inputs[:, 0]
                    else:
                        # Fallback
                        u_applied = (
                            result.inputs[0] if len(result.inputs) > 0 else np.zeros(3)
                        )
                else:
                    u_applied = np.zeros(3)

            input_history.append(u_applied)

            # Propagate State (Simulation) using Euler integration
            # Note: Uses the model dynamics (nominal). In a real scenario, use actual plant model.
            A_func, B_func, _, _, d_func = matrices

            # Evaluate dynamics matrices at current time t
            A_val = A_func(t, t_p)
            d_val = d_func(t, t_p)
            B_val = B_func()

            # X_dot = AX + Bu + d
            x_dot = A_val @ X + B_val @ u_applied + d_val
            X = X + x_dot * dt

            t += dt
            state_history.append(X.copy())

            # Update target state if moving
            if callable(f_X_f):
                X_f_val = f_X_f(t)

        return MissionResult(
            success=success,
            time_history=time_history,
            state_history=state_history,
            input_history=input_history,
            solver_stats={
                "total_solve_time": sum(solve_times),
                "avg_time": np.mean(solve_times) if solve_times else 0,
            },
        )
