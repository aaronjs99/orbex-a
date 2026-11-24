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
rendezvous and docking, with optional parameter-estimate adaptation.
Coupled robust missions are orchestrated by ADTMPCMissionRunner.

Key Design Principles:
- Solver-agnostic: Uses the orbexa.solvers abstraction layer
- No direct solver imports (GEKKO, CasADi, etc.)
- Clean separation between problem formulation and solving
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from orbexa.control.mpc.problem import build_from_dynamics
from orbexa.solvers import SolverResult, get_solver, get_solver_from_config

logger = logging.getLogger(__name__)


class MPCController:
    """
    Model Predictive Controller for Spacecraft RPO.

    This controller is solver-agnostic and uses the orbexa.solvers
    abstraction layer to solve MPC problems.
    """

    def __init__(
        self,
        solver_backend: str = "gekko",
        solver_config: dict[str, Any] | None = None,
        config_path: str | None = None,
        proposal_provider=None,
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
        if proposal_provider is not None:
            from orbexa.solvers.proposal_seed import ProposalSeededSolver

            self._solver = ProposalSeededSolver(self._solver, proposal_provider)

    def solve_step(
        self,
        initial_state: np.ndarray,
        final_state: np.ndarray,
        control_input_0: np.ndarray,
        start_anom: float,
        anom_step: float,
        num_steps: int,
        dynamics: tuple[Callable, np.ndarray, np.ndarray, np.ndarray, Callable],
        bounds: tuple[list[dict], list[dict]],
        time_periapsis: float = 0.0,
        eccentricity: float = 0.0,
        **kwargs,
    ) -> SolverResult:
        """
        Solve a single MPC optimization step.

        Args:
            initial_state: Initial state vector (n,)
            final_state: Target/reference state vector (n,)
            control_input_0: Initial control guess (m,)
            start_anom: Current simulation anomaly
            anom_step: Anomaly step size
            num_steps: MPC horizon length
            dynamics: Tuple of (A_func, B, Q, R, d_func)
            bounds: Tuple of (state_bounds, input_bounds)
            time_periapsis: Time of periapsis (for orbital dynamics)
            eccentricity: Orbit eccentricity
            **kwargs: Additional parameters (tube_mpc, target_params, etc.)

        Returns:
            SolverResult with optimized trajectory
        """
        A_func, B, state_cost_matrix, input_cost_matrix, d_func = dynamics
        state_bounds, input_bounds = bounds
        state_cost_matrix = kwargs.get("state_cost_matrix", state_cost_matrix)
        input_cost_matrix = kwargs.get("input_cost_matrix", input_cost_matrix)

        # Prepare additional kwargs, avoiding duplicates for Q and R
        extra_kwargs = kwargs.copy()
        extra_kwargs.pop("state_cost_matrix", None)
        extra_kwargs.pop("input_cost_matrix", None)
        extra_kwargs.pop("control_0", None)

        # Build the MPC problem
        problem = build_from_dynamics(
            A_func=A_func,
            input_matrix=B,
            state_cost_matrix=state_cost_matrix,
            input_cost_matrix=input_cost_matrix,
            d_func=d_func,
            initial_state=initial_state,
            final_state=final_state,
            control_0=control_input_0,
            start_anom=start_anom,
            num_steps=num_steps,
            anom_step=anom_step,
            time_periapsis=time_periapsis,
            eccentricity=eccentricity,
            state_bounds=state_bounds,
            input_bounds=input_bounds,
            **extra_kwargs,
        )

        # Solve using the abstracted solver
        result = self._solver.solve_problem(problem)

        return result
