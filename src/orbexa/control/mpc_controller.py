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
from orbexa.control.mpc_problem_builder import build_from_dynamics
from orbexa.utils.anomaly import true_anomaly_to_time, dq_dt, dt_dq
from orbexa.control.dynamic_tube_model import ancillary_controller
from orbexa.estimation.adaptor import run_adaptation, run_adaptor_op

logger = logging.getLogger(__name__)


@dataclass
class MissionResult:
    """Result from a complete MPC mission."""

    success: bool
    anom_history: List[float]
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
        initial_state: np.ndarray,
        final_state: np.ndarray,
        control_input_0: np.ndarray,
        start_anom: float,
        anom_step: float,
        num_steps: int,
        dynamics: Tuple[Callable, np.ndarray, np.ndarray, np.ndarray, Callable],
        bounds: Tuple[List[Dict], List[Dict]],
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

        # Prepare additional kwargs, avoiding duplicates for Q and R
        extra_kwargs = kwargs.copy()
        extra_kwargs.pop("state_cost_matrix", None)
        extra_kwargs.pop("input_cost_matrix", None)

        # Build the MPC problem
        problem = build_from_dynamics(
            A_func=A_func,
            input_matrix=B,
            state_cost_matrix=state_cost_matrix,
            input_cost_matrix=input_cost_matrix,
            d_func=d_func,
            initial_state=initial_state,
            final_state=final_state,
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

    def run_mission(
        self,
        operation: str,
        anom_step: float,
        start_anom: float,
        num_chasers: int,
        num_mpc_steps: int,
        num_act_steps: int,
        initial_state: np.ndarray,
        target_state: Union[np.ndarray, Callable],
        control_input_0: np.ndarray,
        dynamics_func: Callable,  # Explicit dynamics factory
        dynamics_params: Dict[str, Any],  # e.g. {mean_motion: ..., ecc: ...}
        bounds: Tuple[List[Dict], List[Dict]],
        max_mission_steps: int,
        **kwargs,
    ) -> MissionResult:
        """
        Run a full mission simulation loop (True Anomaly based).

        Args:
           anom_step: True anomaly step size (rad).
        """
        # Calculate initial q based on start_time if needed, or assume q_0 is the independent var start
        eccentricity = dynamics_params.get("eccentricity", 0.0)
        mean_motion = dynamics_params.get("mean_motion", 0.001)
        time_periapsis = dynamics_params.get("time_periapsis", 0.0)

        anom = start_anom
        # eccentric_anomaly_0 and q_0 calculation block removed because we use start_anom directly
        state = initial_state.copy()

        logger.info(f"Starting mission: {operation}")
        logger.info(f"Initial state: {state}")
        logger.info(
            f"Initial Anomaly (anom): {anom:.4f} rad, Step (anom_step): {anom_step:.4f} rad"
        )

        # Determine target state
        if callable(target_state):
            # Pass independent variable (anom)
            final_state_val = target_state(anom)
        else:
            final_state_val = target_state

        # Lists to store history
        anom_history = [anom]
        state_history = [state.copy()]
        input_history = []

        solve_times = []
        success = True

        # Initial solution guess
        control_input_seed = control_input_0

        for step_idx in range(max_mission_steps):
            logger.debug(
                f"MPC Solve Step {step_idx+1}/{max_mission_steps} | Anomaly: {anom:.4f}"
            )

            # Update dynamics model for current step using anom
            matrices, _, _ = dynamics_func(q=anom, **dynamics_params)

            # Prepare solve kwargs
            solve_kwargs = kwargs.copy()
            solve_kwargs.pop("time_periapsis", None)
            solve_kwargs.pop("t_periapsis", None)
            solve_kwargs.pop("eccentricity", None)
            tube_config = solve_kwargs.pop("tube_mpc", None)
            adapt_config = solve_kwargs.pop("adaptive", None)

            result = self.solve_step(
                initial_state=state,
                final_state=final_state_val,
                control_input_0=control_input_seed,
                start_anom=anom,
                anom_step=anom_step,
                num_steps=num_mpc_steps,
                dynamics=matrices,
                bounds=bounds,
                time_periapsis=time_periapsis,
                eccentricity=eccentricity,
                use_anomaly_scaling=False,  # No explicit scaling needed, solver uses A(q)
                **solve_kwargs,
            )

            solve_times.append(result.solve_time)

            if not result.success:
                logger.warning(f"Solver failed at step {step_idx}")
                success = False
                break

            # Actuations loop
            act_count = min(num_act_steps, num_mpc_steps)

            for act_idx in range(act_count):
                # Extract input
                if isinstance(result.control_trajectory, list):
                    applied_control = np.array(
                        [u_ch[act_idx] for u_ch in result.control_trajectory]
                    )
                elif isinstance(result.control_trajectory, np.ndarray):
                    if result.control_trajectory.ndim == 2:
                        applied_control = result.control_trajectory[:, act_idx]
                    else:
                        applied_control = (
                            result.control_trajectory
                            if act_idx == 0
                            else np.zeros_like(result.control_trajectory)
                        )
                else:
                    applied_control = np.zeros(3)

                # Tube MPC
                if tube_config and tube_config.get("enabled"):
                    if (
                        hasattr(result, "state_trajectory")
                        and result.state_trajectory is not None
                    ):
                        if (
                            result.state_trajectory.ndim == 2
                            and result.state_trajectory.shape[1] > act_idx
                        ):
                            x_nom = result.state_trajectory[:, act_idx]
                            # Tube controller might need physical time if it uses continuous dynamics internally?
                            # Or we adapt it to q? Assuming we pass q.
                            tube_control = ancillary_controller(
                                t=anom,
                                t_p=time_periapsis,
                                t_f=anom + anom_step,
                                dt=anom_step,
                                mean_motion=mean_motion,
                                nom_state=x_nom,
                                act_state=state,
                                **tube_config,
                            )
                            applied_control += tube_control

                input_history.append(applied_control)

                # Propagate State (Simulated Plant)
                # matrices are A(q) etc.
                A_func, B_func, _, _, d_func = matrices
                # We assume A_func works with anom
                A_val = A_func(anom, time_periapsis)
                d_val = d_func(anom, time_periapsis)
                B_val = B_func()

                # State propagation: x_next = x + (Ax + Bu + d)*anom_step
                state_dot = A_val @ state + B_val @ applied_control + d_val
                state = state + state_dot * anom_step

                anom += anom_step

                anom_history.append(anom)
                state_history.append(state.copy())

                # Update target
                if callable(target_state):
                    final_state_val = target_state(anom)

            # Adaptation
            if adapt_config and adapt_config.get("enabled"):
                try:
                    run_adaptation(
                        W=np.array(state_history),
                        t_periapsis=time_periapsis,
                        mean_motion=mean_motion,
                        **adapt_config,
                    )
                except Exception as e:
                    logger.warning(f"Adaptation failed: {e}")

        logger.info(f"Mission {operation} completed. Success: {success}")
        return MissionResult(
            success=success,
            anom_history=anom_history,
            state_history=state_history,
            input_history=input_history,
            solver_stats={
                "total_solve_time": sum(solve_times),
                "avg_time": np.mean(solve_times) if solve_times else 0,
            },
        )
