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
from orbexa.utils.anomaly import true_anomaly_to_time, dtheta_dt, dt_dtheta

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
        state_0: np.ndarray,
        state_f: np.ndarray,
        control_input_0: np.ndarray,
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
            state_0: Initial state vector (n,)
            state_f: Target/reference state vector (n,)
            control_input_0: Initial control guess (m,)
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

        # Prepare additional kwargs, avoiding duplicates for Q and R
        extra_kwargs = kwargs.copy()
        extra_kwargs.pop("Q", None)
        extra_kwargs.pop("R", None)

        # Build the MPC problem
        problem = build_from_dynamics(
            A_func=A_func,
            B=B,
            Q=Q,
            R=R,
            d_func=d_func,
            x_0=state_0,
            x_f=state_f,
            t_start=t_start,
            num_steps=num_steps,
            dt=dt,
            t_periapsis=t_periapsis,
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
        dt: float,
        t_0: float,
        num_chasers: int,
        num_mpc_steps: int,
        num_act_steps: int,
        state_0: np.ndarray,
        f_state_f: Union[np.ndarray, Callable],
        control_input_0: np.ndarray,
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
        state = state_0.copy()

        # Determine initial true anomaly q from t
        # (Assuming t_0 is time and we need q to start anomaly-based propagation)
        ecc = dynamics_params.get("eccentricity", 0.0)
        n_motion = dynamics_params.get("mean_motion", 0.001)
        t_p = dynamics_params.get("t_periapsis", 0.0)

        # We need a q variable to track anomaly. dt is dq.
        # But wait, t_start in A_func is time.
        # If we are in OC mode, we want to reach the goal in anomaly space.

        # Convert initial velocity m/s -> m/rad
        # dq/dt at t_0
        # Simplest: M = n*(t - tp). Then solve Kepler for E, then q.
        mean_anomaly_0 = n_motion * (t - t_p)
        eccentric_anomaly_0 = mean_anomaly_0  # Initial guess
        for _ in range(5):
            eccentric_anomaly_0 = mean_anomaly_0 + ecc * np.sin(eccentric_anomaly_0)
        q_0 = 2 * np.arctan(
            np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(eccentric_anomaly_0 / 2)
        )

        q = q_0
        dq = dt  # renamed for clarity inside the loop

        # State remains in [m, m/s].
        # We don't convert velocities to m/rad because we will scale the dynamics instead.

        logger.info(f"Starting mission: {operation}")
        logger.info(f"Initial state: {state}")
        logger.debug(f"MPC Steps: {num_mpc_steps}, Actuation Steps: {num_act_steps}")

        # Determine target state
        if callable(f_state_f):
            state_f_val = f_state_f(t)
        else:
            state_f_val = f_state_f

        # Lists to store history
        time_history = [t]
        state_history = [state.copy()]
        input_history = []

        solve_times = []
        success = True

        # Initial solution guess
        control_input_seed = control_input_0

        for step_idx in range(max_mission_steps):
            logger.debug(
                f"MPC Solve Step {step_idx+1}/{max_mission_steps} | Time: {t:.2f}s"
            )
            # Update dynamics model for current step
            matrices, _, _ = dynamics_func(**dynamics_params)

            # Extract necessary params for solve_step
            t_p = dynamics_params.get("t_periapsis", 0.0)
            ecc = dynamics_params.get("eccentricity", 0.0)

            # Prepare solve kwargs to avoid duplicate arguments
            solve_kwargs = kwargs.copy()
            solve_kwargs.pop("t_periapsis", None)
            solve_kwargs.pop("eccentricity", None)

            # Wrapper for dynamics to convert time-base matrices to anomaly-base
            # dX/dq = (dX/dt) * (dt/dq)
            def anomaly_dynamics_wrapper(t_in, tp_in, **dw_kwargs):
                # Get current dq/dt to find scaling 1/(dq/dt)
                solver_obj = dw_kwargs.get("solver", None)
                dqdt_val = dtheta_dt(
                    q,
                    eccentricity=ecc,
                    mean_motion=n_motion,
                    t_periapsis=tp_in,
                    solver=solver_obj,
                )
                dtdq_val = 1.0 / dqdt_val

                A_t, B_t, Q_t, R_t, d_t = matrices
                return (A_t, B_t, Q_t, R_t, d_t), dtdq_val

            # Since resolve needs the wrapped matrices, we build a special problem
            # But wait, solve_step expects the matrices tuple directly.
            # We will pass the time-matrices to solve_step, but we must ensure
            # the SOLVER knows to scale them.

            # Actually, the cleanest way is for solve_step to build a problem
            # where A_q = A_t * dtdq.

            # Let's modify solve_step to accept an optional dtdq scaling or handle it in the dynamics.

            result = self.solve_step(
                state_0=state,
                state_f=state_f_val,
                control_input_0=control_input_seed,
                t_start=t,
                dt=dq,  # anomaly step
                num_steps=num_mpc_steps,
                dynamics=matrices,  # Time-base matrices
                bounds=bounds,
                t_periapsis=t_p,
                eccentricity=ecc,
                use_anomaly_scaling=True,  # New flag
                **solve_kwargs,
            )

            solve_times.append(result.solve_time)

            if not result.success:
                logger.warning(f"Solver failed at step {step_idx}")
                success = False
                break

            # Actuations loop: Apply multiple steps from the optimized trajectory
            # num_act_steps should be <= num_mpc_steps
            act_count = min(num_act_steps, num_mpc_steps)

            for act_idx in range(act_count):
                # Extract input at current actuation index
                if isinstance(result.inputs, list):
                    # Gekko style: list of per-variable arrays
                    u_applied = np.array([u_ch[act_idx] for u_ch in result.inputs])
                elif isinstance(result.inputs, np.ndarray):
                    if result.inputs.ndim == 2:
                        # Shape (num_inputs, num_steps)
                        u_applied = result.inputs[:, act_idx]
                    else:
                        # Fallback for single step results
                        u_applied = (
                            result.inputs
                            if act_idx == 0
                            else np.zeros_like(result.inputs)
                        )
                else:
                    u_applied = np.zeros(3)

                input_history.append(u_applied)

                # Propagate State (Simulated Plant)
                A_func, B_func, _, _, d_func = matrices
                A_val = A_func(t, t_p)
                d_val = d_func(t, t_p)
                B_val = B_func()

                # Propagate State (Simulated Plant)
                # dX/dq = (dX/dt) * (dt/dq)
                A_func, B_func, _, _, d_func = matrices
                A_val = A_func(t, t_p)
                d_val = d_func(t, t_p)
                B_val = B_func()

                # Get dt/dq for the plant too
                dqdt = dtheta_dt(
                    q, eccentricity=ecc, mean_motion=n_motion, t_periapsis=t_p
                )
                dtdq = 1.0 / dqdt

                # state_dot_q = (A*state + B*u + d) * dtdq
                state_dot_q = (A_val @ state + B_val @ u_applied + d_val) * dtdq
                state = state + state_dot_q * dq

                # Real time elapsed: dt_time = dq * (dt/dq)
                dt_time = dq * dtdq

                t += dt_time
                q += dq

                # Log progress
                state_history.append(state.copy())
                time_history.append(t)

                # Update target state if moving/dynamic
                if callable(f_state_f):
                    state_f_val = f_state_f(t)

        logger.info(f"Mission {operation} completed. Success: {success}")
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
