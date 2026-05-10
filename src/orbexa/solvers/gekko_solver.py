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
GEKKO solver backend for MPC optimization.
"""

import time
import numpy as np
import logging
from typing import Dict, Any

from gekko import GEKKO
from orbexa.solvers.base import SolverBase, SolverResult, MPCProblem
from orbexa.utils.anomaly import dt_dq
from orbexa.utils.math_utils import tait_bryan_to_rotation_matrix

logger = logging.getLogger(__name__)


class GekkoSolver(SolverBase):
    """
    GEKKO-based MPC solver.

    Configuration options:
        remote (bool): Use remote GEKKO server. Default: False
        max_iter (int): Maximum solver iterations. Default: 3000
        max_memory (int): Maximum memory in MB. Default: 512
        solver_type (int): GEKKO solver type (3=IPOPT). Default: 3
        disp (bool): Display solver output. Default: False
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.remote = self.config.get("remote", False)
        self.max_iter = self.config.get("max_iter", 3000)
        self.max_memory = self.config.get("max_memory", 512)
        self.solver_type = self.config.get("solver_type", 3)
        self.disp = self.config.get("disp", False)
        self._model = None
        self._states = None
        self._inputs = None

    def setup(self, problem: MPCProblem) -> None:
        """
        Set up GEKKO optimization model.

        Handles:
        - Linear Time-Invariant (LTI) systems
        - Linear Time-Varying (LTV) systems (via dynamics_callable)
        - Tube MPC constraint tightening
        """
        self._problem = problem
        logger.debug(
            f"Setting up GEKKO model for {problem.num_states} states, {problem.num_inputs} inputs"
        )

        # Unpack extra params
        extra = problem.extra_params
        start_anom = extra.get("start_anom", 0.0)
        time_periapsis = extra.get("time_periapsis", 0.0)
        eccentricity = extra.get("eccentricity", 0.0)
        mean_motion = extra.get("mean_motion", 0.0)
        target_params = extra.get("target_params") or {}

        # Tube MPC params
        tube_mpc = extra.get("tube_mpc")  # Dict or None
        if tube_mpc:
            # Tube MPC constraint tightening (if enabled)
            if tube_mpc and tube_mpc.get("enabled", False):
                logger.debug("Tube MPC enabled - bounds should be pre-tightened")

        solver = GEKKO(remote=self.remote)
        horizon_end = max(problem.num_steps - 1, 0) * problem.anom_step
        solver.time = np.linspace(0, horizon_end, problem.num_steps)

        num_states = problem.num_states
        num_inputs = problem.num_inputs

        # --- Parameters ---
        # Optimization weight
        w_param = solver.Param(value=np.ones(problem.num_steps))

        # Final Step Indicator
        final_array = np.zeros(problem.num_steps)
        final_array[-1] = 1
        final_param = solver.Param(value=final_array)

        # --- Variables ---
        # State variables
        self._states = [
            solver.Var(value=problem.initial_state[i], fixed_initial=True)
            for i in range(num_states)
        ]

        # Input variables
        # Input variables
        self._inputs = [
            solver.Var(value=0, fixed_initial=False) for i in range(num_inputs)
        ]
        # Initialize inputs with guess if provided
        if "u_0" in extra and len(extra["u_0"]) == num_inputs:
            for i in range(num_inputs):
                self._inputs[i].value = extra["u_0"][i]

        # --- Bounds ---
        if problem.state_bounds:
            for i, bound in enumerate(problem.state_bounds[:num_states]):
                if bound.get("lower") not in ["-Inf", None, float("-inf")]:
                    self._states[i].lower = bound["lower"]
                if bound.get("upper") not in ["+Inf", None, float("inf")]:
                    self._states[i].upper = bound["upper"]

        if problem.input_bounds:
            for i, bound in enumerate(problem.input_bounds[:num_inputs]):
                if bound.get("lower") not in ["-Inf", None, float("-inf")]:
                    self._inputs[i].lower = bound["lower"]
                if bound.get("upper") not in ["+Inf", None, float("inf")]:
                    self._inputs[i].upper = bound["upper"]

        # --- Dynamics ---
        # Independent variable is anomaly (solver.time)
        q_start = extra.get("q_0", 0.0)
        q_var = solver.Param(value=q_start + solver.time)

        # Time variable for LTV dynamics
        t_var = solver.Var(value=start_anom)

        # dTime/dAnomaly scaling
        use_scaling = extra.get("use_anomaly_scaling", False)
        if use_scaling:
            dt_dq_expr = dt_dq(
                q_var,
                eccentricity=eccentricity,
                mean_motion=mean_motion,
                t_periapsis=time_periapsis,
                solver=solver,
            )
            solver.Equation(t_var.dt() == dt_dq_expr)
        else:
            dt_dq_expr = 1.0
            solver.Equation(t_var.dt() == 1.0)

        eqs = []

        # Paper-level obstacle constraints. GEKKO applies these vectorized over
        # the horizon because each Var contains a trajectory over solver.time.
        operation = target_params.get("operation")
        if operation == "rendezvous":
            radius = float(
                target_params.get(
                    "rendezvous_radius", target_params.get("target_radius", 0.0)
                )
            )
            tube_radius = float(target_params.get("tube_radius", 0.0))
            if radius > 0.0:
                safe_radius = radius + tube_radius
                eqs.append(
                    self._states[0] ** 2
                    + self._states[1] ** 2
                    + self._states[2] ** 2
                    >= safe_radius**2
                )
        elif operation == "docking" and target_params.get("shape") == "cylinder":
            radius = float(target_params.get("target_radius", 0.0))
            half_length = float(target_params.get("target_half_length", 0.0))
            tube_radius = float(target_params.get("tube_radius", 0.0))
            orientation = np.asarray(target_params.get("orientation", np.zeros(3)))
            rotation = tait_bryan_to_rotation_matrix(orientation)
            body_from_lvlh = rotation.T

            p_body = []
            for i in range(3):
                p_body.append(
                    sum(body_from_lvlh[i, j] * self._states[j] for j in range(3))
                )

            radial_margin = (
                p_body[0] ** 2 + p_body[1] ** 2 - (radius + tube_radius) ** 2
            )
            axial_margin = solver.abs2(p_body[2]) - (half_length + tube_radius)
            eqs.append(solver.max2(radial_margin, axial_margin) >= 0.0)

        pairwise_constraints = extra.get("pairwise_constraints") or []
        for constraint in pairwise_constraints:
            min_separation = float(constraint.get("min_separation", 0.0))
            reference_positions = np.asarray(
                constraint.get("reference_positions", []), dtype=float
            )
            if min_separation <= 0.0 or reference_positions.size == 0:
                continue
            if reference_positions.ndim == 1:
                reference_positions = np.repeat(
                    reference_positions.reshape(1, 3),
                    problem.num_steps,
                    axis=0,
                )
            if reference_positions.shape[0] != problem.num_steps:
                reference_positions = np.resize(
                    reference_positions, (problem.num_steps, 3)
                )
            ref_x = solver.Param(value=reference_positions[:, 0])
            ref_y = solver.Param(value=reference_positions[:, 1])
            ref_z = solver.Param(value=reference_positions[:, 2])
            eqs.append(
                (self._states[0] - ref_x) ** 2
                + (self._states[1] - ref_y) ** 2
                + (self._states[2] - ref_z) ** 2
                >= min_separation**2
            )

        # Check if we have time-varying dynamics (callable A)
        if callable(problem.extra_params.get("dynamics_callable")):
            A_func = problem.extra_params["dynamics_callable"]
            # B and d might also be callables, or constant
            # For this implementation, we assume A is the main driver of complexity

            # Anomaly calculation. In the paper demo the independent variable
            # is true anomaly directly unless explicit time/anomaly scaling is
            # requested.
            if use_scaling:
                # Use current anomaly variable directly
                q_val = q_var
            else:
                q_val = q_var

            # Evaluate dynamics matrices symbolically using either t or q.
            # orbital_ellp_undrag and similar factories support the 'solver'
            # and 'q' keywords.
            try:
                A_mat_expr = A_func(t_var, time_periapsis, solver=solver, q=q_val)
            except TypeError:
                # Fallback for simple callables
                A_mat_expr = A_func(t_var)

            # D func
            d_func = problem.extra_params.get("disturbance_callable")
            d_vec_expr = np.zeros(num_states)
            if callable(d_func):
                d_vec_expr = d_func(t_var, time_periapsis, solver=solver, q=q_val)
            elif hasattr(problem, "d") and isinstance(problem.d, np.ndarray):
                d_vec_expr = problem.d

            # X_dot = A(t)X + B(t)u + d(t)
            # Assuming constant B for now
            B_mat = problem.input_matrix

            for i in range(num_states):
                # Manual matrix multiplication A*state
                dot_A_state = 0
                for j in range(num_states):
                    val = A_mat_expr[i][j]
                    if isinstance(val, (int, float)) and val == 0:
                        continue
                    dot_A_state += val * self._states[j]

                # Manual matrix multiplication B*control_input
                dot_B_control_input = 0
                for k in range(num_inputs):
                    val = B_mat[i][k]
                    if val == 0:
                        continue
                    dot_B_control_input += val * self._inputs[k]

                # state_dot_q = (A*state + B*u + d) * dt_dq
                eqs.append(
                    self._states[i].dt()
                    == (dot_A_state + dot_B_control_input + d_vec_expr[i]) * dt_dq_expr
                )

        else:
            # Continuous LTI (Original Logic)
            # d_state = A * state + B * control_input
            for i in range(num_states):
                d_state = sum(
                    problem.dynamics_matrix[i, j] * self._states[j]
                    for j in range(num_states)
                )
                d_state += sum(
                    problem.input_matrix[i, j] * self._inputs[j]
                    for j in range(num_inputs)
                )
                eqs.append(self._states[i].dt() == d_state)

        # --- Objective ---
        # Quadratic cost: (x-xf)'Q(x-xf) + u'Ru
        cost_terms = []

        # Iterate over diagonal of Q/R for efficiency if diagonal
        # Or full matrix mult

        # State cost
        for i in range(num_states):
            for j in range(num_states):
                if problem.state_cost_matrix[i, j] != 0:
                    delta_state_i = self._states[i] - problem.final_state[i]
                    delta_state_j = self._states[j] - problem.final_state[j]
                    cost_terms.append(
                        delta_state_i * problem.state_cost_matrix[i, j] * delta_state_j
                    )

        # Input cost
        for i in range(num_inputs):
            for j in range(num_inputs):
                if problem.input_cost_matrix[i, j] != 0:
                    cost_terms.append(
                        self._inputs[i]
                        * problem.input_cost_matrix[i, j]
                        * self._inputs[j]
                    )

        total_cost = solver.Intermediate(sum(cost_terms))

        solver.Equations(eqs)
        solver.Minimize(w_param * total_cost)

        # Solver settings
        solver.options.IMODE = 6  # MPC mode
        solver.options.SOLVER = self.solver_type
        solver.options.MAX_ITER = self.max_iter
        solver.options.MAX_MEMORY = self.max_memory
        solver.options.OTOL = 1e-6
        solver.options.RTOL = 1e-6

        self._model = solver
        self._is_setup = True

    def solve(self) -> SolverResult:
        """Solve the GEKKO optimization problem."""
        if not self._is_setup:
            return SolverResult(
                success=False, message="Solver not set up. Call setup() first."
            )

        start_time = time.time()

        try:
            self._model.solve(disp=self.disp)
            solve_time = time.time() - start_time

            # Extract results
            states = np.array([state_i.value for state_i in self._states])
            inputs = np.array([input_i.value for input_i in self._inputs])
            cost = self._model.options.objfcnval

            return SolverResult(
                success=True,
                state_trajectory=states,
                control_trajectory=inputs,
                cost=cost,
                solve_time=solve_time,
                message="Optimization successful",
                solver_info={
                    "status": self._model.options.APPSTATUS,
                    "iterations": self._model.options.ITERATIONS,
                },
            )
        except Exception as e:
            solve_time = time.time() - start_time
            return SolverResult(
                success=False,
                solve_time=solve_time,
                message=f"GEKKO solver failed: {str(e)}",
            )

    def cleanup(self) -> None:
        """Clean up GEKKO model."""
        if self._model:
            self._model.cleanup()
            self._model = None
        self._states = None
        self._inputs = None
        self._is_setup = False
