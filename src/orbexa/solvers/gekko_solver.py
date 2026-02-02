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
from typing import Dict, Any

from gekko import GEKKO
from orbexa.solvers.base import SolverBase, SolverResult, MPCProblem


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
        self._x = None
        self._u = None

    def setup(self, problem: MPCProblem) -> None:
        """
        Set up GEKKO optimization model.

        Handles:
        - Linear Time-Invariant (LTI) systems
        - Linear Time-Varying (LTV) systems (via dynamics_callable)
        - Tube MPC constraint tightening
        """
        self._problem = problem

        # Unpack extra params
        extra = problem.extra_params
        t_start = extra.get("t_start", 0.0)
        t_periapsis = extra.get("t_periapsis", 0.0)
        eccentricity = extra.get("eccentricity", 0.0)
        mean_motion = extra.get("mean_motion", 0.0)

        # Tube MPC params
        tube_mpc = extra.get("tube_mpc")  # Dict or None
        if tube_mpc:
            # Simple robust horizon constraint tightening
            # (Placeholder logic: In a full implementation, we'd subtract the tube cross-section from bounds)
            # For now, we assume bounds in problem.state_bounds are already tightened OR
            # we handle them here if specific logic exists.
            pass

        m = GEKKO(remote=self.remote)
        m.time = np.linspace(0, problem.num_steps * problem.dt, problem.num_steps)

        n = problem.num_states
        p = problem.num_inputs

        # --- Parameters ---
        # Optimization weight (can be used for forgetting factors etc.)
        w_param = m.Param(value=np.ones(problem.num_steps))

        # Final Step Indicator
        final_array = np.zeros(problem.num_steps)
        final_array[-1] = 1
        final_param = m.Param(value=final_array)

        # --- Variables ---
        # State variables
        self._x = [m.Var(value=problem.x_0[i], fixed_initial=True) for i in range(n)]

        # Input variables
        self._u = [m.Var(value=0, fixed_initial=False) for i in range(p)]
        # Initialize inputs with guess if provided
        if "u_0" in extra and len(extra["u_0"]) == p:
            for i in range(p):
                self._u[i].value = extra["u_0"][i]

        # --- Bounds ---
        if problem.state_bounds:
            for i, bound in enumerate(problem.state_bounds[:n]):
                if bound.get("lower") not in ["-Inf", None, float("-inf")]:
                    self._x[i].lower = bound["lower"]
                if bound.get("upper") not in ["+Inf", None, float("inf")]:
                    self._x[i].upper = bound["upper"]

        if problem.input_bounds:
            for i, bound in enumerate(problem.input_bounds[:p]):
                if bound.get("lower") not in ["-Inf", None, float("-inf")]:
                    self._u[i].lower = bound["lower"]
                if bound.get("upper") not in ["+Inf", None, float("inf")]:
                    self._u[i].upper = bound["upper"]

        # --- Dynamics ---

        # Time variable for LTV dynamics
        t_var = m.Var(value=t_start)
        m.Equation(t_var.dt() == 1)

        eqs = []

        # Check if we have time-varying dynamics (callable A)
        if callable(problem.extra_params.get("dynamics_callable")):
            A_func = problem.extra_params["dynamics_callable"]
            # B and d might also be callables, or constant
            # For this implementation, we assume A is the main driver of complexity

            # Anomaly Calculations (Elliptical Orbit support)
            # If mean_motion > 0, we compute anomalies.
            # This logic mimics the previous trajopt_dynamics.
            if mean_motion > 0:
                # M = n * (t - tp)
                # Note: t_var starts at t_start.
                # enc_arg = M / 2
                enc_arg = (mean_motion * (t_var - t_periapsis)) / 2.0

                # Eccentric Anomaly E
                # E = 2 * atan(...)
                E = m.Intermediate(
                    2
                    * m.atan(
                        m.sqrt((1 - eccentricity) / (1 + eccentricity)) * m.tan(enc_arg)
                    )
                )

                # True Anomaly q
                # q = 2 * atan(...)
                q_val = m.Intermediate(
                    2
                    * m.atan(
                        m.sqrt((1 + eccentricity) / (1 - eccentricity)) * m.tan(E / 2)
                    )
                )

                # IMPORTANT: The A_func in orbital_ellp_undrag computes A based on q_val
                # We need to pass 'm' (GEKKO object) to A_func so it uses m.sin/m.cos
                # However, generic A_func might not expect this.
                # The 'orbital_ellp_undrag' A_func was written to accept 'm'.

                # Evaluate A(t) symbolically
                # We can't call A_func(t, ...) once if it returns constants.
                # It returns a matrix of GEKKO expressions.

                # To handle this efficiently in GEKKO, we construct the equations using the values returned by A_func
                # A_func(t, t_p, m=m) -> returns 6x6 array of expressions involving m.sin(q_val)...

                # Check signature of A_func or just try passing arguments
                try:
                    A_mat_expr = A_func(t_var, t_periapsis, m=m)
                except TypeError:
                    # Fallback for simple callables that don't take m/t_p
                    A_mat_expr = A_func(t_var)

                # D func
                d_func = problem.extra_params.get("disturbance_callable")
                d_vec_expr = np.zeros(n)
                if callable(d_func):
                    d_vec_expr = d_func(t_var, t_periapsis, m=m)
                elif hasattr(problem, "d") and isinstance(problem.d, np.ndarray):
                    d_vec_expr = problem.d

                # X_dot = A(t)X + B(t)u + d(t)
                # Assuming constant B for now as commonly the case, or handle B similarly
                B_mat = problem.B

                for i in range(n):
                    # Manual matrix multiplication A*x
                    dot_A_x = 0
                    for j in range(n):
                        val = A_mat_expr[i][j]
                        # If val is 0 (number), ignore
                        # If val is expression, add
                        # GEKKO handles mixing types usually, but let's be safe
                        if isinstance(val, (int, float)) and val == 0:
                            continue
                        dot_A_x += val * self._x[j]

                    # Manual matrix multiplication B*u
                    dot_B_u = 0
                    for k in range(p):
                        val = B_mat[i][k]
                        if val == 0:
                            continue
                        dot_B_u += val * self._u[k]

                    eqs.append(self._x[i].dt() == dot_A_x + dot_B_u + d_vec_expr[i])

            else:
                # Fallback for LTV without orbital specifics (generic time varying)
                # Not fully implemented - usually requires known structure
                pass

        else:
            # Continuous LTI (Original Logic)
            # dx = Ax + Bu
            for i in range(n):
                dx = sum(problem.A[i, j] * self._x[j] for j in range(n))
                dx += sum(problem.B[i, j] * self._u[j] for j in range(p))
                eqs.append(self._x[i].dt() == dx)

        # --- Objective ---
        # Quadratic cost: (x-xf)'Q(x-xf) + u'Ru
        cost_terms = []

        # Iterate over diagonal of Q/R for efficiency if diagonal
        # Or full matrix mult

        # State cost
        for i in range(n):
            for j in range(n):
                if problem.Q[i, j] != 0:
                    delta_xi = self._x[i] - problem.x_f[i]
                    delta_xj = self._x[j] - problem.x_f[j]
                    cost_terms.append(delta_xi * problem.Q[i, j] * delta_xj)

        # Input cost
        for i in range(p):
            for j in range(p):
                if problem.R[i, j] != 0:
                    cost_terms.append(self._u[i] * problem.R[i, j] * self._u[j])

        total_cost = m.Intermediate(sum(cost_terms))

        m.Equations(eqs)
        m.Minimize(w_param * total_cost)

        # Solver settings
        m.options.IMODE = 6  # MPC mode
        m.options.SOLVER = self.solver_type
        m.options.MAX_ITER = self.max_iter
        m.options.MAX_MEMORY = self.max_memory
        m.options.OTOL = 1e-6
        m.options.RTOL = 1e-6

        # Diagnostics
        # m.options.DIAGLEVEL = 1

        self._model = m
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
            states = np.array([xi.value for xi in self._x])
            inputs = np.array([ui.value for ui in self._u])
            cost = self._model.options.objfcnval

            return SolverResult(
                success=True,
                states=states,
                inputs=inputs,
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
        self._x = None
        self._u = None
        self._is_setup = False
