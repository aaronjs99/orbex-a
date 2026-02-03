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
CasADi solver backend for MPC optimization.

Note: Requires casadi package. Install with: pip install casadi
"""

import time
import numpy as np
from typing import Dict, Any

try:
    import casadi

    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False
    casadi = None

from orbexa.solvers.base import SolverBase, SolverResult, MPCProblem, SolverError


class CasadiSolver(SolverBase):
    """
    CasADi-based MPC solver using direct collocation or shooting.

    Configuration options:
        nlp_solver (str): NLP solver. Default: "ipopt"
            Options: "ipopt", "sqpmethod", "scpgen"
        max_iter (int): Maximum iterations. Default: 3000
        print_level (int): IPOPT print level (0-12). Default: 0
        linear_solver (str): Linear solver. Default: "mumps"
            Options: "mumps", "ma27", "ma57", "pardiso"
        warm_start (bool): Use warm start. Default: False
    """

    def __init__(self, config: Dict[str, Any] = None):
        if not CASADI_AVAILABLE:
            raise ImportError("CasADi not installed. Run: pip install casadi")

        super().__init__(config)
        self.nlp_solver = self.config.get("nlp_solver", "ipopt")
        self.max_iter = self.config.get("max_iter", 3000)
        self.print_level = self.config.get("print_level", 0)
        self.linear_solver = self.config.get("linear_solver", "mumps")
        self.warm_start = self.config.get("warm_start", False)

        self._opti = None
        self._X = None
        self._U = None

    def setup(self, problem: MPCProblem) -> None:
        """Set up CasADi Opti problem."""
        self._problem = problem

        # Discretize if needed
        if problem.dynamics_type == "continuous":
            A_d, B_d = self._discretize(
                problem.dynamics_matrix, problem.input_matrix, problem.anom_step
            )
        else:
            A_d, B_d = problem.dynamics_matrix, problem.input_matrix

        num_states = problem.num_states
        num_inputs = problem.num_inputs
        num_steps = problem.num_steps

        opti = casadi.Opti()

        # Decision variables
        self._states = opti.variable(num_states, num_steps)
        self._controls = opti.variable(num_inputs, num_steps)

        # Parameters for matrices
        A_p = opti.parameter(num_states, num_states)
        B_p = opti.parameter(num_states, num_inputs)
        Q_p = opti.parameter(num_states, num_states)
        R_p = opti.parameter(num_inputs, num_inputs)
        x0_p = opti.parameter(num_states)
        xf_p = opti.parameter(num_states)

        opti.set_value(A_p, A_d)
        opti.set_value(B_p, B_d)
        opti.set_value(A_p, A_d)
        opti.set_value(B_p, B_d)
        opti.set_value(Q_p, problem.state_cost_matrix)
        opti.set_value(R_p, problem.input_cost_matrix)
        opti.set_value(x0_p, problem.initial_state)
        opti.set_value(xf_p, problem.final_state)

        # Objective: sum of quadratic costs
        cost = 0
        for t in range(num_steps):
            x_err = self._states[:, t] - xf_p
            cost += casadi.mtimes(casadi.mtimes(x_err.T, Q_p), x_err)
            cost += casadi.mtimes(
                casadi.mtimes(self._controls[:, t].T, R_p), self._controls[:, t]
            )
        opti.minimize(cost)

        # Dynamics constraints
        for t in range(num_steps - 1):
            opti.subject_to(
                self._states[:, t + 1]
                == casadi.mtimes(A_p, self._states[:, t])
                + casadi.mtimes(B_p, self._controls[:, t])
            )

        # Initial condition
        # Initial condition
        opti.subject_to(self._states[:, 0] == x0_p)

        # State bounds
        if problem.state_bounds:
            for i, bound in enumerate(problem.state_bounds[:num_states]):
                if bound.get("lower") not in ["-Inf", None]:
                    opti.subject_to(self._states[i, :] >= bound["lower"])
                if bound.get("upper") not in ["+Inf", None]:
                    opti.subject_to(self._states[i, :] <= bound["upper"])

        # Input bounds
        if problem.input_bounds:
            for i, bound in enumerate(problem.input_bounds[:num_inputs]):
                if bound.get("lower") not in ["-Inf", None]:
                    opti.subject_to(self._controls[i, :] >= bound["lower"])
                if bound.get("upper") not in ["+Inf", None]:
                    opti.subject_to(self._controls[i, :] <= bound["upper"])

        # Solver options
        opts = {
            "ipopt.max_iter": self.max_iter,
            "ipopt.print_level": self.print_level,
            "ipopt.linear_solver": self.linear_solver,
            "print_time": 0,
        }

        if self.warm_start:
            opts["ipopt.warm_start_init_point"] = "yes"

        opti.solver(self.nlp_solver, opts)

        self._opti = opti
        self._is_setup = True

    def solve(self) -> SolverResult:
        """Solve the CasADi optimization problem."""
        if not self._is_setup:
            return SolverResult(
                success=False, message="Solver not set up. Call setup() first."
            )

        start_time = time.time()

        try:
            sol = self._opti.solve()
            solve_time = time.time() - start_time

            states = np.array(sol.value(self._states))
            inputs = np.array(sol.value(self._controls))
            cost = float(sol.value(self._opti.f))

            return SolverResult(
                success=True,
                state_trajectory=states,
                control_trajectory=inputs,
                cost=cost,
                solve_time=solve_time,
                message="Optimization successful",
                solver_info={
                    "solver_stats": sol.stats(),
                },
            )

        except Exception as e:
            solve_time = time.time() - start_time
            return SolverResult(
                success=False,
                solve_time=solve_time,
                message=f"CasADi solver failed: {str(e)}",
            )

    def cleanup(self) -> None:
        """Clean up CasADi problem."""

    def cleanup(self) -> None:
        """Clean up CasADi problem."""
        self._opti = None
        self._states = None
        self._controls = None
        self._is_setup = False
