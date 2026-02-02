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
            A_d, B_d = self._discretize(problem.A, problem.B, problem.dt)
        else:
            A_d, B_d = problem.A, problem.B

        n = problem.num_states
        m = problem.num_inputs
        T = problem.num_steps

        opti = casadi.Opti()

        # Decision variables
        self._X = opti.variable(n, T)
        self._U = opti.variable(m, T)

        # Parameters for matrices
        A_p = opti.parameter(n, n)
        B_p = opti.parameter(n, m)
        Q_p = opti.parameter(n, n)
        R_p = opti.parameter(m, m)
        x0_p = opti.parameter(n)
        xf_p = opti.parameter(n)

        opti.set_value(A_p, A_d)
        opti.set_value(B_p, B_d)
        opti.set_value(Q_p, problem.Q)
        opti.set_value(R_p, problem.R)
        opti.set_value(x0_p, problem.x_0)
        opti.set_value(xf_p, problem.x_f)

        # Objective: sum of quadratic costs
        cost = 0
        for t in range(T):
            x_err = self._X[:, t] - xf_p
            cost += casadi.mtimes(casadi.mtimes(x_err.T, Q_p), x_err)
            cost += casadi.mtimes(casadi.mtimes(self._U[:, t].T, R_p), self._U[:, t])
        opti.minimize(cost)

        # Dynamics constraints
        for t in range(T - 1):
            opti.subject_to(
                self._X[:, t + 1]
                == casadi.mtimes(A_p, self._X[:, t]) + casadi.mtimes(B_p, self._U[:, t])
            )

        # Initial condition
        opti.subject_to(self._X[:, 0] == x0_p)

        # State bounds
        if problem.state_bounds:
            for i, bound in enumerate(problem.state_bounds[:n]):
                if bound.get("lower") not in ["-Inf", None]:
                    opti.subject_to(self._X[i, :] >= bound["lower"])
                if bound.get("upper") not in ["+Inf", None]:
                    opti.subject_to(self._X[i, :] <= bound["upper"])

        # Input bounds
        if problem.input_bounds:
            for i, bound in enumerate(problem.input_bounds[:m]):
                if bound.get("lower") not in ["-Inf", None]:
                    opti.subject_to(self._U[i, :] >= bound["lower"])
                if bound.get("upper") not in ["+Inf", None]:
                    opti.subject_to(self._U[i, :] <= bound["upper"])

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

            states = np.array(sol.value(self._X))
            inputs = np.array(sol.value(self._U))
            cost = float(sol.value(self._opti.f))

            return SolverResult(
                success=True,
                states=states,
                inputs=inputs,
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
        self._opti = None
        self._X = None
        self._U = None
        self._is_setup = False
