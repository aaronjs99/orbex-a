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
SciPy solver backend for MPC optimization.
"""

import time
import numpy as np
from typing import Dict, Any
from scipy import optimize as opt

from orbexa.solvers.base import SolverBase, SolverResult, MPCProblem


class ScipySolver(SolverBase):
    """
    SciPy-based MPC solver using nonlinear optimization.

    Configuration options:
        method (str): Optimization method. Default: "SLSQP"
            Options: "SLSQP", "trust-constr", "COBYLA"
        max_iter (int): Maximum iterations. Default: 1000
        ftol (float): Function tolerance. Default: 1e-9
        disp (bool): Display output. Default: False
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.method = self.config.get("method", "SLSQP")
        self.max_iter = self.config.get("max_iter", 1000)
        self.ftol = self.config.get("ftol", 1e-9)
        self.disp = self.config.get("disp", False)
        self._A_d = None
        self._B_d = None

    def setup(self, problem: MPCProblem) -> None:
        """Set up SciPy optimization."""
        self._problem = problem

        # Discretize if needed
        if problem.dynamics_type == "continuous":
            self._A_d, self._B_d = self._discretize(problem.A, problem.B, problem.dt)
        else:
            self._A_d, self._B_d = problem.A, problem.B

        self._is_setup = True

    def _objective(self, z: np.ndarray) -> float:
        """Quadratic cost function."""
        p = self._problem
        n, m, T = p.num_states, p.num_inputs, p.num_steps

        X = z[: n * T].reshape(n, T)
        U = z[n * T :].reshape(m, T)

        cost = 0.0
        for t in range(T):
            x_err = X[:, t] - p.x_f
            cost += x_err @ p.Q @ x_err
            cost += U[:, t] @ p.R @ U[:, t]
        return cost

    def _dynamics_constraint(self, z: np.ndarray) -> np.ndarray:
        """Equality constraints for discrete dynamics."""
        p = self._problem
        n, m, T = p.num_states, p.num_inputs, p.num_steps

        X = z[: n * T].reshape(n, T)
        U = z[n * T :].reshape(m, T)

        constraints = []

        # Initial condition
        constraints.extend(X[:, 0] - p.x_0)

        # Dynamics: x_{t+1} = A*x_t + B*u_t
        for t in range(T - 1):
            x_next = self._A_d @ X[:, t] + self._B_d @ U[:, t]
            constraints.extend(X[:, t + 1] - x_next)

        return np.array(constraints)

    def solve(self) -> SolverResult:
        """Solve using SciPy optimization."""
        if not self._is_setup:
            return SolverResult(
                success=False, message="Solver not set up. Call setup() first."
            )

        p = self._problem
        n, m, T = p.num_states, p.num_inputs, p.num_steps

        # Initial guess
        z0 = np.zeros(n * T + m * T)
        for t in range(T):
            z0[t * n : (t + 1) * n] = p.x_0  # Initialize states with x_0

        # Build bounds
        bounds = []
        for t in range(T):
            for i in range(n):
                lb, ub = -np.inf, np.inf
                if p.state_bounds and i < len(p.state_bounds):
                    if p.state_bounds[i].get("lower") not in ["-Inf", None]:
                        lb = p.state_bounds[i]["lower"]
                    if p.state_bounds[i].get("upper") not in ["+Inf", None]:
                        ub = p.state_bounds[i]["upper"]
                bounds.append((lb, ub))

        for t in range(T):
            for i in range(m):
                lb, ub = -np.inf, np.inf
                if p.input_bounds and i < len(p.input_bounds):
                    if p.input_bounds[i].get("lower") not in ["-Inf", None]:
                        lb = p.input_bounds[i]["lower"]
                    if p.input_bounds[i].get("upper") not in ["+Inf", None]:
                        ub = p.input_bounds[i]["upper"]
                bounds.append((lb, ub))

        constraints = {"type": "eq", "fun": self._dynamics_constraint}

        start_time = time.time()

        try:
            result = opt.minimize(
                self._objective,
                z0,
                method=self.method,
                bounds=bounds,
                constraints=constraints,
                options={
                    "maxiter": self.max_iter,
                    "ftol": self.ftol,
                    "disp": self.disp,
                },
            )

            solve_time = time.time() - start_time

            if result.success:
                X = result.x[: n * T].reshape(n, T)
                U = result.x[n * T :].reshape(m, T)

                return SolverResult(
                    success=True,
                    states=X,
                    inputs=U,
                    cost=result.fun,
                    solve_time=solve_time,
                    message="Optimization successful",
                    solver_info={
                        "nit": result.nit,
                        "nfev": result.nfev,
                    },
                )
            else:
                return SolverResult(
                    success=False,
                    solve_time=solve_time,
                    message=f"Optimization failed: {result.message}",
                    solver_info={"result": result},
                )

        except Exception as e:
            solve_time = time.time() - start_time
            return SolverResult(
                success=False,
                solve_time=solve_time,
                message=f"SciPy solver error: {str(e)}",
            )

    def cleanup(self) -> None:
        """Clean up solver state."""
        self._A_d = None
        self._B_d = None
        self._is_setup = False
