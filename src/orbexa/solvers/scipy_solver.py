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
            self._A_d, self._B_d = self._discretize(
                problem.dynamics_matrix, problem.input_matrix, problem.anom_step
            )
        else:
            self._A_d, self._B_d = problem.dynamics_matrix, problem.input_matrix

        self._is_setup = True

    def _objective(self, z: np.ndarray) -> float:
        """Quadratic cost function."""
        p = self._problem
        num_states, num_inputs, num_steps = p.num_states, p.num_inputs, p.num_steps

        states = z[: num_states * num_steps].reshape(num_states, num_steps)
        inputs = z[num_states * num_steps :].reshape(num_inputs, num_steps)

        cost = 0.0
        for t in range(num_steps):
            state_err = states[:, t] - p.final_state
            cost += state_err @ p.state_cost_matrix @ state_err
            cost += inputs[:, t] @ p.input_cost_matrix @ inputs[:, t]
        return cost

    def _dynamics_constraint(self, z: np.ndarray) -> np.ndarray:
        """Equality constraints for discrete dynamics."""
        p = self._problem
        num_states, num_inputs, num_steps = p.num_states, p.num_inputs, p.num_steps

        states = z[: num_states * num_steps].reshape(num_states, num_steps)
        inputs = z[num_states * num_steps :].reshape(num_inputs, num_steps)

        constraints = []

        # Initial condition
        constraints.extend(states[:, 0] - p.initial_state)

        # Dynamics: state_{t+1} = A*state_t + B*input_t
        for t in range(num_steps - 1):
            state_next = self._A_d @ states[:, t] + self._B_d @ inputs[:, t]
            constraints.extend(states[:, t + 1] - state_next)

        return np.array(constraints)

    def solve(self) -> SolverResult:
        """Solve using SciPy optimization."""
        if not self._is_setup:
            return SolverResult(
                success=False, message="Solver not set up. Call setup() first."
            )

        p = self._problem
        num_states, num_inputs, num_steps = p.num_states, p.num_inputs, p.num_steps

        # Initial guess
        z0 = np.zeros(num_states * num_steps + num_inputs * num_steps)
        for t in range(num_steps):
            z0[t * num_states : (t + 1) * num_states] = (
                p.initial_state
            )  # Initialize states with x_0

        # Build bounds
        bounds = []
        for t in range(num_steps):
            for i in range(num_states):
                lb, ub = -np.inf, np.inf
                if p.state_bounds and i < len(p.state_bounds):
                    if p.state_bounds[i].get("lower") not in ["-Inf", None]:
                        lb = p.state_bounds[i]["lower"]
                    if p.state_bounds[i].get("upper") not in ["+Inf", None]:
                        ub = p.state_bounds[i]["upper"]
                bounds.append((lb, ub))

        for t in range(num_steps):
            for i in range(num_inputs):
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
                states = result.x[: num_states * num_steps].reshape(
                    num_states, num_steps
                )
                inputs = result.x[num_states * num_steps :].reshape(
                    num_inputs, num_steps
                )

                return SolverResult(
                    success=True,
                    state_trajectory=states,
                    control_trajectory=inputs,
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
