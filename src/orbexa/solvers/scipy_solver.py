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
        self._d_d = None

    def setup(self, problem: MPCProblem) -> None:
        """Set up SciPy optimization."""
        self._problem = problem

        A = problem.dynamics_matrix
        if callable(A):
            start_anom = problem.extra_params.get("start_anom", 0.0)
            time_periapsis = problem.extra_params.get("time_periapsis", 0.0)
            A = np.asarray(A(start_anom, time_periapsis), dtype=float)

        B = problem.input_matrix
        if callable(B):
            B = np.asarray(B(), dtype=float)

        d_vec = np.zeros(problem.num_states)
        d_callable = problem.extra_params.get("disturbance_callable")
        if callable(d_callable):
            start_anom = problem.extra_params.get("start_anom", 0.0)
            time_periapsis = problem.extra_params.get("time_periapsis", 0.0)
            d_vec = np.asarray(d_callable(start_anom, time_periapsis), dtype=float)
        elif problem.extra_params.get("disturbance_vector") is not None:
            d_vec = np.asarray(problem.extra_params["disturbance_vector"], dtype=float)

        # Discretize if needed
        if problem.dynamics_type == "continuous":
            self._A_d, self._B_d = self._discretize(
                np.asarray(A, dtype=float), np.asarray(B, dtype=float), problem.anom_step
            )
            self._d_d = d_vec * problem.anom_step
        else:
            self._A_d, self._B_d = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
            self._d_d = d_vec

        self._is_setup = True

    def _rollout_from_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """Propagate the linearized discrete dynamics for an input sequence."""
        p = self._problem
        num_states, num_steps = p.num_states, p.num_steps
        states = np.zeros((num_states, num_steps), dtype=float)
        states[:, 0] = p.initial_state
        for t in range(num_steps - 1):
            states[:, t + 1] = (
                self._A_d @ states[:, t]
                + self._B_d @ inputs[:, t]
                + self._d_d
            )
        return states

    def _objective_inputs(self, z: np.ndarray) -> float:
        """Quadratic cost over a dynamics-consistent rollout."""
        p = self._problem
        num_inputs, num_steps = p.num_inputs, p.num_steps
        inputs = z.reshape(num_inputs, num_steps)
        states = self._rollout_from_inputs(inputs)

        cost = 0.0
        for t in range(num_steps):
            state_err = states[:, t] - p.final_state
            cost += state_err @ p.state_cost_matrix @ state_err
            cost += inputs[:, t] @ p.input_cost_matrix @ inputs[:, t]
        return cost

    def _affine_constraints_inputs(self, z: np.ndarray) -> np.ndarray:
        """Linearized safety constraints for secondary SciPy artifacts."""
        p = self._problem
        constraints = p.extra_params.get("affine_constraints", [])
        if not constraints:
            return np.ones(1)

        inputs = z.reshape(p.num_inputs, p.num_steps)
        states = self._rollout_from_inputs(inputs)
        margins = []
        num_steps = p.num_steps

        for constraint in constraints:
            if hasattr(constraint, "normal"):
                normal = np.asarray(constraint.normal, dtype=float)
                offset = float(constraint.offset)
            else:
                normal = np.asarray(constraint["normal"], dtype=float)
                offset = float(constraint["offset"])
            for t in range(num_steps):
                state = states[:, t]
                vector = state[: len(normal)]
                margins.append(float(normal @ vector + offset))

        return np.asarray(margins, dtype=float)

    def _state_bound_constraints_inputs(self, z: np.ndarray) -> np.ndarray:
        """Return positive margins for finite state bounds."""
        p = self._problem
        if not p.state_bounds:
            return np.ones(1)
        inputs = z.reshape(p.num_inputs, p.num_steps)
        states = self._rollout_from_inputs(inputs)
        margins = []
        for state_idx, bound in enumerate(p.state_bounds[: p.num_states]):
            lower = bound.get("lower")
            upper = bound.get("upper")
            if lower not in ["-Inf", None, float("-inf")]:
                margins.extend(states[state_idx, :] - float(lower))
            if upper not in ["+Inf", None, float("inf")]:
                margins.extend(float(upper) - states[state_idx, :])
        return np.asarray(margins or [1.0], dtype=float)

    def solve(self) -> SolverResult:
        """Solve using SciPy optimization."""
        if not self._is_setup:
            return SolverResult(
                success=False, message="Solver not set up. Call setup() first."
            )

        p = self._problem
        num_inputs, num_steps = p.num_inputs, p.num_steps

        z0 = np.zeros(num_inputs * num_steps)
        seed = p.extra_params.get("u_0")
        if seed is not None and len(seed) == num_inputs:
            z0[:] = np.repeat(
                np.asarray(seed, dtype=float)[:, None], num_steps, axis=1
            ).ravel()

        # Build input bounds. State evolution is eliminated by propagation.
        bounds = []
        for i in range(num_inputs):
            for _ in range(num_steps):
                lb, ub = -np.inf, np.inf
                if p.input_bounds and i < len(p.input_bounds):
                    if p.input_bounds[i].get("lower") not in ["-Inf", None]:
                        lb = p.input_bounds[i]["lower"]
                    if p.input_bounds[i].get("upper") not in ["+Inf", None]:
                        ub = p.input_bounds[i]["upper"]
                bounds.append((lb, ub))

        constraints = []
        if p.extra_params.get("affine_constraints"):
            constraints.append({"type": "ineq", "fun": self._affine_constraints_inputs})
        if p.state_bounds:
            constraints.append({"type": "ineq", "fun": self._state_bound_constraints_inputs})

        start_time = time.time()

        try:
            result = opt.minimize(
                self._objective_inputs,
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
                inputs = result.x.reshape(num_inputs, num_steps)
                states = self._rollout_from_inputs(inputs)

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
        self._d_d = None
        self._is_setup = False
