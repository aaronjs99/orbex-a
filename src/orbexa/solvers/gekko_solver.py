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
        """Set up GEKKO optimization model."""
        self._problem = problem

        # Discretize if needed
        if problem.dynamics_type == "continuous":
            A_d, B_d = self._discretize(problem.A, problem.B, problem.dt)
        else:
            A_d, B_d = problem.A, problem.B

        m = GEKKO(remote=self.remote)
        m.time = np.linspace(0, problem.num_steps * problem.dt, problem.num_steps)

        n = problem.num_states
        p = problem.num_inputs

        # State variables
        self._x = [m.Var(value=problem.x_0[i], fixed_initial=True) for i in range(n)]

        # Input variables
        self._u = [m.Var(value=0, fixed_initial=False) for i in range(p)]

        # Apply bounds
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

        # Dynamics constraints (continuous form, GEKKO handles integration)
        eqs = []
        for i in range(n):
            dx = sum(problem.A[i, j] * self._x[j] for j in range(n))
            dx += sum(problem.B[i, j] * self._u[j] for j in range(p))
            eqs.append(self._x[i].dt() == dx)

        # Objective: quadratic cost
        cost = 0
        for i in range(n):
            cost += problem.Q[i, i] * (self._x[i] - problem.x_f[i]) ** 2
        for i in range(p):
            cost += problem.R[i, i] * self._u[i] ** 2

        m.Equations(eqs)
        m.Minimize(cost)

        # Solver settings
        m.options.IMODE = 6  # MPC mode
        m.options.SOLVER = self.solver_type
        m.options.MAX_ITER = self.max_iter
        m.options.MAX_MEMORY = self.max_memory
        m.options.OTOL = 1e-7
        m.options.RTOL = 1e-7

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
