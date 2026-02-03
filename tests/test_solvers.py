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
Tests for ORBEX-A solver backends.
"""

import numpy as np
import pytest
from orbexa.solvers import (
    get_solver,
    get_solver_from_config,
    GekkoSolver,
    ScipySolver,
    SolverBase,
    MPCProblem,
    SolverResult,
)


class TestSolverBase:
    """Test abstract base class and data structures."""

    def test_mpc_problem_creation(self):
        """Test MPCProblem dataclass creation."""
        A = np.eye(2)
        B = np.array([[0], [1]])
        Q = np.eye(2)
        R = np.array([[0.1]])
        x_0 = np.zeros(2)
        x_f = np.array([1.0, 0.0])

        problem = MPCProblem(
            dynamics_matrix=A,
            input_matrix=B,
            state_cost_matrix=Q,
            input_cost_matrix=R,
            initial_state=x_0,
            final_state=x_f,
            num_steps=10,
            anom_step=0.1,
        )

        assert problem.dynamics_matrix.shape == (2, 2)
        assert problem.input_matrix.shape == (2, 1)
        assert problem.num_steps == 10

    def test_solver_result_creation(self):
        """Test SolverResult dataclass."""
        result = SolverResult(success=True, cost=1.5, solve_time=0.1)
        assert result.success
        assert result.cost == 1.5


class TestGekkoSolver:
    """Test GEKKO solver backend."""

    def test_gekko_instantiation(self):
        """Test GEKKO solver can be instantiated."""
        solver = GekkoSolver()
        assert solver.name == "GekkoSolver"
        assert not solver._is_setup

    def test_gekko_config(self):
        """Test GEKKO solver configuration."""
        config = {"remote": True, "max_iter": 1000}
        solver = GekkoSolver(config)
        assert solver.remote is True
        assert solver.max_iter == 1000


class TestScipySolver:
    """Test SciPy solver backend."""

    def test_scipy_instantiation(self):
        """Test SciPy solver can be instantiated."""
        solver = ScipySolver()
        assert solver.name == "ScipySolver"

    def test_scipy_config(self):
        """Test SciPy solver configuration."""
        config = {"method": "trust-constr", "max_iter": 500}
        solver = ScipySolver(config)
        assert solver.method == "trust-constr"
        assert solver.max_iter == 500


class TestSolverFactory:
    """Test solver factory functions."""

    def test_get_solver_gekko(self):
        """Test getting GEKKO solver."""
        solver = get_solver("gekko")
        assert isinstance(solver, GekkoSolver)

    def test_get_solver_scipy(self):
        """Test getting SciPy solver."""
        solver = get_solver("scipy")
        assert isinstance(solver, ScipySolver)

    def test_get_solver_invalid(self):
        """Test getting invalid solver raises error."""
        with pytest.raises(ValueError):
            get_solver("invalid_solver")

    def test_get_solver_from_config(self):
        """Test getting solver from config file."""
        solver = get_solver_from_config("config/default.yaml")
        # Default config uses GEKKO
        assert isinstance(solver, GekkoSolver)

    def test_get_solver_with_config(self):
        """Test getting solver with custom config."""
        config = {"max_iter": 5000, "method": "trust-constr"}
        solver = get_solver("scipy", config)
        assert solver.max_iter == 5000
        assert solver.method == "trust-constr"
