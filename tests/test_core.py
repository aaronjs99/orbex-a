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
Tests for core package imports and basic functionality.
"""

import pytest
import numpy as np


class TestPackageImports:
    """Test that all package modules can be imported."""

    def test_import_orbexa(self):
        """Test main package import."""
        import orbexa

        assert hasattr(orbexa, "__version__")
        assert orbexa.__version__ == "2.0.0"

    def test_import_params(self):
        """Test params module import."""
        from orbexa import params

        assert hasattr(params, "dt")
        assert hasattr(params, "n")
        assert hasattr(params, "actOrbitParams")

    def test_import_dynamics(self):
        """Test dynamics module import."""
        from orbexa import dynamics

        assert hasattr(dynamics, "orbital_ellp_undrag")
        assert hasattr(dynamics, "cwhEquations")
        assert hasattr(dynamics, "orbitalParams")

    def test_import_spacecraft(self):
        """Test spacecraft module import."""
        from orbexa.core.spacecraft import Spacecraft, Target, Chaser

        assert Spacecraft is not None
        assert Target is not None
        assert Chaser is not None

    def test_import_utils(self):
        """Test utils module import."""
        from orbexa import utils

        assert hasattr(utils, "discretize")
        assert hasattr(utils, "load_config")
        assert hasattr(utils, "genSkewSymMat")

    def test_import_solvers(self):
        """Test solvers module import."""
        from orbexa.solvers import get_solver, GekkoSolver, ScipySolver

        assert get_solver is not None
        assert GekkoSolver is not None
        assert ScipySolver is not None


class TestParams:
    """Test parameters module."""

    def test_dt_positive(self):
        """Test that dt is positive."""
        from orbexa import params

        assert params.dt > 0

    def test_orbit_params(self):
        """Test orbital parameters structure."""
        from orbexa import params

        assert "eccentricity" in params.actOrbitParams
        assert "drag_alpha" in params.actOrbitParams
        assert "drag_beta" in params.actOrbitParams


class TestDynamics:
    """Test dynamics functions."""

    def test_cwh_equations(self):
        """Test CWH equations return correct structure."""
        from orbexa.core.dynamics import cwhEquations

        matrices, constraints, bounds = cwhEquations(dt=0.1)
        A, B, Q, R, d = matrices
        assert isinstance(A, np.ndarray) or callable(A)

    def test_orbital_ellp_undrag(self):
        """Test elliptical orbit dynamics."""
        from orbexa.core.dynamics import orbital_ellp_undrag

        matrices, constraints, bounds = orbital_ellp_undrag(dt=0.1, eccentricity=0.1)
        A, B, Q, R, d = matrices
        assert callable(A)


class TestUtils:
    """Test utility functions."""

    def test_discretize(self):
        """Test state-space discretization."""
        from orbexa.utils import discretize

        A = np.array([[0, 1], [-1, 0]])
        B = np.array([[0], [1]])
        A_d, B_d = discretize(0.1, A, B)
        assert A_d.shape == (2, 2)
        assert B_d.shape == (2, 1)

    def test_gen_skew_sym_mat(self):
        """Test skew-symmetric matrix generation."""
        from orbexa.utils import genSkewSymMat

        v = [1, 2, 3]
        S = genSkewSymMat(v)
        assert S.shape == (3, 3)
        # Skew-symmetric: S = -S^T
        np.testing.assert_array_almost_equal(S, -S.T)

    def test_load_config(self):
        """Test configuration loading."""
        from orbexa.utils import load_config

        config = load_config("config/default.yaml")
        assert "sim" in config
        assert "solver" in config


class TestSolvers:
    """Test solver backends."""

    def test_get_solver_gekko(self):
        """Test GEKKO solver instantiation."""
        from orbexa.solvers import get_solver

        solver = get_solver("gekko")
        assert solver is not None

    def test_get_solver_scipy(self):
        """Test SciPy solver instantiation."""
        from orbexa.solvers import get_solver

        solver = get_solver("scipy")
        assert solver is not None

    def test_invalid_solver(self):
        """Test that invalid solver raises error."""
        from orbexa.solvers import get_solver

        with pytest.raises(ValueError):
            get_solver("invalid_solver")


class TestSpacecraft:
    """Test spacecraft classes."""

    def test_spacecraft_init(self):
        """Test Spacecraft initialization."""
        from orbexa.core.spacecraft import Spacecraft

        sc = Spacecraft()
        assert sc.numStates == 6

    def test_target_init(self):
        """Test Target initialization."""
        from orbexa.core.spacecraft import Target

        target = Target(
            {"initState": np.zeros(6)},
            {
                "angularVelocity": np.array([0.1, 0.0, 0.0]),
                "momInertia": np.eye(3) * 100,
            },
        )
        assert hasattr(target, "angularVelocity")
        assert hasattr(target, "momInertia")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
