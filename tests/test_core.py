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
from orbexa.core.config import SimulationConfig


class TestPackageImports:
    """Test that all package modules can be imported."""

    def test_import_orbexa(self):
        """Test main package import."""
        import orbexa

        assert hasattr(orbexa, "__version__")
        assert orbexa.__version__ == "2.0.0"

    def test_import_config(self):
        """Test new config module import."""
        from orbexa.core import config

        assert hasattr(config, "SimulationConfig")

        cfg = config.SimulationConfig.load()
        assert cfg.anom_step > 0
        assert cfg.orbit.mean_motion > 0

    def test_import_dynamics(self):
        """Test dynamics module import."""
        from orbexa import dynamics

        assert hasattr(dynamics, "orbital_ellp_undrag")
        assert hasattr(dynamics, "cwh_equations")
        assert hasattr(dynamics, "orbital_params")
        assert hasattr(dynamics, "orbital_circ_undrag")
        assert hasattr(dynamics, "triple_integrator")

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
        assert hasattr(utils, "gen_skew_sym_mat")

    def test_import_solvers(self):
        """Test solvers module import."""
        from orbexa.solvers import get_solver, GekkoSolver, ScipySolver

        assert get_solver is not None
        assert GekkoSolver is not None
        assert ScipySolver is not None


class TestConfig:
    """Test configuration module."""

    def test_load_default(self):
        config = SimulationConfig.load()
        assert config.num_update_steps > 0

    def test_dt_positive(self):
        config = SimulationConfig.load()
        assert config.anom_step > 0


class TestDynamics:
    """Test dynamics functions."""

    def test_cwh_equations(self):
        """Test CWH equations return correct structure."""
        from orbexa.core.dynamics import cwh_equations

        # Must pass mean_motion etc.
        matrices, constraints, bounds = cwh_equations(anom_step=0.1, mean_motion=0.01)
        A, B, Q, R, d = matrices
        assert isinstance(A, np.ndarray) or callable(A)

    def test_orbital_ellp_undrag(self):
        """Test elliptical orbit dynamics."""
        from orbexa.core.dynamics import orbital_ellp_undrag

        matrices, constraints, bounds = orbital_ellp_undrag(
            anom_step=0.1, mean_motion=0.01, eccentricity=0.1
        )
        A, B, Q, R, d = matrices
        assert callable(A)


class TestUtils:
    """Test utility functions."""

    def test_discretize(self):
        """Test state-space discretization."""
        from orbexa.utils import discretize

        A = np.array([[0, 1], [-1, 0]])
        B = np.array([[0], [1]])
        A_d, B_d, _, _ = discretize(0.1, A, B)
        assert A_d.shape == (2, 2)
        assert B_d.shape == (2, 1)

    def test_gen_skew_sym_mat(self):
        """Test skew-symmetric matrix generation."""
        from orbexa.utils import gen_skew_sym_mat

        v = [1, 2, 3]
        S = gen_skew_sym_mat(v)
        assert S.shape == (3, 3)
        np.testing.assert_array_almost_equal(S, -S.T)


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
        """Test Spacecraft initialization with config."""
        from orbexa.core.spacecraft import Spacecraft

        config = SimulationConfig.load()
        sc = Spacecraft(config)
        assert sc.num_states == 6

    def test_target_init(self):
        """Test Target initialization."""
        from orbexa.core.spacecraft import Target

        config = SimulationConfig.load()
        target = Target(config)

        assert hasattr(target, "angular_velocity")
        assert hasattr(target, "mom_inertia")

    def test_chaser_init(self):
        """Test Chaser initialization."""
        from orbexa.core.spacecraft import Chaser

        config = SimulationConfig.load()
        chaser = Chaser(config)
        assert chaser.mean_motion == config.orbit.mean_motion
