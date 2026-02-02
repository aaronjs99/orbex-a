import pytest
import numpy as np
from orbexa.control.mpc import MPCController, mpc
from orbexa.core.dynamics import orbital_ellp_undrag


@pytest.fixture
def mpc_config():
    return {}


@pytest.fixture
def dynamics_matrices():
    # Helper to generate matrices
    dt = 1.0
    matrices, _, _ = orbital_ellp_undrag(dt=dt, eccentricity=0.0)
    # returns ((A, B, Q, R, d), constraints, bounds)
    return matrices  # (A, B, Q, R, d)


class TestMPCController:
    def test_init(self, mpc_config):
        controller = MPCController(mpc_config)
        assert controller is not None
        assert controller.config == mpc_config

    def test_solve_step_structure(self, mpc_config, dynamics_matrices):
        controller = MPCController(mpc_config)

        # Setup dummy inputs
        t_s = 0.0
        num_mpc_steps = 5
        num_act_steps = 1
        dt = 1.0

        time_params = {
            "t_s": t_s,
            "timeSeq": np.linspace(0, num_mpc_steps * dt, num_mpc_steps),
            "numMPCSteps": num_mpc_steps,
            "numActSteps": num_act_steps,
        }

        # Identity matrices for simplicity if dynamics_matrices fails
        A, B, Q, R, d = dynamics_matrices
        # Ensure A is callable if it's supposed to be?
        # orbital_ellp_undrag returns functions for A, d.

        # Mock bounds
        bounds = (
            [{"upper": "+Inf", "lower": "-Inf"}] * 6,
            [{"upper": "+Inf", "lower": "-Inf"}] * 3,
        )

        solver_params = {
            "remote": False,  # Use local generic solve if possible
            "disp": False,
            "X_0": [0.0] * 6,
            "U_0": [0.0] * 3,
            "X_f": [0.0] * 6,
        }

        # We need mock matrices that Gekko can handle or pass real ones?
        # A_nom is a function A(t, t_p, m).

        # This test might fail if GEKKO is not installed or local solve fails.
        # But we check if it RUNS, return code might be 1 (fail) but not crash.

        try:
            status, xn, xa, un, ua, targets = controller.solve_step(
                time_params,
                nom_matrices=(A, B, Q, R, d),
                act_matrices=(A, B, d),
                bounds=bounds,
                solver_params=solver_params,
                num_chasers=1,
            )
            # Check basic return structure
            assert isinstance(status, int)
        except ImportError:
            pytest.skip("Gekko not installed")
        except Exception as e:
            # If local solver executable missing, it raises exception usually
            if "Executable not found" in str(e):
                pytest.warns(UserWarning, match="Gekko local solver not found")
            else:
                raise e

    def test_legacy_mpc_alias(self):
        # Just check signature, don't run full mission (too heavy)
        assert callable(mpc)
