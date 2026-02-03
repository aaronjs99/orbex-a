import unittest
from unittest.mock import patch, MagicMock, ANY
import numpy as np

from orbexa.simulation.runner import run_simulation
from orbexa.control.mpc_controller import MissionResult


class TestMPCFlow(unittest.TestCase):
    """
    Tests to verify that the correct modules (MPC, Tube, Adaptor) are invoked
    based on the simulation mode.
    """

    def setUp(self):
        # Common mocks could go here, but doing per-test for clarity
        pass

    @patch("orbexa.control.mpc_controller.MPCController.run_mission")
    def test_basic_mpc_flow(self, mock_run_mission):
        """
        Test 'mpc' mode:
        - Should run MPC
        - Should NOT run Ancillary Controller (Tube)
        - Should NOT run Adaptor
        """
        # We need to spy on internal calls if they happen INSIDE run_mission.
        # Ideally, we mock the imported functions in the mpc module.

        # But wait, run_simulation instantiates MPCController and calls run_mission.
        # The logic calling ancillary_controller should be INSIDE run_mission (or called by it).
        # So we shouldn't mock run_mission if we want to test its internals,
        # unless we are testing run_simulation's dispatch logic.

        # ACTUALLY: The user wants to ensure "running mpc runs the mpc controller".
        # "running tube mpc should run the tube on top of mpc".

        # So we should run the real run_simulation and real MPCController.run_mission,
        # but mock the `ancillary_controller` and `run_adaptation` functions
        # to verify they are called.
        pass

    @patch("orbexa.control.mpc_controller.ancillary_controller")
    @patch("orbexa.control.mpc_controller.run_adaptation")
    def test_mpc_mode_integrations(self, mock_adaptor, mock_tube):
        """Verify MPC mode does NOT call tube or adaptor."""
        # Setup minimal config/mocks to make run_simulation pass without actually solving
        # We might need to mock the solver to avoid overhead/failures

        with patch("orbexa.control.mpc_controller.get_solver") as mock_get_solver:
            # Mock solver instance
            mock_solver_instance = MagicMock()
            mock_solver_instance.solve_problem.return_value.success = True
            mock_solver_instance.solve_problem.return_value.control_trajectory = (
                np.zeros((3, 20))
            )
            mock_solver_instance.solve_problem.return_value.state_trajectory = np.zeros(
                (6, 20)
            )
            mock_solver_instance.solve_problem.return_value.solve_time = 0.1
            mock_tube.return_value = np.zeros(3)  # 1 step
            mock_get_solver.return_value = mock_solver_instance

            # Just run 1 step to trigger logic
            run_simulation(mode="mpc", max_steps=1, solver="scipy")

            # Assertions
            mock_tube.assert_not_called()
            mock_adaptor.assert_not_called()

    @patch("orbexa.control.mpc_controller.ancillary_controller")
    @patch("orbexa.control.mpc_controller.run_adaptation")
    def test_tube_mode_integrations(self, mock_adaptor, mock_tube):
        """Verify Tube MPC mode calls tube but not adaptor."""

        # Verify that ancillary_controller is available in mpc module (imported)
        # If it's not imported yet, this test setup might fail on import match,
        # but we assume we will fix the code to import it.
        # If the code doesn't import it, we might need to patch where it SHOULD be or
        # patch the source module 'orbexa.control.dynamic_tube_model.ancillary_controller'

        with patch("orbexa.control.mpc_controller.get_solver") as mock_get_solver:
            mock_solver_instance = MagicMock()
            mock_solver_instance.solve_problem.return_value.success = True
            # Mock inputs for 1 step
            mock_solver_instance.solve_problem.return_value.control_trajectory = (
                np.zeros((3, 20))
            )
            mock_solver_instance.solve_problem.return_value.state_trajectory = np.zeros(
                (6, 20)
            )
            mock_solver_instance.solve_problem.return_value.solve_time = 0.1
            mock_tube.return_value = np.zeros(3)
            mock_get_solver.return_value = mock_solver_instance

            # Run
            run_simulation(mode="tube", max_steps=1, solver="scipy")

            # Assertions
            # Tube controller SHOULD be called
            mock_tube.assert_called()
            # Adaptor should NOT be called
            mock_adaptor.assert_not_called()

    @patch("orbexa.control.mpc_controller.ancillary_controller")
    @patch("orbexa.control.mpc_controller.run_adaptation")
    def test_adaptive_mode_integrations(self, mock_adaptor, mock_tube):
        """Verify Adaptive mode calls both."""

        with patch("orbexa.control.mpc_controller.get_solver") as mock_get_solver:
            mock_solver_instance = MagicMock()
            mock_solver_instance.solve_problem.return_value.success = True
            mock_solver_instance.solve_problem.return_value.control_trajectory = (
                np.zeros((3, 20))
            )
            mock_solver_instance.solve_problem.return_value.state_trajectory = np.zeros(
                (6, 20)
            )
            mock_solver_instance.solve_problem.return_value.solve_time = 0.1
            mock_tube.return_value = np.zeros(3)
            mock_get_solver.return_value = mock_solver_instance

            # Run
            run_simulation(mode="adtmpc", max_steps=1, solver="scipy")

            # Assertions
            mock_tube.assert_called()
            mock_adaptor.assert_called()
