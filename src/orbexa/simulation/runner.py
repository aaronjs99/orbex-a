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
Simulation Runner

Core simulation execution logic, separated from CLI.
"""

import numpy as np
import logging
from typing import Dict, Optional

from orbexa.control import MPCController, MissionResult
from orbexa.core.config import SimulationConfig
from orbexa.core.dynamics import orbital_ellp_undrag
from orbexa.simulation.modes import CONTROL_MODES, get_mode_config
from orbexa.utils.io_utils import save_data, create_filename
from dataclasses import asdict
from pathlib import Path

logger = logging.getLogger(__name__)


def run_simulation(
    mode: str,
    max_steps: int = 5,
    solver: str = "gekko",
    config: Optional[SimulationConfig] = None,
) -> MissionResult:
    """
    Run simulation with specified control mode.

    Args:
        mode: Control mode ("oc", "mpc", "tube", "adtmpc")
        max_steps: Maximum MPC iterations
        solver: Solver backend ("gekko", "scipy", "casadi")
        config: Simulation configuration object.

    Returns:
        MissionResult with trajectory data
    """
    mode_config = get_mode_config(mode)

    if config is None:
        config = SimulationConfig.load()

    # Set random seed
    np.random.seed(config.seed)

    logger.info("=" * 60)
    logger.info(f"  {mode_config.name}")
    logger.info(f"  {mode_config.description}")
    logger.info("=" * 60)

    logger.debug(f"Initializing MPC controller with solver: {solver}")

    controller = MPCController(solver_backend=solver)

    # Initial conditions from config or hardcoded for sample task
    # initial_state = np.array([-50.0, 10.0, 5.0, 0.05, -0.01, 0.01])
    initial_state = config.orbit.initial_conditions
    target_state = np.zeros(6)
    control_input_0 = np.zeros(3)

    # OC uses single iteration
    effective_steps = 1 if mode == "oc" else max_steps

    # Prepare dynamics params
    dynamics_params = {
        "anom_step": config.anom_step,
        "mean_motion": config.orbit.mean_motion,  # derived property
        "eccentricity": config.orbit.eccentricity,
        # "alpha": ? "beta": ? if using drag
    }

    result = controller.run_mission(
        operation="rendezvous",
        anom_step=config.anom_step,  # Use anomaly step anom_step
        start_anom=0.0,
        num_chasers=1,
        num_mpc_steps=mode_config.num_mpc_steps,
        num_act_steps=mode_config.num_act_steps,
        initial_state=initial_state,
        target_state=target_state,
        control_input_0=control_input_0,
        dynamics_func=orbital_ellp_undrag,
        dynamics_params=dynamics_params,
        bounds=(config.mpc.state_bounds, config.mpc.input_bounds),
        max_mission_steps=effective_steps,
        tube_mpc=(
            {
                "enabled": mode_config.tube_mpc_enabled,
                "lambda_gain": config.tube.sliding_gains,
                "alpha": config.tube.bandwidth_0,
                "phi": config.tube.boundary_layer_0,
                "eccentricity_range": (
                    config.tube.bandwidth_range["lower"],
                    config.tube.bandwidth_range["upper"],
                ),
            }
            if mode_config.tube_mpc_enabled
            else None
        ),
        adaptive=(
            {
                "enabled": mode_config.adaptive_enabled,
                "init_params": {
                    "eccentricity": [0.0, 0.0],
                    "alpha": [config.tube.bandwidth_0[0], config.tube.bandwidth_0[0]],
                    "beta": [0.0, 0.0],
                },
                "range_params": {
                    "dt": config.anom_step,
                    "data_range": 50,
                    "adaptation_range": 10,
                },
                "u_t": [np.zeros(50).tolist(), np.zeros(50).tolist(), np.zeros(50).tolist()],
                "D": 0.01,
            }
            if mode_config.adaptive_enabled
            else None
        ),
        # Extra explicit kwargs for solver
        time_periapsis=config.orbit.time_periapsis,
        state_cost_matrix=config.mpc.Q,
        input_cost_matrix=config.mpc.R,
    )

    logger.info(f"Result: {'SUCCESS' if result.success else 'FAILED'}")
    logger.info(f"Steps completed: {len(result.anom_history)}")
    logger.info(f"Final state: {result.state_history[-1]}")
    logger.info(f"Total solve time: {result.solver_stats['total_solve_time']:.3f}s")

    # Save mission data to gitignored data/ folder
    data_dir = Path("data") / mode
    data_path = create_filename(data_dir, ".json")

    # Convert MissionResult to dict and save
    mission_data = asdict(result)
    mission_data["mode"] = mode

    if save_data(data_path, mission_data):
        logger.info(f"Mission data saved to: {data_path}")

    return result


def run_all_modes(
    max_steps: int = 5, solver: str = "gekko"
) -> Dict[str, Optional[MissionResult]]:
    """Run all control modes and compare."""
    results = {}

    logger.info("=" * 60)
    logger.info("  ORBEX-A Control Mode Comparison")
    logger.info("=" * 60)

    # Load shared config
    config = SimulationConfig.load()

    for mode in CONTROL_MODES:
        try:
            results[mode] = run_simulation(mode, max_steps, solver, config=config)
        except Exception as exception:
            logger.error(f"  [{mode.upper()}] FAILED: {exception}")
            logger.debug("Traceback:", exc_info=True)
            results[mode] = None

    _print_summary(results)
    return results


def _print_summary(results: Dict[str, Optional[MissionResult]]) -> None:
    """Print comparison summary table."""
    logger.info("=" * 60)
    logger.info("  Summary")
    logger.info("=" * 60)
    logger.info(f"{'Mode':<12} {'Status':<10} {'Steps':<8} {'Time (s)':<10}")
    logger.info("-" * 40)

    for mode, result in results.items():
        if result:
            status = "OK" if result.success else "FAIL"
            steps = len(result.anom_history)
            time_s = result.solver_stats["total_solve_time"]
            logger.info(f"{mode:<12} {status:<10} {steps:<8} {time_s:<10.3f}")
        else:
            logger.info(f"{mode:<12} {'ERROR':<10} {'-':<8} {'-':<10}")
