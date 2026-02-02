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

    controller = MPCController(solver_backend=solver)

    # Initial conditions from config or hardcoded for sample task
    # X_0 = np.array([-50.0, 10.0, 5.0, 0.05, -0.01, 0.01])
    X_0 = config.orbit.initial_conditions
    X_f = np.zeros(6)
    U_0 = np.zeros(3)

    # OC uses single iteration
    effective_steps = 1 if mode == "oc" else max_steps

    # Prepare dynamics params
    dynamics_params = {
        "dt": config.dt,
        "mean_motion": config.orbit.mean_motion,  # derived property
        "eccentricity": config.orbit.eccentricity,
        # "alpha": ? "beta": ? if using drag
    }

    result = controller.run_mission(
        operation="rendezvous",
        dt=config.dt,
        t_0=0.0,
        num_chasers=1,
        num_mpc_steps=mode_config.num_mpc_steps,
        num_act_steps=mode_config.num_act_steps,
        X_0=X_0,
        f_X_f=X_f,
        U_0=U_0,
        dynamics_func=orbital_ellp_undrag,
        dynamics_params=dynamics_params,
        bounds=(config.mpc.state_bounds, config.mpc.input_bounds),
        max_mission_steps=effective_steps,
        tube_mpc=(
            {"enabled": mode_config.tube_mpc_enabled}
            if mode_config.tube_mpc_enabled
            else None
        ),
        adaptive=(
            {"enabled": mode_config.adaptive_enabled}
            if mode_config.adaptive_enabled
            else None
        ),
        # Extra explicit kwargs for solver
        t_periapsis=config.orbit.t_periapsis,
    )

    logger.info(f"Result: {'SUCCESS' if result.success else 'FAILED'}")
    logger.info(f"Steps completed: {len(result.time_history)}")
    logger.info(f"Total solve time: {result.solver_stats['total_solve_time']:.3f}s")

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
        except Exception as e:
            logger.error(f"  [{mode.upper()}] FAILED: {e}")
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
            steps = len(result.time_history)
            time_s = result.solver_stats["total_solve_time"]
            logger.info(f"{mode:<12} {status:<10} {steps:<8} {time_s:<10.3f}")
        else:
            logger.info(f"{mode:<12} {'ERROR':<10} {'-':<8} {'-':<10}")
