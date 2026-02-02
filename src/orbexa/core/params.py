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
ORBEX-A Simulation Parameters.

This module loads global parameters from the `config/default.yaml` file.
It provides a central access point for all simulation settings, ensuring
consistency across modules.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from orbexa.utils.io_utils import load_config

# Load Configuration
try:
    _CONFIG = load_config("config/default.yaml")
except Exception as e:
    print(f"Warning: Could not load config/default.yaml: {e}")
    print("Using hardcoded defaults.")
    _CONFIG = {}


def _get(path: str, default: Any) -> Any:
    """Helper to safely get nested config values."""
    keys = path.split(".")
    val = _CONFIG
    try:
        for key in keys:
            val = val[key]
        return val
    except (KeyError, TypeError):
        return default


# =============================================================================
# 1. Simulation Setup
# =============================================================================
decoupledMode: bool = _get("sim.decoupled_mode", True)
"""Use decoupled dynamics model."""

debug_section: bool = _get("sim.debug_section", True)
"""Enable debug section output."""

debug_plots: bool = _get("sim.debug_plots", False)
"""Enable debug plotting."""


# =============================================================================
# 2. Orbital Mechanics
# =============================================================================
# Periapsis Time Anomalies
t_p: float = _get("periapsis_anomalies.t_p", 0.0)
"""Time of periapsis passage."""

E_p: float = _get("periapsis_anomalies.E_p", 0.0)
"""Eccentric anomaly at periapsis."""

M_p: float = _get("periapsis_anomalies.M_p", 0.0)
"""Mean anomaly at periapsis."""

q_p: float = _get("periapsis_anomalies.q_p", 0.0)
"""True anomaly at periapsis."""

# Standard Constants
_dt_default = (math.pi / 20) / 200
dt: float = _get("sim.dt", _dt_default)
"""Simulation time step (seconds)."""

_a_default = (6378.1363 + 300.00) * 1000
a: float = float(_get("orbit.a", _a_default))
"""Semi-major axis (meters)."""

mu: float = float(_get("orbit.mu", 3.986004418e14))
"""Standard gravitational parameter for Earth (m^3/s^2)."""

# Derived Constant
n: float = np.sqrt(mu / a**3)
"""Mean motion (rad/s)."""


# =============================================================================
# 3. Reference Orbits & Adaptation
# =============================================================================
actOrbitParams: Dict[str, float] = _get(
    "mpc_orbit_params.actual",
    {
        "eccentricity": 0.125,
        "drag_alpha": 1.300e-7,
        "drag_beta": 2.600e-7,
    },
)
"""Actual parameters of the orbit being simulated."""

initAdaptParams: Dict[str, List[float]] = _get(
    "mpc_orbit_params.init_range",
    {
        "eccentricity": [0.000, 0.600],
        "drag_alpha": [0.00e-7, 9.00e-7],
        "drag_beta": [2.60e-7, 2.60e-7],
    },
)
"""Initial parameter uncertainty ranges for adaptation."""

nomOrbitParams: Dict[str, float] = _get(
    "mpc_orbit_params.nominal",
    {
        "eccentricity": 0.000,
        "drag_alpha": 0.000e-7,
        "drag_beta": 2.600e-7,
    },
)
"""Nominal orbit parameters."""

# Derived ranges for easy access
minEccentricity: float = initAdaptParams["eccentricity"][0]
maxEccentricity: float = initAdaptParams["eccentricity"][1]
minDragAlpha: float = initAdaptParams["drag_alpha"][0]
maxDragAlpha: float = initAdaptParams["drag_alpha"][1]
minDragBeta: float = initAdaptParams["drag_beta"][0]
maxDragBeta: float = initAdaptParams["drag_beta"][1]


# =============================================================================
# 4. Chaser Configuration
# =============================================================================
numChasers: int = _get("chaser.num_chasers", 1)
"""Number of chaser spacecraft."""

initialTimeLapse: int = _get("chaser.initial_time_lapse", 100)
"""Initial time delay before control starts."""

neighborMaxDist: float = _get("chaser.neighbor_max_dist", 1800.0)
"""Maximum distance for neighbor communication."""

totalTime: int = _get("sim.total_time", 100)
"""Total simulation duration."""

numUpdateSteps: int = _get("sim.num_update_steps", 200)
"""Number of steps per update cycle."""

iterTime: float = numUpdateSteps * dt
"""Time per iteration."""

numMPCSteps: Dict[str, int] = _get(
    "chaser.num_mpc_steps", {"rendezvous": 80, "docking": 40}
)
"""Horizon length for different modes."""

numActSteps: Dict[str, int] = _get(
    "chaser.num_act_steps", {"rendezvous": 20, "docking": 2}
)
"""Actuation steps for different modes."""

# State and Input Constraints
stateBounds: List[Dict[str, Any]] = _get(
    "chaser.state_bounds", [{"lower": "-Inf", "upper": "+Inf"} for _ in range(6)]
)
"""State bounds (position and velocity)."""

inputBounds: List[Dict[str, float]] = _get(
    "chaser.input_bounds", [{"lower": -1e5, "upper": 1e5} for _ in range(3)]
)
"""Control input bounds."""

forceBounds: List[Dict[str, float]] = _get(
    "chaser.force_bounds", [{"lower": -7e3, "upper": 7e3} for _ in range(3)]
)
"""Force bounds."""

goalBounds: Tuple[float, float] = tuple(_get("chaser.goal_bounds", [425.0, 475.0]))
"""Acceptable goal distance range."""


# =============================================================================
# 5. Target Properties
# =============================================================================
# Initial Orientation and Angular Velocity
th_T0: np.ndarray = np.array(_get("target.th_T0", [0.000, 0.200, -0.100]))
"""Initial Euler angles (orientation)."""

w_T0: np.ndarray = np.array(_get("target.w_T0", [-0.040, 0.100, -0.020]))
"""Initial angular velocity."""

# Inertia Matrix
I_T0: np.ndarray = np.array(
    _get(
        "target.inertia",
        [
            [800.0, 2.0, 5.0],
            [4.0, 450.0, 3.0],
            [6.0, 8.0, 200.0],
        ],
    )
)
"""Target moment of inertia tensor."""

pyramidalLimit: Dict[str, float] = _get(
    "target.pyramidal_limit",
    {
        "mu_x": 0.00,
        "mu_y": 0.00,
    },
)
"""Pyramidal approach cone parameters."""

radialLimit: float = _get("target.radial_limit", 1.0)
"""Radial Keep-Out Zone limit."""

targetShape: str = _get("target.shape", "cylinder")
"""Target geometry type: 'cylinder' or 'ellipsoid'."""

if targetShape == "cylinder":
    targetLimit = _get("target.limit", {"l_T": 0.60, "r_T": 0.80})
elif targetShape == "ellipsoid":
    targetLimit = _get("target.limit", {"r_Tx": 0.30, "r_Ty": 0.25, "r_Tz": 0.40})
else:
    # Use hardcoded default for cylinder if unknown
    targetLimit = {"l_T": 0.60, "r_T": 0.80}

targetCenter: np.ndarray = np.array(_get("target.center", [0, 0, 0]))
"""Target geometric center."""


# =============================================================================
# 6. Target Deflection & Discretization
# =============================================================================
TStopTime: float = _get("target.stop_time", 5.0)
"""Time to stop target rotation."""

chaserMinDist: float = _get("target.chaser_min_dist", 0.0)
"""Minimum safe distance for chaser."""

discretizeDockers: bool = _get("target.discretize_dockers", True)
"""Enable surface discretization for docking points."""

cylDiscPoints: List[List[float]] = []
if targetShape == "cylinder":
    for length in np.linspace(
        -targetLimit.get("l_T", 0.6), targetLimit.get("l_T", 0.6), 2
    ):
        for theta in np.arange(0, 2 * np.pi, np.pi / 8):
            cylDiscPoints.append(
                [
                    targetCenter[0] + targetLimit.get("r_T", 0.8) * np.cos(theta),
                    targetCenter[1] + targetLimit.get("r_T", 0.8) * np.sin(theta),
                    targetCenter[2] + length,
                ]
            )

# Observation & Tuning Parameters
targetObservationError: Dict[str, float] = {
    "ang_pos": _get("target.observation_error.ang_pos", 4.0e-8),
    "ang_vel": _get("target.observation_error.ang_vel", 4.0e-8),
}

chaserTuning: Dict[str, Any] = {
    "observe_target_epsilon": _get("chaser_tuning.observe_target_epsilon", 0.01),
    "goal_calc_w": _get("chaser_tuning.goal_calc.w", 1.0e3),
    "goal_calc_v": _get("chaser_tuning.goal_calc.v", [1.0e2, 1.0e6, 1.0e6]),
}


# =============================================================================
# 7. Adaptive Tube MPC Settings
# =============================================================================
tubeMPC: Dict[str, Any] = _get(
    "tube_mpc",
    {
        "runTube": True,
        "alpha_0": [0.25, 0.25, 0.25],
        "omega_0": [0.00, 0.00, 0.00],
        "phi_0": [0.50, 0.50, 0.50],
        "eta": [0.10, 0.10, 0.10],
        "Lambda": [60.00, 60.00, 60.00],
        "v_0": [0.00, 0.00, 0.00],
        "alpha_range": {
            "lower": 1.00e-2,
            "upper": 1.00e1,
        },
    },
)
# Handle snake_case vs camelCase mismatch in config vs code if necessary
# Config uses snake_case keys like run_tube, but code expects camelCase runTube
# We might need a small mapping here if the code is strict about keys
if "run_tube" in tubeMPC:
    tubeMPC["runTube"] = tubeMPC.pop("run_tube")
if "alpha_range" in tubeMPC and isinstance(tubeMPC["alpha_range"], dict):
    pass  # assume keys are correct in sub-dict
