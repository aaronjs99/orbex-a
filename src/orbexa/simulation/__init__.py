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
ORBEX-A Simulation Module

High-level simulation runners and control mode configurations.
"""

from orbexa.simulation.modes import CONTROL_MODES, get_mode_config
from orbexa.simulation.paper_system import (
    ChaserConfig,
    MissionPhase,
    PaperSystemResult,
    PaperSystemRunner,
    PaperTruth,
    ParameterBelief,
    TumblingCylinderTarget,
)
from orbexa.simulation.runner import run_simulation, run_all_modes

__all__ = [
    "CONTROL_MODES",
    "get_mode_config",
    "ChaserConfig",
    "MissionPhase",
    "PaperSystemResult",
    "PaperSystemRunner",
    "PaperTruth",
    "ParameterBelief",
    "TumblingCylinderTarget",
    "run_simulation",
    "run_all_modes",
]
