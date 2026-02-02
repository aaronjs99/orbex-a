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
ORBEX-A Control Module

MPC and control algorithms for spacecraft rendezvous.
All control logic is solver-agnostic.
"""

from orbexa.control.mpc import MPCController, MissionResult
from orbexa.control.problem_builder import (
    ORBEXAProblemConfig,
    build_mpc_problem,
    build_from_dynamics,
)
from orbexa.control.dynamic_tube import ancillary_controller, calc_delta, calc_d

__all__ = [
    "MPCController",
    "MissionResult",
    "ORBEXAProblemConfig",
    "build_mpc_problem",
    "build_from_dynamics",
    "ancillary_controller",
    "calc_delta",
    "calc_d",
]
