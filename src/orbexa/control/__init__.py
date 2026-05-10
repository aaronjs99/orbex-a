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

from orbexa.control.mpc_controller import MPCController, MissionResult
from orbexa.control.mpc_problem_builder import (
    ORBEXAProblemConfig,
    build_mpc_problem,
    build_from_dynamics,
)
from orbexa.control.dynamic_tube_model import ancillary_controller, calc_delta, calc_d
from orbexa.control.dynamic_tube_model import (
    TubeProfile,
    input_tightening_from_profile,
    propagate_tube_profile,
    tighten_box_bounds,
    tighten_target_params,
)
from orbexa.control.linearization import (
    AffineConstraint,
    linearize_cylinder_constraint,
    linearize_rendezvous_constraint,
)
from orbexa.control.constraints import (
    CylinderConstraint,
    collision_params_from_target_config,
    rendezvous_margin,
    rotating_body_point_velocity,
    rotating_docking_point,
    target_frame_position,
)

__all__ = [
    "MPCController",
    "MissionResult",
    "ORBEXAProblemConfig",
    "build_mpc_problem",
    "build_from_dynamics",
    "ancillary_controller",
    "calc_delta",
    "calc_d",
    "TubeProfile",
    "input_tightening_from_profile",
    "propagate_tube_profile",
    "tighten_box_bounds",
    "tighten_target_params",
    "AffineConstraint",
    "linearize_cylinder_constraint",
    "linearize_rendezvous_constraint",
    "CylinderConstraint",
    "collision_params_from_target_config",
    "rendezvous_margin",
    "rotating_body_point_velocity",
    "rotating_docking_point",
    "target_frame_position",
]
