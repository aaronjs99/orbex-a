# /***********************************************************
# *                                                         *
# * Copyright (c) 2026                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu                                *
# * Contact: aaronjs@ucla.edu                               *
# *                                                         *
# ***********************************************************/

"""
ORBEX-A Core Module

Core data structures, parameters, and orbital dynamics models.
"""

from orbexa.core.dynamics import (
    DynamicsModel,
    orbital_ellp_drag,
    orbital_ellp_undrag,
    orbital_circ_undrag,
    cwh_equations,
    orbital_params,
    triple_integrator,
)
from orbexa.core.config import SimulationConfig


def __getattr__(name):
    if name in {"Spacecraft", "Target", "Chaser"}:
        from orbexa.core.spacecraft import Chaser, Spacecraft, Target

        return {"Spacecraft": Spacecraft, "Target": Target, "Chaser": Chaser}[name]
    raise AttributeError(name)

__all__ = [
    # Dynamics
    "DynamicsModel",
    "orbital_ellp_drag",
    "orbital_ellp_undrag",
    "orbital_circ_undrag",
    "cwh_equations",
    "orbital_params",
    "triple_integrator",
    # Spacecraft
    "Spacecraft",
    "Target",
    "Chaser",
    # Config
    "SimulationConfig",
]
