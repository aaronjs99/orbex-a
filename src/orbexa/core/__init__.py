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
ORBEX-A Core Module

Core data structures, parameters, and orbital dynamics models.
"""

from orbexa.core.params import *
from orbexa.core.dynamics import (
    orbital_ellp_undrag,
    orbital_circ_undrag,
    cwhEquations,
    orbitalParams,
    tripleIntegrator,
)
from orbexa.core.spacecraft import Spacecraft, Target, Chaser

__all__ = [
    # Parameters
    "dt",
    "n",
    "actOrbitParams",
    "initAdaptParams",
    "stateBounds",
    "inputBounds",
    "goalBounds",
    # Dynamics
    "orbital_ellp_undrag",
    "orbital_circ_undrag",
    "cwhEquations",
    "orbitalParams",
    "tripleIntegrator",
    # Spacecraft
    "Spacecraft",
    "Target",
    "Chaser",
]
