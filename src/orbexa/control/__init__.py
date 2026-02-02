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
"""

from orbexa.control.mpc import MPCController, trajopt_mpc, mpc

__all__ = [
    "MPCController",
    "trajopt_mpc",
    "mpc",
]
