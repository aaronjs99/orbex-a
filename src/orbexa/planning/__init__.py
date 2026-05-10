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
ORBEX-A Planning Module

Mission planning, task allocation, and observation optimization.
"""

from orbexa.planning.task_allocation import TaskAllocationSystem
from orbexa.planning.deflection import target_deflect

__all__ = [
    "TaskAllocationSystem",
    "target_deflect",
]
