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
ORBEX-A Planning Module

Mission planning, task allocation, and observation optimization.
"""

from orbexa.planning.taskalloc import TaskAllocAgent
from orbexa.planning.deflection import target_deflect

__all__ = [
    "TaskAllocAgent",
    "target_deflect",
]
