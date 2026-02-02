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
ORBEX-A Estimation Module

State estimation, adaptation, and tube computation.
"""

from orbexa.estimation.adaptor import adaptor
from orbexa.estimation.enclosures import min_enclosing_ellipsoid
from orbexa.estimation.dynamictube import ancillary_controller, calcDelta, calcD

# Note: adaptor_plot is available in visualization, or via adaptor module if re-exported.
# For clarity, we only export estimation core functions here.

__all__ = [
    "adaptor",
    "min_enclosing_ellipsoid",
    "ancillary_controller",
    "calcDelta",
    "calcD",
]
