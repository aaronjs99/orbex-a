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

State estimation and adaptation algorithms.
"""

from orbexa.estimation.adaptor import SMIDAdaptor, SMIDRecord, run_adaptation, run_adaptor_op
from orbexa.estimation.enclosures import min_enclosing_ellipsoid

__all__ = [
    "run_adaptation",
    "run_adaptor_op",
    "SMIDAdaptor",
    "SMIDRecord",
    "min_enclosing_ellipsoid",
]
