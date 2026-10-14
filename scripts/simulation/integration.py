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

"""Numeric held-input integration shared by nominal and plant propagation."""
import numpy as np


def rk4_step(state, control, anomaly, step, matrices, periapsis=0.0):
    """One held-input transition, shared by nominal and plant propagation."""
    a, b, _, _, d = matrices
    input_term = np.asarray(b(), dtype=float) @ control
    def rhs(q, x):
        return np.asarray(a(q, periapsis), dtype=float) @ x + input_term + np.asarray(d(q, periapsis), dtype=float)
    k1 = rhs(anomaly, state)
    k2 = rhs(anomaly + step / 2, state + step * k1 / 2)
    k3 = rhs(anomaly + step / 2, state + step * k2 / 2)
    k4 = rhs(anomaly + step, state + step * k3)
    return state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
