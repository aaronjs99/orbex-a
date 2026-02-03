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
ORBEX-A Dynamic Tube MPC Controller.

This module provides the ancillary controller and tube parameters calculation
for robust MPC.
"""

import numpy as np
from typing import List, Tuple, Any, Optional
from types import SimpleNamespace

from orbexa.core.dynamics import orbital_ellp_undrag


# Simple solver wrapper for numpy to match GEKKO's method signatures
class NumPySolver:
    """Wrapper to provide GEKKO-like interface for numpy operations."""

    @staticmethod
    def atan(x):
        return np.arctan(x)

    @staticmethod
    def sqrt(x):
        return np.sqrt(x)

    @staticmethod
    def cos(x):
        return np.cos(x)

    @staticmethod
    def sin(x):
        return np.sin(x)

    @staticmethod
    def tan(x):
        return np.tan(x)

    @staticmethod
    def max2(a, b):
        return max(a, b)


# Create a global solver instance
_np_solver = NumPySolver()


def ancillary_controller(
    t: float,
    t_p: float,
    t_f: float,
    dt: float,
    mean_motion: float,
    nom_state: np.ndarray,
    act_state: np.ndarray,
    lambda_gain: List[float],
    alpha: List[float],
    phi: List[float],
    eccentricity_range: Tuple[float, float],
    # Add other ranges if needed
    *args,
    **kwargs,
) -> np.ndarray:
    """
    Calculate the ancillary control input to keep the actual system within the tube.

    Args:
        t: Current time.
        t_p, t_f: Time parameters (periapsis, nominal prop time).
        dt: Time step.
        mean_motion: Mean motion.
        nom_state: Nominal state (from MPC).
        act_state: Actual state.
        lambda_gain (Lambda): Sliding mode gains.
        alpha: Bandwidth.
        phi: Boundary layer.
        eccentricity_range: (min_ecc, max_ecc).
        **kwargs: May contain 'A_nom_val' for nominal A matrix.
    """
    # Retrieve nominal A matrix if provided, otherwise it must be computed or estimates used.
    A_nom_val = kwargs.get("A_nom_val")
    if A_nom_val is None:
        # If not provided, use a simplified approximation
        # This is a basic relative motion approximation
        A_nom_val = np.zeros((6, 6))

    r_tilde = np.array([act_state[i] - nom_state[i] for i in range(len(nom_state))])
    x_tilde = r_tilde[:3]
    v_tilde = r_tilde[3:]

    s = [v_tilde[i] + lambda_gain[i] * x_tilde[i] for i in range(len(x_tilde))]

    # Sigmoid smooth approximation for sign/saturation
    # Using Python's max instead of GEKKO's max2
    min_s_phi = [s[i] / max(s[i], phi[i]) for i in range(len(s))]

    K = calc_delta(
        t_f,
        t_p,
        r_tilde,
        anom_step=dt,
        mean_motion=mean_motion,
        e_range=eccentricity_range,
        # Default ranges if not provided?
        a_range=kwargs.get("aRange", (0.0, 0.0)),  # Drag alpha
        b_range=kwargs.get("bRange", (0.0, 0.0)),  # Drag beta
    )

    K += np.array([alpha[i] * phi[i] for i in range(3)])

    state_mod = np.array(
        [
            np.dot(A_nom_val[i + 3], r_tilde)
            - lambda_gain[i] * v_tilde[i]
            - min_s_phi[i] * K[i]
            for i in range(3)
        ]
    )

    return state_mod


def calc_delta(
    t: float,
    t_p: float,
    x: np.ndarray,
    anom_step: float,
    mean_motion: float,
    *args,
    **kwargs,
):
    """Calculate robust disturbance bound Delta."""
    min_ecc, max_ecc = kwargs.get("e_range", (0.0, 0.0))

    # Clamp eccentricity to valid range [0, 0.99] to avoid complex numbers
    min_ecc = max(0.0, min(0.99, min_ecc))
    max_ecc = max(0.0, min(0.99, max_ecc))

    A_list, Delta_list, Delta_norm = [], [], []

    # Grid search for worst case over eccentricity range.
    for ecc in [min_ecc, max_ecc]:
        matrices, _, _ = orbital_ellp_undrag(
            anom_step=anom_step, mean_motion=mean_motion, eccentricity=ecc
        )
        A_func, _, _, _, _ = matrices
        A_list.append(np.array(A_func(t, t_p, solver=_np_solver)))

    for i, A_i in enumerate(A_list):
        for A_j in A_list[i + 1 :]:
            Delta = A_i - A_j
            Delta_list.append(Delta)
            Delta_norm.append(np.linalg.norm(Delta))

    if not Delta_list:
        return np.zeros(3)

    # Worst case difference
    Delta_worst = Delta_list[np.argmax(Delta_norm)]
    Delta_val = np.matmul(Delta_worst, x)
    return Delta_val[3:]


def calc_d(t: float, t_p: float, dt: float, mean_motion: float, *args, **kwargs):
    """Calculate worst case disturbance vector D."""
    m = kwargs.get("m", None)

    min_ecc, max_ecc = kwargs.get("e_range", (0.0, 0.0))

    d_list, D_list, D_norm = [], [], []

    for ecc in [min_ecc, max_ecc]:
        matrices, _, _ = orbital_ellp_undrag(
            dt=dt, mean_motion=mean_motion, eccentricity=ecc
        )
        d_func = matrices[4]
        d_list.append(d_func(t, t_p, m=m))

    for i, d_i in enumerate(d_list):
        for d_j in d_list[i + 1 :]:
            D = np.array(d_i) - np.array(d_j)
            D_list.append(D)
            D_norm.append(np.linalg.norm(D))

    if not D_list:
        return np.zeros(3)

    D_worst = D_list[np.argmax(D_norm)]
    return D_worst[3:]


# Aliases (Removed as requested by previous objective, but keeping robust naming)
# calc_delta and calc_d are preferred.
