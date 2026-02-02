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

from orbexa.core.dynamics import orbital_ellp_undrag


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
        **kwargs: Must contain 'm' (GEKKO model) and range params.
    """
    m = kwargs["m"]

    # Retrieve nominal A matrix if provided, otherwise it must be computed or estimates used.
    # In Tube MPC, A_nom_val usually comes from the nominal trajectory linearization.
    A_nom_val = kwargs.get("A_nom_val")
    if A_nom_val is None:
        # If not provided, one might compute it using orbital_ellp_undrag locally
        # or raise an error depending on design strictness.
        pass

    r_tilde = np.array([act_state[i] - nom_state[i] for i in range(len(nom_state))])
    x_tilde = r_tilde[:3]
    v_tilde = r_tilde[3:]

    s = [v_tilde[i] + lambda_gain[i] * x_tilde[i] for i in range(len(x_tilde))]

    # Sigmoid smooth approximation for sign/saturation
    # Using GEKKO max2 for saturation logic
    min_s_phi = [s[i] / m.max2(s[i], phi[i]) for i in range(len(s))]

    K = calc_delta(
        t_f,
        t_p,
        r_tilde,
        dt=dt,
        mean_motion=mean_motion,
        m=m,
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
    t: float, t_p: float, x: np.ndarray, dt: float, mean_motion: float, *args, **kwargs
):
    """Calculate robust disturbance bound Delta."""
    m = kwargs["m"]

    min_ecc, max_ecc = kwargs.get("e_range", (0.0, 0.0))
    # Original used global params defaults.
    # We must provide them now.

    # Note: orbital_ellp_undrag signature changed to:
    # orbital_ellp_undrag(dt, mean_motion, eccentricity, ...)

    A_list, Delta_list, Delta_norm = [], [], []

    A_list, Delta_list, Delta_norm = [], [], []

    # Grid search for worst case over eccentricity range.
    # Drag parameters (alpha/beta) were present in legacy code but are currently unused/constant.
    for ecc in [min_ecc, max_ecc]:
        matrices, _, _ = orbital_ellp_undrag(
            dt=dt, mean_motion=mean_motion, eccentricity=ecc
        )
        A_func, _, _, _, _ = matrices
        A_list.append(np.array(A_func(t, t_p, m=m)))

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
