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
from gekko import GEKKO

from orbexa.core import params as p
from orbexa.core.dynamics import orbital_ellp_undrag


def ancillary_controller(
    t_p: float,
    t_f: float,
    nom_state: np.ndarray,
    act_state: np.ndarray,
    A_nom_val: np.ndarray,
    Lambda: List[float],
    alpha: List[float],
    phi: List[float],
    *args,
    **kwargs,
) -> np.ndarray:
    """
    Calculate the ancillary control input to keep the actual system within the tube.

    Args:
        t_p, t_f: Time parameters.
        nom_state: Nominal state (from MPC).
        act_state: Actual state.
        A_nom_val: Nominal A matrix value.
        Lambda, alpha, phi: Tube controller gains.
        **kwargs: Must contain 'm' (GEKKO model) and range params.
    """
    m = kwargs["m"]

    r_tilde = np.array([act_state[i] - nom_state[i] for i in range(len(nom_state))])
    x_tilde = r_tilde[:3]
    v_tilde = r_tilde[3:]

    s = [v_tilde[i] + Lambda[i] * x_tilde[i] for i in range(len(x_tilde))]

    # Sigmoid smooth approximation for sign/saturation?
    # Original: s[i] / m.max2(s[i], phi[i]) -> logic seems like saturation 1/max(s, phi)?
    # If s < phi, returns s/phi. If s > phi, returns s/s = 1.
    # But max2(a,b) returns max. So if s < phi, result is s/phi. If s > phi, result is s/s=1.
    # This is effectively sat(s/phi). Correct.
    min_s_phi = [s[i] / m.max2(s[i], phi[i]) for i in range(len(s))]

    K = calc_delta(
        t_f,
        t_p,
        r_tilde,
        m=m,
        e_range=kwargs.get("eRange"),
        a_range=kwargs.get("aRange"),
        b_range=kwargs.get("bRange"),
    )

    K += np.array([alpha[i] * phi[i] for i in range(3)])

    state_mod = np.array(
        [
            np.dot(A_nom_val[i + 3], r_tilde)
            - Lambda[i] * v_tilde[i]
            - min_s_phi[i] * K[i]
            for i in range(3)
        ]
    )

    return state_mod


def calc_delta(t, t_p, x, *args, **kwargs):
    """Calculate robust disturbance bound Delta."""
    m = kwargs["m"]
    dt = p.dt

    min_ecc, max_ecc = kwargs.get("eRange", (p.minEccentricity, p.maxEccentricity))
    min_alpha, max_alpha = kwargs.get("aRange", (p.minDragAlpha, p.maxDragAlpha))
    min_beta, max_beta = kwargs.get("bRange", (p.minDragBeta, p.maxDragBeta))

    A_list, Delta_list, Delta_norm = [], [], []

    # Grid search for worst case
    for ecc in [min_ecc, max_ecc]:
        for alpha in [min_alpha, max_alpha]:
            for beta in [min_beta, max_beta]:
                matrices, _, _ = orbital_ellp_undrag(
                    dt, eccentricity=ecc, alpha=alpha, beta=beta
                )
                A_func, _, _, _, _ = matrices
                A_list.append(np.array(A_func(t, t_p, m=m)))

    for i, A_i in enumerate(A_list):
        for A_j in A_list[i + 1 :]:
            Delta = A_i - A_j
            Delta_list.append(Delta)
            Delta_norm.append(np.linalg.norm(Delta))

    # Worst case difference
    Delta_worst = Delta_list[np.argmax(Delta_norm)]
    Delta_val = np.matmul(Delta_worst, x)
    return Delta_val[3:]


def calc_d(t, t_p, *args, **kwargs):
    """Calculate worst case disturbance vector D."""
    m = kwargs.get("m", None)  # Optional m
    dt = p.dt

    min_ecc, max_ecc = kwargs.get("eRange", (p.minEccentricity, p.maxEccentricity))
    min_alpha, max_alpha = kwargs.get("aRange", (p.minDragAlpha, p.maxDragAlpha))
    min_beta, max_beta = kwargs.get("bRange", (p.minDragBeta, p.maxDragBeta))

    d_list, D_list, D_norm = [], [], []

    for ecc in [min_ecc, max_ecc]:
        for alpha in [min_alpha, max_alpha]:
            for beta in [min_beta, max_beta]:
                matrices, _, _ = orbital_ellp_undrag(
                    dt, eccentricity=ecc, alpha=alpha, beta=beta
                )
                d_func = matrices[4]  # d is 5th element
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


# Aliases
calcDelta = calc_delta
calcD = calc_d
