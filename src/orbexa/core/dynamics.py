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
ORBEX-A Orbital Dynamics Models.

This module provides various orbital dynamics models including:
- Clohessy-Wiltshire-Hill (CWH) equations for relative motion
- Undamped orbital dynamics (circular and elliptical)
- Triple integrator dynamics
- Orbital parameter calculations

All models support both continuous and discrete time representations.
This module is strictly functional and stateless; all parameters must be passed
as arguments.
"""

import math
import numpy as np
from types import SimpleNamespace
from typing import Tuple, List, Dict, Union, Any, Callable, Optional

from orbexa.utils import discretize, gen_skew_sym_mat


# =============================================================================
# 1. Orbital Parameter Calculation
# =============================================================================
def orbital_params(r_osc: np.ndarray, v_osc: np.ndarray, mu: float) -> SimpleNamespace:
    """
    Calculate orbital elements from state vectors.

    Args:
        r_osc (np.ndarray): Position vector (3,).
        v_osc (np.ndarray): Velocity vector (3,).
        mu (float): Standard gravitational parameter.

    Returns:
        SimpleNamespace: Object containing orbital elements:
            - r (float): Range.
            - v (float): Velocity magnitude.
            - h (float): Specific angular momentum.
            - e (float): Eccentricity.
            - a (float): Semi-major axis.
            - E (float): Eccentric anomaly (rad).
            - M (float): Mean anomaly (rad).
            - q (float): True anomaly (rad).
    """
    h_vec = np.cross(r_osc, v_osc)
    h = np.linalg.norm(h_vec)
    r = np.linalg.norm(r_osc)
    v = np.linalg.norm(v_osc)

    # Eccentricity vector
    e_vec = (np.cross(v_osc, h_vec) / mu) - (r_osc / r)
    e = np.linalg.norm(e_vec)

    # Semi-major axis
    energy = (v**2 / 2) - (mu / r)
    if abs(energy) < 1e-10:
        a = float("inf")
    else:
        a = -mu / (2 * energy)

    # True Anomaly
    if np.dot(r_osc, v_osc) >= 0:
        q = np.arccos(np.dot(e_vec, r_osc) / (e * r))
    else:
        q = 2 * np.pi - np.arccos(np.dot(e_vec, r_osc) / (e * r))

    if np.isnan(q):
        q = 0.0

    # Eccentric Anomaly
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(q / 2))

    # Mean Anomaly
    M = E - e * np.sin(E)

    # Correct quadrant for q
    if np.dot(r_osc, v_osc) < 0:
        q = 2 * math.pi - q

    return SimpleNamespace(r=r, v=v, h=h, e=e, a=a, E=E, M=M, q=q)


# =============================================================================
# 2. Clohessy-Wiltshire-Hill (CWH) Dynamics
# =============================================================================
def cwh_equations(
    dt: float,
    mean_motion: float,
    state_bounds: Optional[List[Dict[str, Any]]] = None,
    input_bounds: Optional[List[Dict[str, float]]] = None,
    discretize_model: bool = True,
    **kwargs,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    Tuple[Optional[List[Dict]], Optional[List[Dict]]],
]:
    """
    Clohessy-Wiltshire-Hill (CWH) equations for relative orbital motion.

    Also known as Hill's equations. Linearized dynamics for a chaser
    relative to a target in a circular orbit.

    Args:
        dt (float): Time step (seconds).
        mean_motion (float): Mean motion of reference orbit (rad/s).
        state_bounds (list, optional): List of state bound dicts.
        input_bounds (list, optional): List of input bound dicts.
        discretize_model (bool): Whether to discretize the model.

    Returns:
        tuple: (Matrices, Constraints, Bounds)
            - Matrices: (A, B, Q, R, d)
            - Constraints: (x_0, x_f) (default zero initialization)
            - Bounds: (state_bounds, input_bounds)
    """
    n = mean_motion

    # Continuous State Matrix
    A = np.array(
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [3 * n**2, 0, 0, 0, 2 * n, 0],
            [0, 0, 0, -2 * n, 0, 0],
            [0, 0, -(n**2), 0, 0, 0],
        ]
    )

    # Continuous Input Matrix
    B = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Cost Matrices (Default Identity)
    Q = np.identity(6)
    R = np.identity(3)

    # Disturbance (Zero)
    d = np.zeros(6)

    if discretize_model:
        A, B, _, _ = discretize(dt, A, B)

    matrices = (A, B, Q, R, d)
    constraints = (None, None)
    bounds = (state_bounds, input_bounds)

    return matrices, constraints, bounds


# =============================================================================
# 3. Time-Varying Orbital Dynamics (Elliptical)
# =============================================================================
def orbital_ellp_undrag(
    dt: float,
    mean_motion: float,
    eccentricity: float,
    state_bounds: Optional[List[Dict[str, Any]]] = None,
    input_bounds: Optional[List[Dict[str, float]]] = None,
    *args,
    **kwargs,
) -> Tuple[Callable, Callable, np.ndarray, np.ndarray, Callable]:
    """
    Linear parameter-varying model for elliptical orbits without drag.

    Returns functions for A(t) and B(t) as the system is time-varying.

    Args:
        dt (float): Time step.
        mean_motion (float): Mean motion (rad/s).
        eccentricity (float): Orbit eccentricity.
        state_bounds (list, optional): State bounds.
        input_bounds (list, optional): Input bounds.

    Returns:
        tuple: (matrices_funcs, constraints, bounds)
            - matrices is a tuple of (A_func, B_func, Q, R, d_func)
    """
    rho = 1.0  # Density scaling factor

    # Dynamics Functions (depend on true anomaly q)
    def A_func(t, t_p, *args, **kwargs):
        """Evaluation of LTV matrix A at time t or anomaly q."""
        solver = kwargs.get("solver", kwargs.get("m", np))
        q_val = kwargs.get("q")

        if q_val is None:
            if eccentricity == 0.0:
                q_val = mean_motion * (t - t_p)
            else:
                # Convert time to anomaly
                M_val = mean_motion * (t - t_p)
                enc_arg = M_val / 2.0
                E_val = 2 * solver.atan(
                    solver.sqrt((1 - eccentricity) / (1 + eccentricity))
                    * solver.tan(enc_arg)
                )
                q_val = 2 * solver.atan(
                    solver.sqrt((1 + eccentricity) / (1 - eccentricity))
                    * solver.tan(E_val / 2)
                )

        # Denominators
        den1 = (1 - eccentricity**2) ** 3
        den2 = (1 - eccentricity**2) ** 1.5

        coef_1 = mean_motion**2 * rho
        coef_2 = mean_motion * rho**2

        A_mat = [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [
                coef_1 * (3 + eccentricity * solver.cos(q_val)) / den1,
                coef_1 * (eccentricity * solver.sin(q_val)) / den1,
                0,
                coef_2 * (eccentricity * solver.sin(q_val)) / den2,
                2 * coef_2 / den2,
                0,
            ],
            [
                coef_1 * (eccentricity * solver.sin(q_val)) / den1,
                coef_1 * eccentricity * solver.cos(q_val) / den1,
                0,
                -2 * coef_2 / den2,
                coef_2 * (eccentricity * solver.sin(q_val)) / den2,
                0,
            ],
            [
                0,
                0,
                -coef_1 * (1) / den1,
                0,
                0,
                coef_2 * (eccentricity * solver.sin(q_val)) / den2,
            ],
        ]
        return np.array(A_mat) if "m" not in kwargs else A_mat

    def B_func(*args, **kwargs):
        return np.array(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )

    def d_func(t, t_p, *args, **kwargs):
        return np.zeros(6)

    # Cost Matrices
    Q = np.identity(6)
    R = np.identity(3)

    matrices = (A_func, B_func, Q, R, d_func)
    constraints = (None, None)
    bounds = (state_bounds, input_bounds)

    return matrices, constraints, bounds


# =============================================================================
# 4. Circular Orbit Undamped Dynamics
# =============================================================================
def orbital_circ_undrag(
    mean_motion: float,
    state_bounds: Optional[List[Dict[str, Any]]] = None,
    input_bounds: Optional[List[Dict[str, float]]] = None,
    *args,
    **kwargs,
) -> Tuple[Callable, Callable, np.ndarray, np.ndarray, Callable]:
    """
    Dynamics model for circular orbits without drag.

    Returns functions for A(t) and B(t), though they are constant here.
    """

    def A_func(*args, **kwargs):
        return np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [3 * mean_motion**2, 0, 0, 0, 2 * mean_motion, 0],
                [0, 0, 0, -2 * mean_motion, 0, 0],
                [0, 0, -(mean_motion**2), 0, 0, 0],
            ]
        )

    def B_func(*args, **kwargs):
        return np.array(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )

    def d_func(*args, **kwargs):
        return np.zeros(6)

    Q = np.identity(6)
    R = np.identity(3)

    matrices = (A_func, B_func, Q, R, d_func)
    constraints = (None, None)
    bounds = (state_bounds, input_bounds)

    return matrices, constraints, bounds


# =============================================================================
# 5. Triple Integrator Dynamics
# =============================================================================
def triple_integrator(dt: float, discretize_model: bool = True, **kwargs) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    Tuple[None, None],
]:
    """
    Simple triple integrator dynamics (snapshot model).

    Args:
        dt (float): Time step.
        discretize_model (bool): Whether to return discrete matrices.

    Returns:
        tuple: (Matrices, Constraints, Bounds)
    """
    A = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    B = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    d = np.zeros(9)
    Q = np.identity(9)
    R = np.identity(3)

    if discretize_model:
        A, B, _, _ = discretize(dt, A, B)

    matrices = (A, B, Q, R, d)
    constraints = (None, None)
    bounds = (None, None)

    return matrices, constraints, bounds
