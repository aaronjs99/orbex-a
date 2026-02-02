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
"""

import math
import numpy as np
from types import SimpleNamespace
from typing import Tuple, List, Dict, Union, Any, Callable

# Use core.params
from orbexa.core import params as p
from orbexa.utils import discretize, genSkewSymMat


# =============================================================================
# 1. Orbital Parameter Calculations
# =============================================================================
def orbital_params(
    r_osc: np.ndarray, v_osc: np.ndarray, mu: float = p.mu
) -> SimpleNamespace:
    """
    Calculate orbital elements from state vectors.

    Args:
        r_osc: Osculating position vector (3,).
        v_osc: Osculating velocity vector (3,).
        mu: Gravitational parameter.

    Returns:
        SimpleNamespace containing:
            - r: Radius magnitude
            - v: Velocity magnitude
            - h: Angular momentum vector
            - e: Eccentricity vector
            - a: Semi-major axis
            - E: Eccentric anomaly
            - M: Mean anomaly
            - q: True anomaly
    """
    r = np.linalg.norm(r_osc)
    v = np.linalg.norm(v_osc)

    # Angular momentum
    h = np.cross(r_osc, v_osc)

    # Eccentricity vector
    e = np.cross(v_osc, h) / mu - r_osc / r

    # Semi-major axis (vis-viva equation)
    a = 1 / (2 / r - v**2 / mu)

    # Eccentric anomaly
    E = math.acos((1 - r / a) / np.linalg.norm(e))

    # Mean anomaly (Kepler's Equation)
    M = E - np.linalg.norm(e) * math.sin(E)

    # True anomaly
    q = math.acos(np.dot(e, r_osc) / (np.linalg.norm(e) * r))
    if np.dot(r_osc, v_osc) < 0:
        q = 2 * math.pi - q

    return SimpleNamespace(r=r, v=v, h=h, e=e, a=a, E=E, M=M, q=q)


# =============================================================================
# 2. Clohessy-Wiltshire-Hill (CWH) Dynamics
# =============================================================================
def cwh_equations(
    dt: float, n: float = p.n, discretize_model: bool = True, **kwargs
) -> Tuple[Any, Any, Any]:
    """
    Clohessy-Wiltshire-Hill (CWH) equations for relative orbital motion.

    Also known as Hill's equations. Linearized dynamics for a chaser
    relative to a target in a circular orbit.

    Args:
        dt: Time step.
        n: Mean motion of the reference orbit.
        discretize_model: Whether to return discrete matrices.
        **kwargs: Additional arguments.

    Returns:
        tuple: (Matrices, Constraints, Bounds)
            - Matrices: (A, B, Q, R, d)
            - Constraints: (x_0, x_f)
            - Bounds: (stateBounds, inputBounds)
    """
    # Continuous State Matrix
    A = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [3 * n**2, 0.0, 0.0, 0.0, 2 * n, 0.0],
            [0.0, 0.0, 0.0, -2 * n, 0.0, 0.0],
            [0.0, 0.0, -(n**2), 0.0, 0.0, 0.0],
        ]
    )

    # Continuous Input Matrix
    B = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Default Costs
    Q = np.identity(6)
    R = np.identity(3) * 10
    d = np.zeros(6)  # Disturbance

    if discretize_model:
        A, B = discretize(dt, A, B)

    matrices = (A, B, Q, R, d)
    constraints = (None, None)
    bounds = (p.stateBounds, p.inputBounds)

    return matrices, constraints, bounds


# =============================================================================
# 3. Triple Integrator Dynamics
# =============================================================================
def triple_integrator(
    dt: float, discretize_model: bool = True, **kwargs
) -> Tuple[Any, Any, Any]:
    """
    Triple integrator dynamics model (jerk control).

    Args:
        dt: Time step.
        discretize_model: Whether to return discrete matrices.
        **kwargs: Additional arguments.

    Returns:
        tuple: (Matrices, Constraints, Bounds)
    """
    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    B = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ]
    )

    Q = np.identity(6)
    R = np.identity(2)
    d = np.zeros(6)

    if discretize_model:
        A, B = discretize(dt, A, B)

    matrices = (A, B, Q, R, d)
    constraints = (None, None)
    bounds = (None, None)

    return matrices, constraints, bounds


# =============================================================================
# 4. Elliptical Orbit Dynamics (Undamped)
# =============================================================================
def orbital_ellp_undrag(
    *args, **kwargs
) -> Tuple[Callable, Callable, np.ndarray, np.ndarray, Callable]:
    """
    Linear parameter-varying model for elliptical orbits without drag.

    Args:
        *args: Variable arguments.
        **kwargs: Must contain 'dt', 'bounds', 'constraints'.

    Returns:
        tuple: (A(t), B, Q, R, d(t)) - Functions of time or anomaly.
    """
    dt = kwargs["dt"]
    constraints = kwargs.get("constraints", (None, None))
    bounds = kwargs.get("bounds", (None, None))

    # Dynamics Functions (depend on true anomaly q)
    def A(t, t_p, *args, **kwargs):
        m = kwargs.get("m", math)
        ecc = p.actOrbitParams["eccentricity"]

        # Calculate true anomaly q
        E = 2 * m.atan(m.sqrt((1 - ecc) / (1 + ecc)) * m.tan((t - t_p) / 2))
        M = E - ecc * m.sin(E)
        q = M + p.q_p

        # Radial distance (normalized)
        rho = 1 + ecc * m.cos(q)

        # State Matrix
        A_mat = [
            [0, 0, 0, 1, 0, 0],  # x_dot
            [0, 0, 0, 0, 1, 0],  # y_dot
            [0, 0, 0, 0, 0, 1],  # z_dot
            [
                p.n**2 * rho * (3 + ecc * m.cos(q)) / (1 - ecc**2) ** 3,
                p.n**2 * rho * (ecc * m.sin(q)) / (1 - ecc**2) ** 3,
                0,
                p.n * rho**2 * (ecc * m.sin(q)) / (1 - ecc**2) ** 1.5,
                2 * p.n * rho**2 / (1 - ecc**2) ** 1.5,
                0,
            ],  # x_ddot
            [
                p.n**2 * rho * (ecc * m.sin(q)) / (1 - ecc**2) ** 3,
                p.n**2 * rho * ecc * m.cos(q) / (1 - ecc**2) ** 3,
                0,
                -2 * p.n * rho**2 / (1 - ecc**2) ** 1.5,
                p.n * rho**2 * (ecc * m.sin(q)) / (1 - ecc**2) ** 1.5,
                0,
            ],  # y_ddot
            [
                0,
                0,
                -p.n**2 * rho * (1) / (1 - ecc**2) ** 3,
                0,
                0,
                p.n * rho**2 * (ecc * m.sin(q)) / (1 - ecc**2) ** 1.5,
            ],  # z_ddot
        ]
        return np.array(A_mat) if "m" not in kwargs else A_mat

    B = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    Q = np.identity(6)
    R = np.identity(3) * 1e-4

    def d(t, t_p, *args, **kwargs):
        # Disturbance vector (currently zero)
        return [0.0] * 6

    return (A, B, Q, R, d), constraints, bounds


# =============================================================================
# 5. Circular Orbit Dynamics (Undamped)
# =============================================================================
def orbital_circ_undrag(
    *args, **kwargs
) -> Tuple[Callable, Callable, np.ndarray, np.ndarray, Callable]:
    """
    Dynamics model for circular orbits without drag.

    Args:
        *args: Variable arguments.
        **kwargs: Must contain 'dt', 'bounds', 'constraints'.

    Returns:
        tuple: (A(t), B, Q, R, d(t))
    """
    dt = kwargs["dt"]
    constraints = kwargs.get("constraints", (None, None))
    bounds = kwargs.get("bounds", (None, None))

    def A(t, *args, **kwargs):
        return np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [3 * p.n**2, 0, 0, 0, 2 * p.n, 0],
                [0, 0, 0, -2 * p.n, 0, 0],
                [0, 0, -p.n**2, 0, 0, 0],
            ]
        )

    B = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    Q = np.identity(6)
    R = np.identity(3) * 1e-4

    def d(t, *args, **kwargs):
        return np.zeros(6)

    return (A, B, Q, R, d), constraints, bounds


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================
orbitalParams = orbital_params
cwhEquations = cwh_equations
tripleIntegrator = triple_integrator
