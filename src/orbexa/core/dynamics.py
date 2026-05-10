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
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Tuple, List, Dict, Any, Callable, Optional

from orbexa.utils import discretize, gen_skew_sym_mat


@dataclass(frozen=True)
class DynamicsModel:
    """Typed wrapper for dynamics factories that still support tuple consumers."""

    A: Callable
    B: Callable
    Q: np.ndarray
    R: np.ndarray
    d: Callable
    state_bounds: Optional[List[Dict[str, Any]]] = None
    input_bounds: Optional[List[Dict[str, float]]] = None
    independent_variable: str = "true_anomaly"

    def as_tuple(self) -> Tuple[Callable, Callable, np.ndarray, np.ndarray, Callable]:
        return self.A, self.B, self.Q, self.R, self.d

    def legacy_return(self):
        return self.as_tuple(), (None, None), (self.state_bounds, self.input_bounds)


def _backend(kwargs: Dict[str, Any]):
    return kwargs.get("solver", kwargs.get("m", np))


def _sqrt(backend, value):
    return backend.sqrt(value) if hasattr(backend, "sqrt") else np.sqrt(value)


def _exp(backend, value):
    return backend.exp(value) if hasattr(backend, "exp") else np.exp(value)


def _default_specific_angular_momentum(
    backend,
    *,
    mean_motion: float,
    eccentricity,
    mu: float,
    semi_major_axis: Optional[float],
):
    if semi_major_axis is None:
        if mean_motion <= 0.0:
            semi_major_axis = 1.0
        else:
            semi_major_axis = (mu / mean_motion**2) ** (1.0 / 3.0)
    return _sqrt(backend, mu * semi_major_axis * (1.0 - eccentricity**2))


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
    anom_step: float,
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
    state_cost_matrix = np.identity(6)
    input_cost_matrix = np.identity(3)

    # Disturbance (Zero)
    d = np.zeros(6)

    if discretize_model:
        A, B, _, _ = discretize(anom_step, A, B)

    matrices = (A, B, state_cost_matrix, input_cost_matrix, d)
    constraints = (None, None)
    bounds = (state_bounds, input_bounds)

    return matrices, constraints, bounds


# =============================================================================
# 3. Time-Varying Orbital Dynamics (Elliptical)
# =============================================================================
def orbital_ellp_drag(
    anom_step: Optional[float] = None,
    mean_motion: float = 0.001,
    eccentricity: float = 0.0,
    state_bounds: Optional[List[Dict[str, Any]]] = None,
    input_bounds: Optional[List[Dict[str, float]]] = None,
    alpha: float = 0.0,
    beta: float = 0.0,
    mu: float = 3.986004418e14,
    semi_major_axis: Optional[float] = None,
    specific_angular_momentum: Optional[float] = None,
    theta0: float = 0.0,
    *args,
    **kwargs,
) -> Tuple[Callable, Callable, np.ndarray, np.ndarray, Callable]:
    """
    Extended Tschauner-Hempel dynamics with quadratic drag.

    This implements the paper's Appendix B model with true anomaly as the
    independent variable. The drag constants are ``alpha`` for the target and
    ``beta`` for the chaser; zero values recover the no-drag model.
    """
    if anom_step is None:
        anom_step = kwargs.get("dt")
    if anom_step is None:
        raise ValueError("anom_step is required for orbital_ellp_drag")

    alpha = kwargs.get("drag_alpha", alpha)
    beta = kwargs.get("drag_beta", beta)
    specific_angular_momentum = kwargs.get("h", specific_angular_momentum)
    specific_angular_momentum = kwargs.get(
        "specific_angular_momentum", specific_angular_momentum
    )
    theta0 = kwargs.get("theta0", theta0)

    def _orbit_radius_terms(q_val, local_kwargs):
        solver = _backend(local_kwargs)
        h_val = specific_angular_momentum
        if h_val is None:
            h_val = _default_specific_angular_momentum(
                solver,
                mean_motion=mean_motion,
                eccentricity=eccentricity,
                mu=mu,
                semi_major_axis=semi_major_axis,
            )

        exp_term = _exp(solver, 2.0 * alpha * q_val)
        denom = exp_term + eccentricity * solver.cos(q_val - theta0)
        scale = h_val**2 * (1.0 + 4.0 * alpha**2) / mu
        radius = scale / denom
        denom_prime = 2.0 * alpha * exp_term - eccentricity * solver.sin(
            q_val - theta0
        )
        radius_prime = -scale * denom_prime / denom**2
        return radius, radius_prime, h_val

    def A_func(t, t_p=0.0, *args, **kwargs):
        """Evaluate the true-anomaly dynamics matrix."""
        q_val = kwargs.get("q", t)
        radius, radius_prime, h_val = _orbit_radius_terms(q_val, kwargs)
        gamma = beta - alpha
        radius_ratio = radius_prime / radius
        gravity_curvature = 3.0 * radius * mu / h_val**2

        A_mat = [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [-gamma * radius_ratio, gamma, 0.0, -gamma, 2.0, 0.0],
            [
                -gamma,
                -gamma * radius_ratio + gravity_curvature,
                0.0,
                -2.0,
                -gamma,
                0.0,
            ],
            [0.0, 0.0, -1.0 - gamma * radius_ratio, 0.0, 0.0, gamma],
        ]
        return A_mat if "m" in kwargs else np.array(A_mat, dtype=object)

    def B_func(*args, **kwargs):
        return np.array(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=float,
        )

    def d_func(t, t_p=0.0, *args, **kwargs):
        q_val = kwargs.get("q", t)
        solver = _backend(kwargs)
        radius, radius_prime, h_val = _orbit_radius_terms(q_val, kwargs)
        gamma = beta - alpha
        drag_disturbance = gamma * radius * _exp(solver, alpha * q_val) / _sqrt(
            solver, h_val
        )
        values = [
            0.0,
            0.0,
            0.0,
            radius * drag_disturbance,
            -radius_prime * drag_disturbance,
            0.0,
        ]
        return values if "m" in kwargs else np.array(values, dtype=object)

    state_cost_matrix = np.asarray(kwargs.get("state_cost_matrix", np.identity(6)))
    input_cost_matrix = np.asarray(kwargs.get("input_cost_matrix", np.identity(3)))

    model = DynamicsModel(
        A=A_func,
        B=B_func,
        Q=state_cost_matrix,
        R=input_cost_matrix,
        d=d_func,
        state_bounds=state_bounds,
        input_bounds=input_bounds,
    )
    return model.legacy_return()


def orbital_ellp_undrag(
    anom_step: Optional[float] = None,
    mean_motion: float = 0.001,
    eccentricity: float = 0.0,
    state_bounds: Optional[List[Dict[str, Any]]] = None,
    input_bounds: Optional[List[Dict[str, float]]] = None,
    *args,
    **kwargs,
) -> Tuple[Callable, Callable, np.ndarray, np.ndarray, Callable]:
    """Elliptical true-anomaly dynamics without differential drag."""
    kwargs.pop("alpha", None)
    kwargs.pop("beta", None)
    kwargs.pop("drag_alpha", None)
    kwargs.pop("drag_beta", None)
    return orbital_ellp_drag(
        anom_step=anom_step,
        mean_motion=mean_motion,
        eccentricity=eccentricity,
        state_bounds=state_bounds,
        input_bounds=input_bounds,
        alpha=0.0,
        beta=0.0,
        *args,
        **kwargs,
    )


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

    state_cost_matrix = np.identity(6)
    input_cost_matrix = np.identity(3)

    matrices = (A_func, B_func, state_cost_matrix, input_cost_matrix, d_func)
    constraints = (None, None)
    bounds = (state_bounds, input_bounds)

    return matrices, constraints, bounds


# =============================================================================
# 5. Triple Integrator Dynamics
# =============================================================================
def triple_integrator(
    anom_step: float, discretize_model: bool = True, **kwargs
) -> Tuple[
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
    state_cost_matrix = np.identity(9)
    input_cost_matrix = np.identity(3)

    if discretize_model:
        A, B, _, _ = discretize(anom_step, A, B)

    matrices = (A, B, state_cost_matrix, input_cost_matrix, d)
    constraints = (None, None)
    bounds = (None, None)

    return matrices, constraints, bounds
