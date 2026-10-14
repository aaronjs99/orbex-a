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

import numpy as np
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Tuple, List, Dict, Any, Callable, Optional

from orbexa.utils import discretize, gen_skew_sym_mat
from orbexa.utils.anomaly import orbital_radius_terms


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


def _exp(backend, value):
    return backend.exp(value) if hasattr(backend, "exp") else np.exp(value)


# =============================================================================
# 1. Orbital Parameter Calculation
# =============================================================================
def orbital_params(r_osc: np.ndarray, v_osc: np.ndarray, mu: float) -> SimpleNamespace:
    """
    Calculate elements of a bound elliptical orbit from SI inertial vectors.

    Anomalies are returned in [0, 2*pi). Circular orbits use argument of
    latitude from the ascending node, or longitude from +X if equatorial,
    because their periapsis and true anomaly are otherwise undefined.

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
    r_osc = np.asarray(r_osc, dtype=float).reshape(3)
    v_osc = np.asarray(v_osc, dtype=float).reshape(3)
    if (
        not np.isfinite(mu)
        or mu <= 0.0
        or not np.all(np.isfinite(r_osc))
        or not np.all(np.isfinite(v_osc))
    ):
        raise ValueError("finite orbital vectors and positive mu are required")
    h_vec = np.cross(r_osc, v_osc)
    h = np.linalg.norm(h_vec)
    r = np.linalg.norm(r_osc)
    v = np.linalg.norm(v_osc)
    if r <= 0.0 or h <= 0.0:
        raise ValueError("nonzero radius and angular momentum are required")

    # Eccentricity vector
    e_vec = (np.cross(v_osc, h_vec) / mu) - (r_osc / r)
    e = np.linalg.norm(e_vec)

    # Semi-major axis
    energy = (v**2 / 2) - (mu / r)
    if abs(energy) < 1e-10:
        a = float("inf")
    else:
        a = -mu / (2 * energy)

    if e >= 1.0:
        raise ValueError("orbital_params currently requires an elliptical orbit")

    # atan2 selects the inbound quadrant once, without an acos roundoff error.
    normal = h_vec / h
    if e > 1.0e-10:
        periapsis_direction = e_vec / e
        q = np.arctan2(
            np.dot(r_osc, np.cross(normal, periapsis_direction)),
            np.dot(r_osc, periapsis_direction),
        ) % (2.0 * np.pi)
    else:
        # Periapsis is undefined for a circular orbit. Return argument of
        # latitude from the ascending node, or orbital longitude from +X for
        # an equatorial orbit, consistently as q=E=M.
        node = np.cross(np.array([0.0, 0.0, 1.0]), normal)
        reference = (
            node / np.linalg.norm(node)
            if np.linalg.norm(node) > 1.0e-10
            else np.array([1.0, 0.0, 0.0])
        )
        q = np.arctan2(
            np.dot(r_osc, np.cross(normal, reference)), np.dot(r_osc, reference)
        ) % (2.0 * np.pi)
    if e <= 1.0e-10:
        E = M = q
    else:
        E = np.arctan2(np.sqrt(1.0 - e**2) * np.sin(q), e + np.cos(q)) % (
            2.0 * np.pi
        )
        M = E - e * np.sin(E)

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

    This compatibility factory uses time in seconds and the conventional
    axes x=radial outward, y=transverse, z=orbit normal. Its legacy parameter
    name ``anom_step`` denotes seconds here. To compare with the physical
    elliptical factory, map positions and velocities with
    ``(x_ellp, y_ellp, z_ellp) = (-y_cwh, x_cwh, z_cwh)`` and convert the
    velocity independent variable from time to anomaly.

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
):
    """Physical LVLH relative dynamics, version ``physical-lvlh-v1``.

    The six states are ``[p, dp/dq]`` with p in meters and dp/dq in m/rad.
    Axes are x opposite the transverse direction, y radially outward, and z
    along the target's orbital angular momentum. They form a right-handed
    frame; x is not opposite the full velocity on an eccentric orbit.

    Inputs are anomaly accelerations in m/rad^2, so B has a lower identity.
    A physical thruster acceleration in m/s^2 must be divided by (dq/dt)^2
    before applying it here. All geometry therefore acts on physical meters.

    The model uses Carter--Humi's pre-transformation, small-separation and
    small-radial-speed approximation with inverse-radius quadratic drag.
    Alpha/beta are dimensionless target/chaser drag constants, not C_D.
    It is not a full atmosphere or nonlinear two-spacecraft propagator.
    No-drag circular motion reduces to HCW in the stated axis convention.
    ``anom_step`` is retained for API compatibility; callbacks are continuous
    in unwrapped q, and discretization is the consumer's responsibility.
    """
    if anom_step is None:
        anom_step = kwargs.get("dt")
    if anom_step is None:
        raise ValueError("anom_step is required for orbital_ellp_drag")
    alpha = kwargs.get("drag_alpha", alpha)
    beta = kwargs.get("drag_beta", beta)
    specific_angular_momentum = kwargs.get("h", specific_angular_momentum)
    theta0 = kwargs.get("theta0", theta0)

    def radius_terms(q, local_kwargs):
        return orbital_radius_terms(
            q,
            eccentricity=eccentricity,
            alpha=alpha,
            mu=mu,
            semi_major_axis=semi_major_axis,
            mean_motion=mean_motion,
            specific_angular_momentum=specific_angular_momentum,
            theta0=theta0,
            solver=_backend(local_kwargs),
        )

    def A_func(t, t_p=0.0, *args, **local_kwargs):
        q = local_kwargs.get("q", t)
        radius, radius_prime, h = radius_terms(q, local_kwargs)
        k = radius_prime / radius
        gamma = beta - alpha
        curvature = mu * radius * _exp(_backend(local_kwargs), 2.0 * alpha * q) / h**2
        damping = 2.0 * k - gamma
        values = [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0 - curvature, -damping, 0.0, damping, 2.0, 0.0],
            [damping, 1.0 + 2.0 * curvature, 0.0, -2.0, damping, 0.0],
            [0.0, 0.0, -curvature, 0.0, 0.0, damping],
        ]
        return values if "m" in local_kwargs else np.array(values, dtype=object)

    def B_func(*args, **local_kwargs):
        return np.vstack((np.zeros((3, 3)), np.eye(3)))

    def d_func(t, t_p=0.0, *args, **local_kwargs):
        q = local_kwargs.get("q", t)
        radius, radius_prime, _ = radius_terms(q, local_kwargs)
        gamma = beta - alpha
        values = [0.0, 0.0, 0.0, gamma * radius, -gamma * radius_prime, 0.0]
        return values if "m" in local_kwargs else np.array(values, dtype=object)

    return DynamicsModel(
        A=A_func,
        B=B_func,
        Q=np.asarray(kwargs.get("state_cost_matrix", np.identity(6))),
        R=np.asarray(kwargs.get("input_cost_matrix", np.identity(3))),
        d=d_func,
        state_bounds=state_bounds,
        input_bounds=input_bounds,
    ).legacy_return()


def orbital_ellp_undrag(
    anom_step: Optional[float] = None,
    mean_motion: float = 0.001,
    eccentricity: float = 0.0,
    state_bounds: Optional[List[Dict[str, Any]]] = None,
    input_bounds: Optional[List[Dict[str, float]]] = None,
    *args,
    **kwargs,
) -> Tuple[Callable, Callable, np.ndarray, np.ndarray, Callable]:
    """Physical ``[p, dp/dq]`` LVLH dynamics with both drag constants zero.

    Uses the same axes and m/rad^2 input convention as ``orbital_ellp_drag``.
    """
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
    Time is in seconds, velocity in m/s, and inputs in m/s^2. This older
    factory retains x=radial outward, y=transverse, z=orbit normal, as in
    ``cwh_equations``; see its documented mapping to the elliptical axes.
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
