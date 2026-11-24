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

"""orbital.py: ORBEX-A Orbital Dynamics Models.

This module provides various orbital dynamics models including:
- Clohessy-Wiltshire-Hill (CWH) equations for relative motion
- Undamped orbital dynamics (circular and elliptical)
- Triple integrator dynamics
- Orbital parameter calculations

All models support both continuous and discrete time representations.
This module is strictly functional and stateless; all parameters must be passed
as arguments.

Copyright: Aaron John Sabu
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from orbexa.models.dynamics.anomaly import orbital_radius_terms


@dataclass(frozen=True)
class DynamicsModel:
    """Typed wrapper for dynamics factories that still support tuple consumers."""

    A: Callable
    B: Callable
    Q: np.ndarray
    R: np.ndarray
    d: Callable
    state_bounds: list[dict[str, Any]] | None = None
    input_bounds: list[dict[str, float]] | None = None
    independent_variable: str = "true_anomaly"

    def as_tuple(self) -> tuple[Callable, Callable, np.ndarray, np.ndarray, Callable]:
        return self.A, self.B, self.Q, self.R, self.d

    def legacy_return(self):
        return self.as_tuple(), (None, None), (self.state_bounds, self.input_bounds)


def _backend(kwargs: dict[str, Any]):
    return kwargs.get("solver", kwargs.get("m", np))


def _exp(backend, value):
    return backend.exp(value) if hasattr(backend, "exp") else np.exp(value)


# =============================================================================
# 1. Orbital Parameter Calculation
# =============================================================================
# =============================================================================
# 2. Clohessy-Wiltshire-Hill (CWH) Dynamics
# =============================================================================
# =============================================================================
# 3. Time-Varying Orbital Dynamics (Elliptical)
# =============================================================================


def orbital_ellp_drag(
    anom_step: float | None = None,
    mean_motion: float = 0.001,
    eccentricity: float = 0.0,
    state_bounds: list[dict[str, Any]] | None = None,
    input_bounds: list[dict[str, float]] | None = None,
    alpha: float = 0.0,
    beta: float = 0.0,
    mu: float = 3.986004418e14,
    semi_major_axis: float | None = None,
    specific_angular_momentum: float | None = None,
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


# =============================================================================
# 4. Circular Orbit Undamped Dynamics
# =============================================================================
# =============================================================================
# 5. Triple Integrator Dynamics
# =============================================================================
