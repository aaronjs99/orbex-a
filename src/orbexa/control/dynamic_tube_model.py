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
from dataclasses import dataclass
from typing import List, Tuple

from orbexa.core.dynamics import orbital_ellp_drag


@dataclass(frozen=True)
class TubeProfile:
    """Discrete approximation of the paper's dynamic RCI tube states."""

    phi: np.ndarray
    omega: np.ndarray
    time_grid: np.ndarray

    @property
    def position_radius(self) -> np.ndarray:
        return np.linalg.norm(self.omega, axis=1)

    @property
    def max_position_radius(self) -> float:
        return float(np.max(self.position_radius))

    @property
    def max_state_error(self) -> np.ndarray:
        return np.concatenate(
            (
                np.max(np.abs(self.omega), axis=0),
                np.max(np.abs(self.phi), axis=0),
            )
        )


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
    def exp(x):
        return np.exp(x)

    @staticmethod
    def max2(a, b):
        return max(a, b)


# Create a global solver instance
_np_solver = NumPySolver()


def propagate_tube_profile(
    *,
    start_anom: float,
    num_steps: int,
    anom_step: float,
    mean_motion: float,
    t_periapsis: float = 0.0,
    lambda_gain: List[float],
    alpha: List[float],
    phi_0: List[float],
    eccentricity_range: Tuple[float, float],
    initial_error: np.ndarray = None,
    eta: float = 0.0,
    **kwargs,
) -> TubeProfile:
    """
    Propagate the tube boundary-layer and geometry states from Eq. 21d-21e.

    The result is conservative and discrete: each step evaluates the current
    feasible-set corner uncertainty and advances ``phi`` and ``Omega`` with
    forward Euler integration in true anomaly.
    """
    lambda_vec = np.asarray(lambda_gain, dtype=float)
    alpha_vec = np.asarray(alpha, dtype=float)
    phi = np.zeros((num_steps, 3), dtype=float)
    omega = np.zeros((num_steps, 3), dtype=float)
    phi[0] = np.maximum(np.asarray(phi_0, dtype=float), 0.0)

    if initial_error is not None:
        omega[0] = np.abs(np.asarray(initial_error, dtype=float)[:3])

    time_grid = start_anom + anom_step * np.arange(num_steps)

    for step in range(1, num_steps):
        q_prev = time_grid[step - 1]
        state_error_bound = np.concatenate((omega[step - 1], phi[step - 1]))
        delta_bound = np.abs(
            calc_delta(
                q_prev,
                t_periapsis,
                state_error_bound,
                anom_step=anom_step,
                mean_motion=mean_motion,
                e_range=eccentricity_range,
                a_range=kwargs.get("aRange", kwargs.get("a_range", (0.0, 0.0))),
                b_range=kwargs.get("bRange", kwargs.get("b_range", (0.0, 0.0))),
                mu=kwargs.get("mu", 3.986004418e14),
                semi_major_axis=kwargs.get("semi_major_axis"),
                specific_angular_momentum=kwargs.get("specific_angular_momentum"),
            )
        )
        disturbance_bound = np.abs(
            calc_d(
                q_prev,
                t_periapsis,
                dt=anom_step,
                mean_motion=mean_motion,
                e_range=eccentricity_range,
                a_range=kwargs.get("aRange", kwargs.get("a_range", (0.0, 0.0))),
                b_range=kwargs.get("bRange", kwargs.get("b_range", (0.0, 0.0))),
                mu=kwargs.get("mu", 3.986004418e14),
                semi_major_axis=kwargs.get("semi_major_axis"),
                specific_angular_momentum=kwargs.get("specific_angular_momentum"),
            )
        )

        phi_dot = -alpha_vec * phi[step - 1] + delta_bound + disturbance_bound + eta
        omega_dot = -lambda_vec * omega[step - 1] + phi[step - 1]
        phi[step] = np.maximum(phi[step - 1] + anom_step * phi_dot, 0.0)
        omega[step] = np.maximum(omega[step - 1] + anom_step * omega_dot, 0.0)

    return TubeProfile(phi=phi, omega=omega, time_grid=time_grid)


def tighten_box_bounds(bounds, tightening):
    """Apply symmetric tube tightening to simple lower/upper box bounds."""
    if bounds is None:
        return None

    tightening = np.asarray(tightening, dtype=float)
    tightened = []
    for idx, bound in enumerate(bounds):
        amount = float(tightening[idx]) if idx < len(tightening) else 0.0
        new_bound = dict(bound)
        lower = new_bound.get("lower")
        upper = new_bound.get("upper")
        if lower not in ["-Inf", None, float("-inf")]:
            new_bound["lower"] = float(lower) + amount
        if upper not in ["+Inf", None, float("inf")]:
            new_bound["upper"] = float(upper) - amount
        tightened.append(new_bound)
    return tightened


def input_tightening_from_profile(
    tube_profile: TubeProfile, lambda_gain: List[float]
) -> np.ndarray:
    """Conservative ancillary-input bound for Eq. 28 style tightening."""
    lambda_vec = np.asarray(lambda_gain, dtype=float)
    feedback_bound = np.abs(lambda_vec) * np.abs(tube_profile.omega)
    feedback_bound += np.abs(tube_profile.phi)
    return np.max(feedback_bound, axis=0)


def tighten_target_params(target_params, tube_profile: TubeProfile):
    """Expand nonconvex target exclusion constraints by the tube radius."""
    if not target_params:
        return target_params
    tightened = dict(target_params)
    tightened["tube_radius"] = max(
        float(tightened.get("tube_radius", 0.0)),
        tube_profile.max_position_radius,
    )
    return tightened


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
    A_nom_val = kwargs.get("A_nom_val")
    if A_nom_val is None:
        e_mid = float(np.mean(eccentricity_range))
        a_range = kwargs.get("aRange", (0.0, 0.0))
        b_range = kwargs.get("bRange", (0.0, 0.0))
        matrices, _, _ = orbital_ellp_drag(
            anom_step=dt,
            mean_motion=mean_motion,
            eccentricity=e_mid,
            alpha=float(np.mean(a_range)),
            beta=float(np.mean(b_range)),
            mu=kwargs.get("mu", 3.986004418e14),
            semi_major_axis=kwargs.get("semi_major_axis"),
            specific_angular_momentum=kwargs.get("specific_angular_momentum"),
        )
        A_func, _, _, _, _ = matrices
        A_nom_val = np.asarray(A_func(t, t_p, solver=_np_solver), dtype=float)

    r_tilde = np.array([act_state[i] - nom_state[i] for i in range(len(nom_state))])
    x_tilde = r_tilde[:3]
    v_tilde = r_tilde[3:]

    s = [v_tilde[i] + lambda_gain[i] * x_tilde[i] for i in range(len(x_tilde))]

    sat_s_phi = np.clip(
        np.divide(s, phi, out=np.zeros_like(s, dtype=float), where=np.asarray(phi) != 0),
        -1.0,
        1.0,
    )

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
        mu=kwargs.get("mu", 3.986004418e14),
        semi_major_axis=kwargs.get("semi_major_axis"),
        specific_angular_momentum=kwargs.get("specific_angular_momentum"),
    )

    K += np.array([alpha[i] * phi[i] for i in range(3)])

    state_mod = np.array(
        [
            -np.dot(A_nom_val[i + 3], r_tilde)
            - lambda_gain[i] * v_tilde[i]
            - sat_s_phi[i] * K[i]
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
    min_alpha, max_alpha = kwargs.get("a_range", kwargs.get("aRange", (0.0, 0.0)))
    min_beta, max_beta = kwargs.get("b_range", kwargs.get("bRange", (0.0, 0.0)))

    # Clamp eccentricity to valid range [0, 0.99] to avoid complex numbers
    min_ecc = max(0.0, min(0.99, min_ecc))
    max_ecc = max(0.0, min(0.99, max_ecc))

    A_list, Delta_list, Delta_norm = [], [], []

    # Grid search over the feasible-set corners.
    for ecc in [min_ecc, max_ecc]:
        for alpha in [min_alpha, max_alpha]:
            for beta in [min_beta, max_beta]:
                matrices, _, _ = orbital_ellp_drag(
                    anom_step=anom_step,
                    mean_motion=mean_motion,
                    eccentricity=ecc,
                    alpha=alpha,
                    beta=beta,
                    mu=kwargs.get("mu", 3.986004418e14),
                    semi_major_axis=kwargs.get("semi_major_axis"),
                    specific_angular_momentum=kwargs.get("specific_angular_momentum"),
                )
                A_func, _, _, _, _ = matrices
                A_list.append(np.asarray(A_func(t, t_p, solver=_np_solver), dtype=float))

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
    min_ecc, max_ecc = kwargs.get("e_range", (0.0, 0.0))
    min_alpha, max_alpha = kwargs.get("a_range", kwargs.get("aRange", (0.0, 0.0)))
    min_beta, max_beta = kwargs.get("b_range", kwargs.get("bRange", (0.0, 0.0)))

    d_list, D_list, D_norm = [], [], []

    for ecc in [min_ecc, max_ecc]:
        for alpha in [min_alpha, max_alpha]:
            for beta in [min_beta, max_beta]:
                matrices, _, _ = orbital_ellp_drag(
                    anom_step=dt,
                    mean_motion=mean_motion,
                    eccentricity=ecc,
                    alpha=alpha,
                    beta=beta,
                    mu=kwargs.get("mu", 3.986004418e14),
                    semi_major_axis=kwargs.get("semi_major_axis"),
                    specific_angular_momentum=kwargs.get("specific_angular_momentum"),
                )
                d_func = matrices[4]
                d_list.append(np.asarray(d_func(t, t_p, solver=_np_solver), dtype=float))

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
