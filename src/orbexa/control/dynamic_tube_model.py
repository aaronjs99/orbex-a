# /***********************************************************
# * Copyright (c) 2026                                      *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# * Authors: Aaron John Sabu, Brett T. Lopez                *
# * Contact: {aaronjs, btlopez}@ucla.edu                    *
# ***********************************************************/

"""Solver-independent boundary-layer feedback and dynamic tube arithmetic.

Phi bounds the sliding error ``e_v + Lambda e_p``; Omega bounds position error.
The nominal model and its parameter-uncertainty envelope must refer to the same
physical LVLH dynamics. The continuous-feedback tube argument does not by itself
certify a sampled, held ancillary input or numerical integration error.
"""

from dataclasses import dataclass
from typing import List, Tuple
import warnings

import numpy as np

from orbexa.core.dynamics import orbital_ellp_drag
from orbexa.control.uncertainty import uncertainty_envelope, nominal_acceleration_bound


def _vector(value, name, *, nonnegative=False):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _matrix(value, name, *, nonnegative=False):
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 6) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3 by 6 matrix")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True)
class TubeProfile:
    """Nodewise bounds, gains, and ancillary input reserves for one horizon."""

    phi: np.ndarray
    omega: np.ndarray
    time_grid: np.ndarray
    lambda_gain: np.ndarray = None
    alpha: np.ndarray = None
    input_reserve: np.ndarray = None
    assumptions: Tuple[str, ...] = ()

    @property
    def position_radius(self):
        return np.linalg.norm(self.omega, axis=1)

    @property
    def max_position_radius(self):
        return float(np.max(self.position_radius))

    @property
    def velocity_error_bound(self):
        if self.lambda_gain is None:
            raise ValueError("Velocity error bounds require the profile's sliding gains")
        return np.asarray(self.phi) + np.abs(self.lambda_gain) * np.asarray(self.omega)

    @property
    def max_state_error(self):
        return np.concatenate((np.max(self.omega, axis=0),
                               np.max(self.velocity_error_bound, axis=0)))


def advance_tube(phi, omega, alpha, lambda_gain, disturbance, step):
    """Positive exponential step for constant bandwidth and forcing bounds.

    ``disturbance`` bounds nominal-model uncertainty plus affine disturbances
    over the interval. Phi is integrated exactly for that constant upper bound.
    Omega uses the larger endpoint Phi, bounding its convolution even when Phi
    changes during the interval. No forward-Euler clipping is used.
    """
    phi = _vector(phi, "phi", nonnegative=True)
    omega = _vector(omega, "omega", nonnegative=True)
    alpha = _vector(alpha, "alpha", nonnegative=True)
    gain = _vector(lambda_gain, "lambda_gain", nonnegative=True)
    forcing = _vector(disturbance, "disturbance", nonnegative=True)
    step = float(step)
    if not np.isfinite(step) or step < 0.0:
        raise ValueError("Tube step must be finite and nonnegative")
    if step == 0.0:
        return phi.copy(), omega.copy()
    alpha_weight = np.divide(-np.expm1(-alpha * step), alpha,
                             out=np.full(3, step), where=alpha != 0.0)
    gain_weight = np.divide(-np.expm1(-gain * step), gain,
                            out=np.full(3, step), where=gain != 0.0)
    phi_next = np.exp(-alpha * step) * phi + alpha_weight * forcing
    omega_next = np.exp(-gain * step) * omega + gain_weight * np.maximum(phi, phi_next)
    # Retain exact zero for a zero tube, and round positive endpoints outward.
    phi_next = np.where(phi_next > 0.0, np.nextafter(phi_next, np.inf), 0.0)
    omega_next = np.where(omega_next > 0.0, np.nextafter(omega_next, np.inf), 0.0)
    return phi_next, omega_next


def ancillary_input_bound(A_accel, delta_A, phi, omega, alpha, lambda_gain):
    """Componentwise bound on every term in boundary_layer_feedback.

    For an interval reserve, ``abs(A_accel)`` must bound the nominal acceleration
    matrix throughout that interval. Use nominal_acceleration_bound for this.
    """
    matrix = _matrix(A_accel, "A_accel")
    delta = _matrix(delta_A, "delta_A", nonnegative=True)
    phi = _vector(phi, "phi", nonnegative=True)
    omega = _vector(omega, "omega", nonnegative=True)
    alpha = _vector(alpha, "alpha", nonnegative=True)
    gain = _vector(lambda_gain, "lambda_gain", nonnegative=True)
    velocity = phi + gain * omega
    error = np.concatenate((omega, velocity))
    saturation_gain = delta @ error + alpha * phi
    return np.abs(matrix) @ error + gain * velocity + saturation_gain


def boundary_layer_feedback(A_accel, delta_A, actual, nominal, phi, alpha, lambda_gain):
    """Ancillary input using the explicit nominal model and nonnegative gain."""
    matrix = _matrix(A_accel, "A_accel")
    delta = _matrix(delta_A, "delta_A", nonnegative=True)
    phi = _vector(phi, "phi", nonnegative=True)
    alpha = _vector(alpha, "alpha", nonnegative=True)
    gain = _vector(lambda_gain, "lambda_gain", nonnegative=True)
    actual, nominal = np.asarray(actual, dtype=float), np.asarray(nominal, dtype=float)
    if actual.shape != (6,) or nominal.shape != (6,) or not (
        np.all(np.isfinite(actual)) and np.all(np.isfinite(nominal))
    ):
        raise ValueError("Actual and nominal state must be finite six-vectors")
    error = actual - nominal
    sliding = error[3:] + gain * error[:3]
    saturation = np.sign(sliding)
    np.divide(sliding, phi, out=saturation, where=phi > 0.0)
    saturation = np.clip(saturation, -1.0, 1.0)
    saturation_gain = delta @ np.abs(error) + alpha * phi
    return -matrix @ error - gain * error[3:] - saturation_gain * saturation


def _nominal_context(mean_motion, eccentricity_range, kwargs):
    nominal = kwargs.get("nominal_parameters")
    assumptions = []
    if nominal is None:
        nominal = {
            "eccentricity": float(np.mean(eccentricity_range)),
            "alpha": float(np.mean(kwargs.get("aRange", kwargs.get("a_range", (0.0, 0.0))))),
            "beta": float(np.mean(kwargs.get("bRange", kwargs.get("b_range", (0.0, 0.0))))),
            "theta0": float(kwargs.get("theta0", 0.0)),
        }
        assumptions.append("Legacy midpoint nominal parameters; pass the MPC nominal_parameters explicitly")
    else:
        nominal = dict(nominal)
    mu = float(kwargs.get("mu", nominal.get("mu", 3.986004418e14)))
    semi_major_axis = kwargs.get("semi_major_axis", nominal.get("semi_major_axis"))
    if semi_major_axis is None:
        if mean_motion <= 0.0:
            raise ValueError("Positive mean motion or explicit semi_major_axis is required")
        semi_major_axis = (mu / float(mean_motion)**2)**(1.0 / 3.0)
        assumptions.append("Semi-major axis inferred from mean motion")
    return nominal, float(semi_major_axis), mu, assumptions


def _warn_assumptions(assumptions):
    if assumptions:
        warnings.warn("; ".join(assumptions), RuntimeWarning, stacklevel=3)


def propagate_tube_profile(
    *, start_anom: float, num_steps: int, anom_step: float, mean_motion: float,
    t_periapsis: float = 0.0, lambda_gain: List[float], alpha: List[float],
    phi_0: List[float], eccentricity_range: Tuple[float, float],
    initial_error: np.ndarray = None, eta: float = 0.0, **kwargs,
):
    """Legacy pre-propagation wrapper around the physical tube primitives.

    Supply ``nominal_parameters``, ``nominal_path`` (N by 6), and preferably
    ``nominal_state_bounds`` (N by 6 absolute intersample bounds). Initial tube
    state can be supplied as ``initial_phi`` and ``initial_omega``/``omega_0``.
    Without a path the wrapper uses a zero reference and explicitly labels this
    assumption. Nodewise path samples alone do not certify intersample bounds.
    Coupled nominal/tube optimization should use the primitives directly.
    """
    if int(num_steps) != num_steps or num_steps < 1 or anom_step <= 0.0:
        raise ValueError("Tube horizon requires positive node count and anomaly step")
    num_steps = int(num_steps)
    gain = _vector(lambda_gain, "lambda_gain", nonnegative=True)
    bandwidth = _vector(alpha, "alpha", nonnegative=True)
    nominal, semi_major_axis, mu, assumptions = _nominal_context(
        mean_motion, eccentricity_range, kwargs
    )
    path = kwargs.get("nominal_path", kwargs.get("nominal_trajectory"))
    if path is None:
        path = np.zeros((num_steps, 6))
        assumptions.append("Legacy zero nominal path; tube is not a bound for an unspecified MPC trajectory")
    else:
        path = np.asarray(path, dtype=float)
        if path.shape == (6, num_steps) and path.shape != (num_steps, 6):
            path = path.T
        if path.shape != (num_steps, 6) or not np.all(np.isfinite(path)):
            raise ValueError("nominal_path must have shape (num_steps, 6)")
    state_bounds = kwargs.get("nominal_state_bounds")
    if state_bounds is None:
        state_bounds = np.abs(path)
        if num_steps > 1:
            state_bounds[:-1] = np.maximum(state_bounds[:-1], np.abs(path[1:]))
        assumptions.append("Nominal path endpoint envelope assumes no intersample excursion")
    else:
        state_bounds = np.asarray(state_bounds, dtype=float)
        if state_bounds.shape != (num_steps, 6) or np.any(state_bounds < 0.0) or not np.all(np.isfinite(state_bounds)):
            raise ValueError("nominal_state_bounds must be nonnegative N by 6 bounds")
    assumptions.append("Continuous ancillary feedback assumed; sample-hold and integration residuals require separate bounds")
    _warn_assumptions(assumptions)
    phi, omega = np.zeros((num_steps, 3)), np.zeros((num_steps, 3))
    phi[0] = _vector(kwargs.get("initial_phi", phi_0), "initial_phi", nonnegative=True)
    omega[0] = _vector(kwargs.get("initial_omega", kwargs.get("omega_0", np.zeros(3))),
                       "initial_omega", nonnegative=True)
    if initial_error is not None:
        error = np.asarray(initial_error, dtype=float)
        if error.shape != (6,) or not np.all(np.isfinite(error)):
            raise ValueError("initial_error must be a finite six-vector")
        omega[0] = np.maximum(omega[0], np.abs(error[:3]))
        phi[0] = np.maximum(phi[0], np.abs(error[3:] + gain * error[:3]))
    eta_vec = np.broadcast_to(np.asarray(eta, dtype=float), (3,))
    if not np.all(np.isfinite(eta_vec)) or np.any(eta_vec < 0.0):
        raise ValueError("eta must be finite and nonnegative")
    time_grid = float(start_anom) + float(anom_step) * np.arange(num_steps)
    reserves = np.zeros((num_steps, 3))
    for index, q_start in enumerate(time_grid):
        q_end = time_grid[min(index + 1, num_steps - 1)]
        delta_a, delta_d = uncertainty_envelope(
            q_start, q_end, nominal, eccentricity_range,
            kwargs.get("aRange", kwargs.get("a_range", (0.0, 0.0))),
            kwargs.get("bRange", kwargs.get("b_range", (0.0, 0.0))), semi_major_axis, mu,
        )
        if index + 1 < num_steps:
            forcing = delta_a @ state_bounds[index] + delta_d + eta_vec
            phi[index + 1], omega[index + 1] = advance_tube(
                phi[index], omega[index], bandwidth, gain, forcing, anom_step
            )
        end_index = min(index + 1, num_steps - 1)
        reserves[index] = ancillary_input_bound(
            nominal_acceleration_bound(q_start, q_end, nominal, semi_major_axis), delta_a,
            np.maximum(phi[index], phi[end_index]), np.maximum(omega[index], omega[end_index]),
            bandwidth, gain,
        )
    return TubeProfile(phi, omega, time_grid, gain, bandwidth, reserves, tuple(assumptions))


def tighten_box_bounds(bounds, tightening):
    """Symmetric box tightening; infeasible bounds remain explicitly empty."""
    if bounds is None:
        return None
    tightening = np.asarray(tightening, dtype=float)
    if np.any(tightening < 0.0) or not np.all(np.isfinite(tightening)):
        raise ValueError("Tightening must be finite and nonnegative")
    tightened = []
    for index, bound in enumerate(bounds):
        amount = float(tightening[index]) if index < len(tightening) else 0.0
        new_bound = dict(bound)
        if new_bound.get("lower") not in ("-Inf", None, float("-inf")):
            new_bound["lower"] = float(new_bound["lower"]) + amount
        if new_bound.get("upper") not in ("+Inf", None, float("inf")):
            new_bound["upper"] = float(new_bound["upper"]) - amount
        tightened.append(new_bound)
    return tightened


def input_tightening_from_profile(tube_profile, lambda_gain=None):
    """Return stored ancillary input reserves, never a velocity-error proxy."""
    if tube_profile.input_reserve is None:
        raise ValueError("Tube profile must contain ancillary input_reserve bounds")
    reserve = np.asarray(tube_profile.input_reserve, dtype=float)
    if reserve.ndim == 1:
        return _vector(reserve, "input_reserve", nonnegative=True).copy()
    node_count = len(tube_profile.phi)
    if reserve.shape not in ((node_count, 3), (node_count - 1, 3)) or np.any(reserve < 0.0) or not np.all(np.isfinite(reserve)):
        raise ValueError("Invalid ancillary input reserves")
    return np.max(reserve, axis=0)


def tighten_target_params(target_params, tube_profile):
    if not target_params:
        return target_params
    tightened = dict(target_params)
    tightened["tube_radius"] = max(float(tightened.get("tube_radius", 0.0)),
                                    tube_profile.max_position_radius)
    return tightened


def ancillary_controller(
    t, t_p, t_f, dt, mean_motion, nom_state, act_state, lambda_gain, alpha, phi,
    eccentricity_range, *args, **kwargs,
):
    """Compatibility interface; pass the exact MPC nominal_parameters or A_nom_val."""
    nominal, semi_major_axis, mu, assumptions = _nominal_context(
        mean_motion, eccentricity_range, kwargs
    )
    _warn_assumptions(assumptions)
    matrix = kwargs.get("A_nom_val")
    if matrix is None:
        matrices, _, _ = orbital_ellp_drag(
            anom_step=dt, mean_motion=mean_motion,
            eccentricity=nominal["eccentricity"], alpha=nominal["alpha"], beta=nominal["beta"],
            mu=mu, semi_major_axis=semi_major_axis, theta0=nominal.get("theta0", 0.0),
        )
        matrix = np.asarray(matrices[0](t, t_p), dtype=float)[3:]
    else:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape == (6, 6):
            matrix = matrix[3:]
    delta_a = kwargs.get("delta_A")
    if delta_a is None:
        delta_a, _ = uncertainty_envelope(
            t, max(t, t_f), nominal, eccentricity_range,
            kwargs.get("aRange", kwargs.get("a_range", (0.0, 0.0))),
            kwargs.get("bRange", kwargs.get("b_range", (0.0, 0.0))), semi_major_axis, mu,
        )
    return boundary_layer_feedback(matrix, delta_a, act_state, nom_state, phi, alpha, lambda_gain)


def calc_delta(t, t_p, x, anom_step, mean_motion, *args, **kwargs):
    """Compatibility componentwise bound ``delta_A @ abs(x)`` for model error."""
    e_range = kwargs.get("e_range", kwargs.get("eccentricity_range", (0.0, 0.0)))
    nominal, semi_major_axis, mu, assumptions = _nominal_context(mean_motion, e_range, kwargs)
    _warn_assumptions(assumptions)
    delta_a, _ = uncertainty_envelope(
        t, t, nominal, e_range, kwargs.get("a_range", kwargs.get("aRange", (0.0, 0.0))),
        kwargs.get("b_range", kwargs.get("bRange", (0.0, 0.0))), semi_major_axis, mu,
    )
    state = np.asarray(x, dtype=float)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("Uncertainty state must be a finite six-vector")
    return delta_a @ np.abs(state)


def calc_d(t, t_p, dt, mean_motion, *args, **kwargs):
    """Compatibility nonnegative componentwise affine-disturbance bound."""
    e_range = kwargs.get("e_range", kwargs.get("eccentricity_range", (0.0, 0.0)))
    nominal, semi_major_axis, mu, assumptions = _nominal_context(mean_motion, e_range, kwargs)
    _warn_assumptions(assumptions)
    _, delta_d = uncertainty_envelope(
        t, t, nominal, e_range, kwargs.get("a_range", kwargs.get("aRange", (0.0, 0.0))),
        kwargs.get("b_range", kwargs.get("bRange", (0.0, 0.0))), semi_major_axis, mu,
    )
    return delta_d
