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
physical LVLH dynamics. Coupled planning includes the derived held-input forcing
from sample_hold; numerical integration error still requires a separate bound.
"""

from dataclasses import dataclass
from typing import List, Tuple
import warnings

import numpy as np

from orbexa.core.dynamics.orbital import orbital_ellp_drag
from orbexa.control.tubes.uncertainty import uncertainty_envelope, nominal_acceleration_bound
from orbexa.control.tubes.sample_hold import sample_hold_bounds


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
    tube_model_version: str = "continuous-feedback-legacy"
    hold_width: float = None
    forcing: dict = None

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


def advance_tube(phi, omega, alpha, lambda_gain, disturbance, step, *, phi_peak=None):
    """Positive exponential step for constant bandwidth and forcing bounds.

    ``disturbance`` bounds total acceleration forcing, including the hold
    correction when feedback is sampled. Phi is integrated exactly for it.
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
    peak = np.maximum(phi, phi_next) if phi_peak is None else _vector(phi_peak, "phi_peak", nonnegative=True)
    omega_next = np.exp(-gain * step) * omega + gain_weight * peak
    # Retain exact zero for a zero tube, and round positive endpoints outward.
    phi_next = np.where(phi_next > 0.0, np.nextafter(phi_next, np.inf), 0.0)
    omega_next = np.where(omega_next > 0.0, np.nextafter(omega_next, np.inf), 0.0)
    return phi_next, omega_next


def nominal_reset_forcing(alpha,lambda_gain,endpoint_error_abs,hold_width):
    """Prediction allowance for certified nominal-center endpoint transfers.

    A constant comparison forcing integrates to the declared positive jump
    over one dense hold. This is a numerical-center reset budget, separate
    from the physical acceleration forcing and the ancillary input reserve.
    """
    alpha=_vector(alpha,'alpha',nonnegative=True)
    gain=_vector(lambda_gain,'lambda_gain',nonnegative=True)
    error=np.asarray(endpoint_error_abs,dtype=float)
    if error.shape!=(6,) or not np.all(np.isfinite(error)) or np.any(error<0):
        raise ValueError('Nominal endpoint transfer needs a nonnegative finite six-vector')
    width=float(hold_width)
    if not np.isfinite(width) or width<=0:raise ValueError('Positive endpoint-reset hold width required')
    wa=np.divide(-np.expm1(-alpha*width),alpha,out=np.full(3,width),where=alpha!=0)
    wl=np.divide(-np.expm1(-gain*width),gain,out=np.full(3,width),where=gain!=0)
    return (error[3:]+gain*error[:3])/wa,error[:3]/wl


def held_tube_stage_seed(phi, omega, alpha, lambda_gain, A_abs, delta_A,
                         A_variation, base_forcing, step, hold_width, *,
                         measurement_error_abs=None,nominal_endpoint_error_abs=None):
    """Find a consistent numeric stage seed for the coupled held-input bound.

    For fixed gains the nonnegative forcing is affine in the two stage peaks.
    Monotone iteration from the initial bounds seeks its least fixed point.
    Failure to obtain a finite fixed point is explicit; no margin is clipped.
    This initializes the NLP and does not replace its coupled constraints.
    """
    phi = _vector(phi, "phi", nonnegative=True)
    omega = _vector(omega, "omega", nonnegative=True)
    peak_phi, peak_omega = phi.copy(), omega.copy()
    reset_phi,reset_omega=np.zeros(3),np.zeros(3)
    if nominal_endpoint_error_abs is not None and np.any(nominal_endpoint_error_abs):
        reset_phi,reset_omega=nominal_reset_forcing(alpha,lambda_gain,nominal_endpoint_error_abs,hold_width)
    def advance(terms):
        if not np.any(reset_phi) and not np.any(reset_omega):
            return advance_tube(phi,omega,alpha,lambda_gain,terms['disturbance_bound'],step,phi_peak=peak_phi)
        next_phi,next_omega=advance_tube(phi,omega,alpha,lambda_gain,
            terms['disturbance_bound']+reset_phi,step,phi_peak=peak_phi)
        gain=np.asarray(lambda_gain,dtype=float)
        weight=np.divide(-np.expm1(-gain*float(step)),gain,out=np.full(3,float(step)),where=gain!=0)
        return next_phi,np.nextafter(next_omega+weight*reset_omega,np.inf)
    for _ in range(200):
        terms = sample_hold_bounds(A_abs, delta_A, A_variation, base_forcing,
            peak_phi, peak_omega, alpha, lambda_gain, hold_width,
            measurement_error_abs=measurement_error_abs)
        next_phi, next_omega = advance(terms)
        new_phi, new_omega = np.maximum(phi, next_phi), np.maximum(omega, next_omega)
        change = max(np.max(np.abs(new_phi - peak_phi)), np.max(np.abs(new_omega - peak_omega)))
        peak_phi, peak_omega = new_phi, new_omega
        if change <= 1e-13 * (1 + max(np.max(peak_phi), np.max(peak_omega))):
            terms = sample_hold_bounds(A_abs, delta_A, A_variation, base_forcing,
                peak_phi, peak_omega, alpha, lambda_gain, hold_width,
                measurement_error_abs=measurement_error_abs)
            next_phi, next_omega = advance(terms)
            return next_phi, next_omega, peak_phi, peak_omega, terms
    raise ValueError("Held-input tube stage seed has no converged finite fixed point at the supplied gains")


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
