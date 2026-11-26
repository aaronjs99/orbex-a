# /***********************************************************
# * Copyright (c) 2026                                      *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# * Authors: Aaron John Sabu, Brett T. Lopez                *
# * Contact: {aaronjs, btlopez}@ucla.edu                    *
# ***********************************************************/

"""model.py: Solver-independent boundary-layer feedback and dynamic tube arithmetic.

Phi bounds the sliding error ``e_v + Lambda e_p``; Omega bounds position error.
The nominal model and its parameter-uncertainty envelope must refer to the same
physical LVLH dynamics. Coupled planning includes the derived held-input forcing
from sample_hold; numerical integration error still requires a separate bound.

Copyright: Aaron John Sabu
"""

from dataclasses import dataclass

import numpy as np
from orbexa.control.tubes.sample_hold import sample_hold_bounds


def _vector(value, name, *, nonnegative=False):
    """Return value as a float 3-vector; raise ValueError if not finite or, when
    requested, if any component is negative.
    """
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _matrix(value, name, *, nonnegative=False):
    """Return value as a float 3 by 6 matrix; raise ValueError if not finite or, when
    requested, if any entry is negative.
    """
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
    assumptions: tuple[str, ...] = ()
    tube_model_version: str = "continuous-feedback-legacy"
    hold_width: float = None
    forcing: dict = None

    @property
    def position_radius(self):
        """Return the Euclidean norm of omega at each node."""
        return np.linalg.norm(self.omega, axis=1)

    @property
    def max_position_radius(self):
        """Return the largest per-node position radius as a float."""
        return float(np.max(self.position_radius))

    @property
    def velocity_error_bound(self):
        """Return phi + |lambda_gain| * omega per node.

        Raises ValueError if the profile has no lambda_gain.
        """
        if self.lambda_gain is None:
            raise ValueError(
                "Velocity error bounds require the profile's sliding gains"
            )
        return np.asarray(self.phi) + np.abs(self.lambda_gain) * np.asarray(self.omega)

    @property
    def max_state_error(self):
        """Return the six-vector of maximum omega per axis then maximum velocity error
        bound per axis over all nodes.
        """
        return np.concatenate(
            (np.max(self.omega, axis=0), np.max(self.velocity_error_bound, axis=0))
        )


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
    alpha_weight = np.divide(
        -np.expm1(-alpha * step), alpha, out=np.full(3, step), where=alpha != 0.0
    )
    gain_weight = np.divide(
        -np.expm1(-gain * step), gain, out=np.full(3, step), where=gain != 0.0
    )
    phi_next = np.exp(-alpha * step) * phi + alpha_weight * forcing
    peak = (
        np.maximum(phi, phi_next)
        if phi_peak is None
        else _vector(phi_peak, "phi_peak", nonnegative=True)
    )
    omega_next = np.exp(-gain * step) * omega + gain_weight * peak
    # Retain exact zero for a zero tube, and round positive endpoints outward.
    phi_next = np.where(phi_next > 0.0, np.nextafter(phi_next, np.inf), 0.0)
    omega_next = np.where(omega_next > 0.0, np.nextafter(omega_next, np.inf), 0.0)
    return phi_next, omega_next


def nominal_reset_forcing(alpha, lambda_gain, endpoint_error_abs, hold_width):
    """Prediction allowance for certified nominal-center endpoint transfers.

    A constant comparison forcing integrates to the declared positive jump
    over one dense hold. This is a numerical-center reset budget, separate
    from the physical acceleration forcing and the ancillary input reserve.
    """
    alpha = _vector(alpha, "alpha", nonnegative=True)
    gain = _vector(lambda_gain, "lambda_gain", nonnegative=True)
    error = np.asarray(endpoint_error_abs, dtype=float)
    if error.shape != (6,) or not np.all(np.isfinite(error)) or np.any(error < 0):
        raise ValueError(
            "Nominal endpoint transfer needs a nonnegative finite six-vector"
        )
    width = float(hold_width)
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Positive endpoint-reset hold width required")
    wa = np.divide(
        -np.expm1(-alpha * width), alpha, out=np.full(3, width), where=alpha != 0
    )
    wl = np.divide(
        -np.expm1(-gain * width), gain, out=np.full(3, width), where=gain != 0
    )
    return (error[3:] + gain * error[:3]) / wa, error[:3] / wl


def held_tube_stage_seed(
    phi,
    omega,
    alpha,
    lambda_gain,
    A_abs,
    delta_A,
    A_variation,
    base_forcing,
    step,
    hold_width,
    *,
    measurement_error_abs=None,
    nominal_endpoint_error_abs=None,
):
    """Find a consistent numeric stage seed for the coupled held-input bound.

    For fixed gains the nonnegative forcing is affine in the two stage peaks.
    Monotone iteration from the initial bounds seeks its least fixed point.
    Failure to obtain a finite fixed point is explicit; no margin is clipped.
    This initializes the NLP and does not replace its coupled constraints.
    """
    phi = _vector(phi, "phi", nonnegative=True)
    omega = _vector(omega, "omega", nonnegative=True)
    peak_phi, peak_omega = phi.copy(), omega.copy()
    reset_phi, reset_omega = np.zeros(3), np.zeros(3)
    if nominal_endpoint_error_abs is not None and np.any(nominal_endpoint_error_abs):
        reset_phi, reset_omega = nominal_reset_forcing(
            alpha, lambda_gain, nominal_endpoint_error_abs, hold_width
        )

    def advance(terms):
        """Advance (phi, omega) one step for the given forcing terms.

        Uses terms["disturbance_bound"] (plus the nominal reset forcing when nonzero)
        and the enclosing peak_phi. When reset forcing is nonzero, omega also gains
        the lambda_gain weight times reset_omega, rounded upward.

        Outputs: (next_phi, next_omega).
        """
        if not np.any(reset_phi) and not np.any(reset_omega):
            return advance_tube(
                phi,
                omega,
                alpha,
                lambda_gain,
                terms["disturbance_bound"],
                step,
                phi_peak=peak_phi,
            )
        next_phi, next_omega = advance_tube(
            phi,
            omega,
            alpha,
            lambda_gain,
            terms["disturbance_bound"] + reset_phi,
            step,
            phi_peak=peak_phi,
        )
        gain = np.asarray(lambda_gain, dtype=float)
        weight = np.divide(
            -np.expm1(-gain * float(step)),
            gain,
            out=np.full(3, float(step)),
            where=gain != 0,
        )
        return next_phi, np.nextafter(next_omega + weight * reset_omega, np.inf)

    for _ in range(200):
        terms = sample_hold_bounds(
            A_abs,
            delta_A,
            A_variation,
            base_forcing,
            peak_phi,
            peak_omega,
            alpha,
            lambda_gain,
            hold_width,
            measurement_error_abs=measurement_error_abs,
        )
        next_phi, next_omega = advance(terms)
        new_phi, new_omega = np.maximum(phi, next_phi), np.maximum(omega, next_omega)
        change = max(
            np.max(np.abs(new_phi - peak_phi)), np.max(np.abs(new_omega - peak_omega))
        )
        peak_phi, peak_omega = new_phi, new_omega
        if change <= 1e-13 * (1 + max(np.max(peak_phi), np.max(peak_omega))):
            terms = sample_hold_bounds(
                A_abs,
                delta_A,
                A_variation,
                base_forcing,
                peak_phi,
                peak_omega,
                alpha,
                lambda_gain,
                hold_width,
                measurement_error_abs=measurement_error_abs,
            )
            next_phi, next_omega = advance(terms)
            return next_phi, next_omega, peak_phi, peak_omega, terms
    raise ValueError(
        "Held-input tube stage seed has no converged finite fixed point at the supplied gains"
    )


def boundary_layer_feedback(A_accel, delta_A, actual, nominal, phi, alpha, lambda_gain):
    """Ancillary input using the explicit nominal model and nonnegative gain."""
    matrix = _matrix(A_accel, "A_accel")
    delta = _matrix(delta_A, "delta_A", nonnegative=True)
    phi = _vector(phi, "phi", nonnegative=True)
    alpha = _vector(alpha, "alpha", nonnegative=True)
    gain = _vector(lambda_gain, "lambda_gain", nonnegative=True)
    actual, nominal = np.asarray(actual, dtype=float), np.asarray(nominal, dtype=float)
    if (
        actual.shape != (6,)
        or nominal.shape != (6,)
        or not (np.all(np.isfinite(actual)) and np.all(np.isfinite(nominal)))
    ):
        raise ValueError("Actual and nominal state must be finite six-vectors")
    error = actual - nominal
    sliding = error[3:] + gain * error[:3]
    saturation = np.sign(sliding)
    np.divide(sliding, phi, out=saturation, where=phi > 0.0)
    saturation = np.clip(saturation, -1.0, 1.0)
    saturation_gain = delta @ np.abs(error) + alpha * phi
    return -matrix @ error - gain * error[3:] - saturation_gain * saturation


def input_tightening_from_profile(tube_profile, lambda_gain=None):
    """Return stored ancillary input reserves, never a velocity-error proxy."""
    if tube_profile.input_reserve is None:
        raise ValueError("Tube profile must contain ancillary input_reserve bounds")
    reserve = np.asarray(tube_profile.input_reserve, dtype=float)
    if reserve.ndim == 1:
        return _vector(reserve, "input_reserve", nonnegative=True).copy()
    node_count = len(tube_profile.phi)
    if (
        reserve.shape not in ((node_count, 3), (node_count - 1, 3))
        or np.any(reserve < 0.0)
        or not np.all(np.isfinite(reserve))
    ):
        raise ValueError("Invalid ancillary input reserves")
    return np.max(reserve, axis=0)
