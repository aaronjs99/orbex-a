"""First-exit comparison bounds for the implemented held ancillary controller.

All vector inequalities are componentwise. ``A_variation`` bounds a change in
the nominal acceleration matrix over any feedback hold; a full outer-interval
coefficient width is sufficient and is already a change, not a derivative.
The bounded-term construction is valid for every nonnegative sliding width,
including zero. It makes no small-layer division or sampling smallness claim.
"""
from __future__ import annotations

import numpy as np


TUBE_MODEL_VERSION = "held-ancillary-bounded-v1"
TUBE_ASSUMPTIONS = (
    "Shared exact anomaly/phase, input matrix and fixed semi-major axis; actual parameters stay in the supplied box during a hold",
    "Exact continuous model with held ancillary input, nonnegative stage peaks and the configured maximum feedback hold width",
    "Extra acceleration reserve is an assumed disturbance bound; no derived RK4 or floating-point remainder is supplied",
    "Nominal continuous intersample enclosure starts at the shooting node; dense/outer nominal-center defect is reported separately",
    "This tube model alone supplies sampled geometry and endpoint acceptance; swept safety, invariant terminal tracking and stored continuation require their separately admitted certificate layers",
)


def _nonnegative(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"{name} must be finite, nonnegative and have shape {shape}")
    return result


def sample_hold_expressions(A_abs, delta_A, A_variation, base_forcing,
                            phi_peak, omega_peak, alpha, lambda_gain,
                            hold_width, *, intermediate=lambda value: value):
    """Build the bound with numeric or backend scalar expressions.

    Backend objects appear only in the four stage vectors. ``intermediate``
    lets a backend retain the same algebra without expanding nested terms.
    Numeric callers should use :func:`sample_hold_bounds` for value checks.

    V bounds error velocity, E bounds complete error, and R reserves the full
    held feedback. Dv, De and Ds bound derivatives until a first tube exit.
    The residual rho bounds the two uncertain-error terms separately and the
    changes in nominal cancellation, Lambda feedback and kappa damping.
    """
    a = _nonnegative(A_abs, (3, 6), "A_abs")
    delta = _nonnegative(delta_A, (3, 6), "delta_A")
    variation = _nonnegative(A_variation, (3, 6), "A_variation")
    lam = _nonnegative(lambda_gain, (3,), "lambda_gain")
    tau = float(hold_width)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("Feedback hold width must be positive and finite")
    for name, value in (("base_forcing", base_forcing), ("phi_peak", phi_peak),
                        ("omega_peak", omega_peak), ("alpha", alpha)):
        if len(value) != 3:
            raise ValueError(f"{name} must contain three components")

    def wrap(values):
        return [intermediate(value) for value in values]

    def dot(matrix, vector):
        return [sum(float(coefficient) * value for coefficient, value in zip(row, vector)
                    if coefficient != 0) for row in matrix]

    velocity = wrap([phi_peak[i] + float(lam[i]) * omega_peak[i] for i in range(3)])
    error = list(omega_peak) + velocity
    drift = dot(a + delta, error)
    uncertain_error = dot(delta, error)
    reserve = wrap([drift[i] + float(lam[i]) * velocity[i] + alpha[i] * phi_peak[i]
                    for i in range(3)])
    dv = wrap([drift[i] + base_forcing[i] + reserve[i] for i in range(3)])
    de = velocity + dv
    ds = wrap([dv[i] + float(lam[i]) * velocity[i] for i in range(3)])
    coefficient_change = dot(variation, error)
    cancellation_change = dot(a, de)
    rho = wrap([2 * uncertain_error[i] + coefficient_change[i]
                + tau * (cancellation_change[i] + float(lam[i]) * dv[i] + alpha[i] * ds[i])
                for i in range(3)])
    total = wrap([base_forcing[i] + rho[i] for i in range(3)])
    return dict(base_forcing=list(base_forcing), hold_forcing=rho,
                disturbance_bound=total, velocity_peak=velocity, error_peak=error,
                input_reserve=reserve, velocity_derivative_bound=dv,
                error_derivative_bound=de, sliding_derivative_bound=ds)


def sample_hold_bounds(A_abs, delta_A, A_variation, base_forcing,
                       phi_peak, omega_peak, alpha, lambda_gain, hold_width):
    """Evaluate the bounded-term hold correction with explicit shape checks."""
    values = [_nonnegative(value, (3,), name) for name, value in (
        ("base_forcing", base_forcing), ("phi_peak", phi_peak),
        ("omega_peak", omega_peak), ("alpha", alpha))]
    terms = sample_hold_expressions(A_abs, delta_A, A_variation, *values,
                                   lambda_gain, hold_width)
    result = {name: np.asarray(value, dtype=float) for name, value in terms.items()}
    if not all(np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError("Non-finite held-input tube bound")
    return result
