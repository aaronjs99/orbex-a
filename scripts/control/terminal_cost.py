"""Local Riccati penalty for a finite MPC horizon."""

import numpy as np
from scipy.linalg import expm, solve_discrete_are


def local_terminal_cost(A, B, Q, R, step: float, *, multiplier: float = 1.0) -> np.ndarray:
    """Compute a terminal penalty from a frozen local LQR model.

    The exact zero-order-hold discretization is independent of the OCP backend.
    Q and R are weighted by the anomaly step to match the running objective.
    ``multiplier`` is an explicit optional tuning weight on that matched tail.
    This terminal penalty does not establish nonlinear recursive feasibility;
    the mission's constraints and success checks remain authoritative.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("Terminal cost requires a positive finite prediction step")
    if not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError("Terminal cost multiplier must be finite and nonnegative")
    states, inputs = B.shape
    augmented = np.zeros((states + inputs, states + inputs), dtype=float)
    augmented[:states, :states] = A
    augmented[:states, states:] = B
    transition = expm(augmented * step)
    penalty = solve_discrete_are(
        transition[:states, :states], transition[:states, states:], step * Q, step * R
    )
    return float(multiplier) * 0.5 * (penalty + penalty.T)
