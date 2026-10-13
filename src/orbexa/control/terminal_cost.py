"""Local Riccati penalty for a finite MPC horizon."""

import numpy as np
from scipy.linalg import expm, solve_discrete_are


def local_terminal_cost(A, B, Q, R, step: float) -> np.ndarray:
    """Compute a terminal penalty from a frozen local LQR model.

    The exact zero-order-hold discretization is independent of the OCP backend.
    This terminal penalty does not establish nonlinear recursive feasibility;
    the mission's constraints and success checks remain authoritative.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("Terminal cost requires a positive finite prediction step")
    states, inputs = B.shape
    augmented = np.zeros((states + inputs, states + inputs), dtype=float)
    augmented[:states, :states] = A
    augmented[:states, states:] = B
    transition = expm(augmented * step)
    penalty = solve_discrete_are(
        transition[:states, :states], transition[:states, states:], Q, R
    )
    return 0.5 * (penalty + penalty.T)
