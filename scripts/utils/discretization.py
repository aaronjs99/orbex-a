# /***********************************************************
# *                                                         *
# * Copyright (c) 2026                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu                                *
# * Contact: aaronjs@ucla.edu                               *
# *                                                         *
# ***********************************************************/

"""Continuous-to-discrete linear state-space conversion."""

from typing import Optional, Tuple
import numpy as np
from scipy import signal


def discretize(
    dt: float,
    A: np.ndarray,
    B: np.ndarray,
    C: Optional[np.ndarray] = None,
    D: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a continuous-time system to discrete-time.

    Args:
        dt (float): Sampling time.
        A (np.ndarray): Continuous-time A matrix.
        B (np.ndarray): Continuous-time B matrix.
        C (np.ndarray, optional): Output matrix.
        D (np.ndarray, optional): Feedthrough matrix.

    Returns:
        tuple: (A_d, B_d, C_d, D_d)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    n_states = A.shape[0]
    n_inputs = B.shape[1]

    if C is None:
        C = np.zeros((1, n_states))
        C[0, 0] = 1.0
    else:
        C = np.asarray(C, dtype=float)

    if D is None:
        D = np.zeros((C.shape[0], n_inputs))
    else:
        D = np.asarray(D, dtype=float)

    # Scipy returns 5 args: Ad, Bd, Cd, Dd, dt
    Ad, Bd, Cd, Dd, _ = signal.cont2discrete((A, B, C, D), dt)

    return Ad, Bd, Cd, Dd
