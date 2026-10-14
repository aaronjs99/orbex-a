"""Sample moving state goals and their nominal holding inputs for MPC."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def sample_tracking_reference(
    anomalies: np.ndarray,
    state_at: Callable[[float], np.ndarray],
    derivative_at: Callable[[float], np.ndarray],
    dynamics_at: Callable[[float], np.ndarray],
    input_matrix: np.ndarray,
    disturbance_at: Callable[[float], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return state samples and one feedforward input per prediction interval.

    The derivative uses the same independent variable as the dynamics (true
    anomaly for the orbital mission). Each input balances the nominal dynamics
    at the interval midpoint. Bounds remain the solver's responsibility; this
    reference does not assert that a requested trajectory is feasible.
    """
    anomalies = np.asarray(anomalies, dtype=float)
    if (
        anomalies.ndim != 1
        or len(anomalies) < 2
        or not np.all(np.isfinite(anomalies))
        or np.any(np.diff(anomalies) <= 0.0)
    ):
        raise ValueError("A tracking reference needs at least two increasing finite samples")
    states = np.column_stack([state_at(float(value)) for value in anomalies])
    input_matrix = np.asarray(input_matrix, dtype=float)
    inputs = []
    for value in 0.5 * (anomalies[:-1] + anomalies[1:]):
        state = np.asarray(state_at(float(value)), dtype=float)
        derivative = np.asarray(derivative_at(float(value)), dtype=float)
        drift = np.asarray(dynamics_at(float(value)), dtype=float) @ state
        drift += np.asarray(disturbance_at(float(value)), dtype=float)
        control, _, _, _ = np.linalg.lstsq(input_matrix, derivative - drift, rcond=None)
        inputs.append(control)
    return states, np.column_stack(inputs)
