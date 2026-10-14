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

"""Recorded-window prediction, observable point fitting and fit metrics."""

import logging
import time
import numpy as np
import queue
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional, Union
from scipy.optimize import least_squares
from orbexa.core.dynamics import orbital_ellp_drag

logger = logging.getLogger(__name__)


@dataclass
class ParameterPredictor:
    parameter_keys: Tuple[str, ...]
    error_bound: float
    time_series_prior_weight: float
    observability_absolute_threshold: float
    observability_relative_threshold: float
    observability_unique_threshold: float

    def _blend_estimates(
        self,
        previous: Dict[str, float],
        refined: Dict[str, float],
        feasible_sets: Dict[str, Tuple[float, float]],
        gain: float,
    ) -> Optional[Dict[str, float]]:
        """Scale one update direction to the box without changing its nullspace."""
        lower_step, upper_step = 0.0, float(np.clip(gain, 0.0, 1.0))
        for key in self.parameter_keys:
            start = float(previous[key])
            delta = float(refined[key]) - start
            lower, upper = feasible_sets[key]
            if not np.all(np.isfinite((start, delta, lower, upper))):
                return None
            if delta == 0.0:
                if not lower <= start <= upper:
                    return None
                continue
            endpoints = sorted(((lower - start) / delta, (upper - start) / delta))
            lower_step = max(lower_step, endpoints[0])
            upper_step = min(upper_step, endpoints[1])
        if not np.isfinite(upper_step) or lower_step > upper_step:
            return None
        if upper_step > lower_step:
            upper_step = float(np.nextafter(upper_step, lower_step))
        blended = {
            key: float(previous[key])
            + upper_step * (float(refined[key]) - float(previous[key]))
            for key in self.parameter_keys
        }
        if any(not feasible_sets[key][0] <= blended[key] <= feasible_sets[key][1]
               for key in self.parameter_keys):
            return None
        return blended

    @staticmethod
    def _rk4_step(
        state: np.ndarray,
        control: np.ndarray,
        *,
        anom: float,
        anom_step: float,
        matrices,
        time_periapsis: float,
    ) -> np.ndarray:
        """Propagate one recorded constant-input interval like the mission plant."""
        A_func, B_func, _, _, d_func = matrices
        B_val = np.asarray(B_func(), dtype=float)
        state = np.asarray(state, dtype=float)
        control = np.asarray(control, dtype=float)

        def derivative(local_state, local_anom):
            A_val = np.asarray(A_func(local_anom, time_periapsis), dtype=float)
            d_val = np.asarray(d_func(local_anom, time_periapsis), dtype=float)
            return A_val @ local_state + B_val @ control + d_val

        k1 = derivative(state, anom)
        k2 = derivative(state + 0.5 * anom_step * k1, anom + 0.5 * anom_step)
        k3 = derivative(state + 0.5 * anom_step * k2, anom + 0.5 * anom_step)
        k4 = derivative(state + anom_step * k3, anom + anom_step)
        return state + (anom_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _state_residuals(
        self,
        *,
        states: np.ndarray,
        controls: np.ndarray,
        start_anom: float,
        anom_step: float,
        parameters: Dict[str, float],
        dynamics_context: Dict[str, Any],
    ) -> np.ndarray:
        """Weighted physical one-step residuals, excluding the regularizing prior."""
        params = dict(dynamics_context)
        params.update(parameters)
        params["specific_angular_momentum"] = None
        matrices, _, _ = orbital_ellp_drag(anom_step=anom_step, **params)
        pieces = []
        for idx in range(len(states) - 1):
            predicted = self._rk4_step(
                states[idx], controls[idx],
                anom=start_anom + idx * anom_step,
                anom_step=anom_step, matrices=matrices,
                time_periapsis=params.get("time_periapsis", 0.0),
            )
            error = np.asarray(states[idx + 1] - predicted, dtype=float)
            pieces.extend((error[:3], 2.0 * error[3:]))
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=float)

    def _point_metrics(
        self,
        *,
        parameters: Dict[str, float],
        states: np.ndarray,
        controls: np.ndarray,
        start_anom: float,
        anom_step: float,
        dynamics_context: Dict[str, Any],
    ) -> Tuple[bool, float, float]:
        """Verify the actual point returned to MPC, independently of a set witness."""
        point_prediction_error = self.prediction_error(
            parameters=parameters, states=states, controls=controls,
            start_anom=start_anom, anom_step=anom_step,
            dynamics_context=dynamics_context,
        )
        residual = self._state_residuals(
            parameters=parameters, states=states, controls=controls,
            start_anom=start_anom, anom_step=anom_step,
            dynamics_context=dynamics_context,
        )
        point_fit_error = float(np.sqrt(np.mean(np.square(residual))))
        valid = bool(
            np.isfinite(point_prediction_error)
            and np.isfinite(point_fit_error)
            and point_prediction_error <= self.error_bound
            and point_fit_error <= self.error_bound
        )
        return valid, point_prediction_error, point_fit_error

    def prediction_error(
        self,
        *,
        states: np.ndarray,
        controls: np.ndarray,
        start_anom: float,
        anom_step: float,
        parameters: Dict[str, float],
        dynamics_context: Dict[str, Any],
    ) -> float:
        predicted = np.asarray(states[0], dtype=float).copy()
        params = dict(dynamics_context)
        params.update(parameters)
        params["specific_angular_momentum"] = None
        matrices, _, _ = orbital_ellp_drag(anom_step=anom_step, **params)
        for idx in range(len(states) - 1):
            predicted = self._rk4_step(
                predicted, controls[idx],
                anom=start_anom + idx * anom_step,
                anom_step=anom_step, matrices=matrices,
                time_periapsis=params.get("time_periapsis", 0.0),
            )
        return float(np.linalg.norm(predicted - states[-1]))

    def _time_series_estimate(
        self,
        *,
        feasible_sets: Dict[str, Tuple[float, float]],
        estimates: Dict[str, float],
        states: np.ndarray,
        controls: np.ndarray,
        start_anom: float,
        anom_step: float,
        dynamics_context: Dict[str, Any],
    ) -> Tuple[Dict[str, float], Optional[float], str]:
        """Fit a point belief from the measured time series when data supports it.

        The residual includes a small prior term for numerical regularization, but
        observability is assessed only on the physical state residual.  This avoids
        treating the prior itself as evidence that alpha/beta were identified.
        """
        if len(states) < 2:
            return estimates, None, "skipped:insufficient_window"

        lower = np.array([feasible_sets[key][0] for key in self.parameter_keys], dtype=float)
        upper = np.array([feasible_sets[key][1] for key in self.parameter_keys], dtype=float)
        span = upper - lower
        scale = np.where(span > 0.0, span, 1.0)
        z0 = np.array(
            [
                (float(estimates[key]) - lower[idx]) / scale[idx]
                for idx, key in enumerate(self.parameter_keys)
            ],
            dtype=float,
        )
        z0 = np.clip(z0, 0.0, 1.0)
        residual_scale = max(self.error_bound, 1.0e-6)

        def unpack(z):
            values = lower + np.asarray(z, dtype=float) * span
            return {
                key: float(values[idx]) for idx, key in enumerate(self.parameter_keys)
            }

        def residual(z):
            data_error = self._state_residuals(
                parameters=unpack(z), states=states, controls=controls,
                start_anom=start_anom, anom_step=anom_step,
                dynamics_context=dynamics_context,
            )
            prior_error = self.time_series_prior_weight * (np.asarray(z) - z0)
            return np.concatenate((data_error / residual_scale, prior_error))

        try:
            result = least_squares(
                residual,
                z0,
                bounds=(np.zeros_like(z0), np.ones_like(z0)),
                loss="soft_l1",
                f_scale=1.0,
                max_nfev=80,
                xtol=1.0e-10,
                ftol=1.0e-10,
                gtol=1.0e-10,
            )
            data_residual = lambda z: residual(z)[:- len(z0)]
            jacobian = self._finite_difference_jacobian(data_residual, result.x)
            observable, observability = self._observable_parameter_mask(jacobian)
            # Individual coefficients can be confounded while a combination such
            # as beta-alpha is observable. Preserve the prior's numerical nullspace
            # rather than discarding every component in an identifiable direction.
            _, singular_values, right_vectors = np.linalg.svd(jacobian, full_matrices=False)
            rank_floor = max(
                self.observability_absolute_threshold,
                self.observability_relative_threshold * float(singular_values[0]),
            )
            basis = right_vectors[singular_values > rank_floor].T
            projected_step = basis @ (basis.T @ (result.x - z0))
            refined = self._blend_estimates(
                estimates, unpack(z0 + projected_step), feasible_sets, 1.0
            )
            if refined is None:
                return estimates, None, "failed:projected_step_outside_feasible_sets"
            held = [key for idx, key in enumerate(self.parameter_keys) if not observable[idx]]
            projected_z = np.array(
                [(refined[key] - lower[idx]) / scale[idx]
                 for idx, key in enumerate(self.parameter_keys)], dtype=float
            )
            rms_error = float(
                np.sqrt(np.mean(np.square(data_residual(projected_z))))
                * residual_scale
            )
            held_text = ",".join(held) if held else "none"
            score_text = ",".join(
                (
                    f"{key}={observability[key]['unique_ratio']:.2e}"
                    f"/{observability[key]['column_norm']:.2e}"
                )
                for key in self.parameter_keys
            )
            return (
                refined,
                rms_error,
                f"ok:nfev={result.nfev}:optimizer_cost={result.cost:.3e}:"
                f"fit_rms={rms_error:.6e}:observable_rank={basis.shape[1]}:"
                f"held={held_text}:unique={score_text}",
            )
        except Exception as exc:
            logger.debug("SMID time-series estimate failed: %s", exc)
            return estimates, None, f"failed:{exc}"

    def _finite_difference_jacobian(self, residual, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        base = np.asarray(residual(z), dtype=float)
        jacobian = np.zeros((len(base), len(z)), dtype=float)
        step = 1.0e-4
        for col in range(len(z)):
            z_plus = z.copy()
            z_minus = z.copy()
            z_plus[col] = min(1.0, z_plus[col] + step)
            z_minus[col] = max(0.0, z_minus[col] - step)
            denom = z_plus[col] - z_minus[col]
            if denom <= 0.0:
                continue
            jacobian[:, col] = (
                np.asarray(residual(z_plus), dtype=float)
                - np.asarray(residual(z_minus), dtype=float)
            ) / denom
        return jacobian

    def _observable_parameter_mask(
        self, jacobian: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
        """Return which parameters have usable sensitivity in the data window."""
        jacobian = np.asarray(jacobian, dtype=float)
        if jacobian.size == 0:
            return (
                np.zeros(len(self.parameter_keys), dtype=bool),
                {
                    key: {"column_norm": 0.0, "unique_ratio": 0.0}
                    for key in self.parameter_keys
                },
            )

        column_norms = np.linalg.norm(jacobian, axis=0)
        max_norm = float(np.max(column_norms)) if len(column_norms) else 0.0
        norm_floor = max(
            self.observability_absolute_threshold,
            self.observability_relative_threshold * max_norm,
        )
        observable = np.zeros(len(self.parameter_keys), dtype=bool)
        diagnostics: Dict[str, Dict[str, float]] = {}

        for idx, key in enumerate(self.parameter_keys):
            column = jacobian[:, idx]
            column_norm = float(column_norms[idx])
            unique_ratio = 0.0
            if column_norm >= norm_floor:
                other_indices = [
                    other_idx
                    for other_idx in range(jacobian.shape[1])
                    if other_idx != idx and column_norms[other_idx] >= norm_floor
                ]
                if other_indices:
                    other_columns = jacobian[:, other_indices]
                    u_matrix, singular_values, _ = np.linalg.svd(
                        other_columns, full_matrices=False
                    )
                    rank_floor = (
                        max(float(singular_values[0]) * 1.0e-10, 1.0e-14)
                        if len(singular_values)
                        else 0.0
                    )
                    rank = int(np.sum(singular_values > rank_floor))
                    if rank:
                        basis = u_matrix[:, :rank]
                        unique_component = column - basis @ (basis.T @ column)
                    else:
                        unique_component = column
                    unique_ratio = float(np.linalg.norm(unique_component) / column_norm)
                else:
                    unique_ratio = 1.0
                observable[idx] = unique_ratio >= self.observability_unique_threshold
            diagnostics[key] = {
                "column_norm": column_norm,
                "unique_ratio": unique_ratio,
            }
        return observable, diagnostics

    def _held_keys_from_status(self, status: str) -> set:
        for piece in str(status).split(":"):
            if not piece.startswith("held="):
                continue
            held_text = piece.split("=", 1)[1]
            if not held_text or held_text == "none":
                return set()
            return {key for key in held_text.split(",") if key in self.parameter_keys}
        return set()

    def _normalize_controls(self, controls: np.ndarray, length: int) -> np.ndarray:
        """Require the recorded input for every state-to-state interval."""
        intervals = max(length - 1, 0)
        if controls.size == 0:
            if intervals:
                raise ValueError("SMID requires one recorded control per state interval")
            return np.empty((0, 3), dtype=float)
        if controls.ndim == 1:
            controls = controls.reshape(1, -1)
        if controls.ndim != 2 or controls.shape != (intervals, 3):
            raise ValueError("SMID controls must have shape (number of states - 1, 3)")
        return controls.copy()

    def _copy_ranges(
        self, ranges: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Tuple[float, float]]:
        return {
            key: (float(ranges[key][0]), float(ranges[key][1]))
            for key in self.parameter_keys
        }

    def _ranges_to_lists(
        self, ranges: Dict[str, Tuple[float, float]]
    ) -> Dict[str, List[float]]:
        return {
            key: [float(ranges[key][0]), float(ranges[key][1])]
            for key in self.parameter_keys
        }
