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

"""SMID update policy; conservative mission membership is a separate service."""

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

from orbexa.estimation.smid_records import SMIDRecord
from orbexa.estimation.prediction import ParameterPredictor

class SMIDAdaptor:
    """
    Local recorded-window parameter fitting and candidate-set update policy.

    Numerical point prediction and symbolic fitting have separate components.
    The caller supplies measured states, controls and a parameter box; this class
    never owns plant truth. Candidate bounds are local optimization results,
    rather than a conservative enclosure certificate. The coupled mission uses
    only the returned point estimate and retains the interval contractor's box.
    Point updates stay in the observed parameter subspace.
    """

    def __init__(
        self,
        *,
        parameter_keys: Tuple[str, ...] = ("eccentricity", "alpha", "beta"),
        error_bound: float = 0.08,
        prediction_error_threshold: Optional[float] = None,
        remote: bool = False,
        disp: bool = False,
        max_iter: int = 120,
        time_series_prior_weight: float = 0.15,
        time_series_update_gain: float = 1.0,
        minimum_widths: Optional[Dict[str, float]] = None,
        observability_relative_threshold: float = 0.02,
        observability_unique_threshold: float = 0.08,
        observability_absolute_threshold: float = 1.0e-8,
    ):
        self.parameter_keys = tuple(parameter_keys)
        self.error_bound = float(error_bound)
        self.prediction_error_threshold = (
            float(error_bound)
            if prediction_error_threshold is None
            else float(prediction_error_threshold)
        )
        self.remote = bool(remote)
        self.disp = bool(disp)
        self.max_iter = int(max_iter)
        self.time_series_prior_weight = float(time_series_prior_weight)
        self.time_series_update_gain = float(np.clip(time_series_update_gain, 0.0, 1.0))
        self.minimum_widths = {
            key: float(value) for key, value in (minimum_widths or {}).items()
        }
        self.observability_relative_threshold = float(observability_relative_threshold)
        self.observability_unique_threshold = float(observability_unique_threshold)
        self.observability_absolute_threshold = float(observability_absolute_threshold)
        self.records: List[SMIDRecord] = []

    def update(
        self,
        *,
        feasible_sets: Dict[str, Tuple[float, float]],
        estimates: Dict[str, float],
        states: np.ndarray,
        controls: np.ndarray,
        start_anom: float,
        anom_step: float,
        dynamics_context: Dict[str, Any],
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float], SMIDRecord]:
        states = np.asarray(states, dtype=float)
        controls = self._normalize_controls(np.asarray(controls, dtype=float), len(states))
        fss_before = self._copy_ranges(feasible_sets)
        estimates_before = {key: float(estimates[key]) for key in self.parameter_keys}

        if len(states) < 2:
            record = self._record(
                start_anom=start_anom,
                states=states,
                accepted=False,
                reason="insufficient_window",
                fss_before=fss_before,
                fss_candidate=fss_before,
                fss_after=fss_before,
                estimates_before=estimates_before,
                estimates_after=estimates_before,
                prediction_error=0.0,
                verification_error=None,
                point_prediction_error=0.0,
                solve_status={},
            )
            return feasible_sets, estimates, record

        prediction_error = self.prediction_error(
            states=states,
            controls=controls,
            start_anom=start_anom,
            anom_step=anom_step,
            parameters=estimates_before,
            dynamics_context=dynamics_context,
        )
        ts_estimates, _, ts_status = self._time_series_estimate(
            feasible_sets=fss_before,
            estimates=estimates_before,
            states=states,
            controls=controls,
            start_anom=start_anom,
            anom_step=anom_step,
            dynamics_context=dynamics_context,
        )
        held_keys = self._held_keys_from_status(ts_status)

        if prediction_error <= self.prediction_error_threshold:
            proposed = ts_estimates if ts_status.startswith("ok") else estimates_before
            point = self._blend_estimates(
                estimates_before, proposed, fss_before, self.time_series_update_gain
            )
            point_ok, point_prediction_error, point_fit_error = self._point_metrics(
                parameters=point or estimates_before,
                states=states,
                controls=controls,
                start_anom=start_anom,
                anom_step=anom_step,
                dynamics_context=dynamics_context,
            )
            point_ok = point is not None and point_ok
            estimates_after = point if point_ok else estimates_before
            reason = (
                "time_series_refined_below_threshold"
                if point_ok and ts_status.startswith("ok")
                else "prediction_error_below_threshold"
                if point_ok
                else "point_verification_failed"
            )
            if not point_ok:
                _, point_prediction_error, point_fit_error = self._point_metrics(
                    parameters=estimates_before,
                    states=states,
                    controls=controls,
                    start_anom=start_anom,
                    anom_step=anom_step,
                    dynamics_context=dynamics_context,
                )
            record = self._record(
                start_anom=start_anom,
                states=states,
                accepted=False,
                reason=reason,
                fss_before=fss_before,
                fss_candidate=fss_before,
                fss_after=fss_before,
                estimates_before=estimates_before,
                estimates_after=estimates_after,
                prediction_error=prediction_error,
                verification_error=None,
                point_prediction_error=point_prediction_error,
                point_fit_error=point_fit_error,
                solve_status={
                    "trigger": "skipped",
                    "time_series": ts_status,
                    "observability_held": ",".join(sorted(held_keys)) or "none",
                    "point_update": "verified" if point_ok else "previous_preserved",
                },
            )
            return feasible_sets, estimates_after, record

        candidate: Dict[str, Tuple[float, float]] = {}
        solve_status: Dict[str, str] = {
            "observability_held": ",".join(sorted(held_keys)) or "none"
        }
        for key in self.parameter_keys:
            lower, upper = fss_before[key]
            if key in held_keys:
                candidate[key] = (float(lower), float(upper))
                solve_status[f"{key}_interval"] = "observability_held_previous"
                continue
            min_value, min_status = self._solve_parameter_bound(
                key=key,
                sense="min",
                feasible_sets=fss_before,
                states=states,
                controls=controls,
                start_anom=start_anom,
                anom_step=anom_step,
                dynamics_context=dynamics_context,
            )
            max_value, max_status = self._solve_parameter_bound(
                key=key,
                sense="max",
                feasible_sets=fss_before,
                states=states,
                controls=controls,
                start_anom=start_anom,
                anom_step=anom_step,
                dynamics_context=dynamics_context,
            )
            solve_status[f"{key}_min"] = min_status
            solve_status[f"{key}_max"] = max_status
            new_lower = max(float(lower), float(min_value))
            new_upper = min(float(upper), float(max_value))
            if new_lower > new_upper:
                new_lower, new_upper = float(lower), float(upper)
                solve_status[f"{key}_interval"] = "invalid_candidate_preserved"
            candidate[key] = (float(new_lower), float(new_upper))
        candidate = self._apply_minimum_widths(candidate, fss_before)
        for key in held_keys:
            candidate[key] = fss_before[key]

        verified, _, verification_error, verify_status = (
            self._verify_candidate(
                feasible_sets=candidate,
                states=states,
                controls=controls,
                start_anom=start_anom,
                anom_step=anom_step,
                dynamics_context=dynamics_context,
            )
        )
        solve_status["verification"] = verify_status
        solve_status["time_series"] = ts_status

        if verified and verification_error <= self.error_bound:
            proposed = ts_estimates if ts_status.startswith("ok") else estimates_before
            point = self._blend_estimates(
                estimates_before, proposed, candidate, self.time_series_update_gain
            )
            point_ok, point_prediction_error, point_fit_error = self._point_metrics(
                parameters=point or estimates_before,
                states=states,
                controls=controls,
                start_anom=start_anom,
                anom_step=anom_step,
                dynamics_context=dynamics_context,
            )
            accepted = point is not None and point_ok
            reason = "verified" if accepted else "point_verification_failed"
            fss_after = candidate if accepted else fss_before
            estimates_after = point if accepted else estimates_before
            solve_status["point_update"] = (
                "verified"
                if accepted
                else f"previous_preserved:prediction={point_prediction_error:.6e}:fit={point_fit_error:.6e}"
            )
        else:
            accepted = False
            reason = "verification_failed"
            fss_after = fss_before
            estimates_after = estimates_before
            solve_status["point_update"] = "previous_preserved:set_verification_failed"

        if not accepted:
            _, point_prediction_error, point_fit_error = self._point_metrics(
                parameters=estimates_before,
                states=states,
                controls=controls,
                start_anom=start_anom,
                anom_step=anom_step,
                dynamics_context=dynamics_context,
            )

        record = self._record(
            start_anom=start_anom,
            states=states,
            accepted=accepted,
            reason=reason,
            fss_before=fss_before,
            fss_candidate=self._copy_ranges(candidate),
            fss_after=self._copy_ranges(fss_after),
            estimates_before=estimates_before,
            estimates_after=estimates_after,
            prediction_error=prediction_error,
            verification_error=verification_error,
            point_prediction_error=point_prediction_error,
            point_fit_error=point_fit_error,
            solve_status=solve_status,
        )
        return fss_after, estimates_after, record

    def _apply_minimum_widths(
        self,
        candidate: Dict[str, Tuple[float, float]],
        previous: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Tuple[float, float]]:
        adjusted = self._copy_ranges(candidate)
        for key, min_width in self.minimum_widths.items():
            if key not in adjusted or min_width <= 0.0:
                continue
            lower, upper = adjusted[key]
            prev_lower, prev_upper = previous[key]
            previous_width = max(float(prev_upper) - float(prev_lower), 0.0)
            width_floor = min(float(min_width), previous_width)
            if width_floor <= 0.0 or upper - lower >= width_floor:
                continue
            center = 0.5 * (lower + upper)
            half_width = 0.5 * width_floor
            center = float(np.clip(center, prev_lower + half_width, prev_upper - half_width))
            adjusted[key] = (center - half_width, center + half_width)
        return adjusted

    def _blend_estimates(self, *args, **kwargs):
        return self._predictor()._blend_estimates(*args, **kwargs)

    @staticmethod
    def _rk4_step(*args, **kwargs):
        return ParameterPredictor._rk4_step(*args, **kwargs)

    def _state_residuals(self, *args, **kwargs):
        return self._predictor()._state_residuals(*args, **kwargs)

    def _point_metrics(self, *args, **kwargs):
        return self._predictor()._point_metrics(*args, **kwargs)

    def prediction_error(self, *args, **kwargs):
        return self._predictor().prediction_error(*args, **kwargs)

    def _time_series_estimate(self, *args, **kwargs):
        return self._predictor()._time_series_estimate(*args, **kwargs)

    def _finite_difference_jacobian(self, *args, **kwargs):
        return self._predictor()._finite_difference_jacobian(*args, **kwargs)

    def _observable_parameter_mask(self, *args, **kwargs):
        return self._predictor()._observable_parameter_mask(*args, **kwargs)

    def _held_keys_from_status(self, *args, **kwargs):
        return self._predictor()._held_keys_from_status(*args, **kwargs)

    def _solve_parameter_bound(self, *args, **kwargs):
        return self._window_backend().solve_parameter_bound(*args, **kwargs)

    def _verify_candidate(self, *args, **kwargs):
        return self._window_backend().verify_candidate(*args, **kwargs)

    def _build_window_model(self, *args, **kwargs):
        return self._window_backend().build_window_model(*args, **kwargs)

    def _record(
        self,
        *,
        start_anom: float,
        states: np.ndarray,
        accepted: bool,
        reason: str,
        fss_before: Dict[str, Tuple[float, float]],
        fss_candidate: Dict[str, Tuple[float, float]],
        fss_after: Dict[str, Tuple[float, float]],
        estimates_before: Dict[str, float],
        estimates_after: Dict[str, float],
        prediction_error: float,
        verification_error: Optional[float],
        solve_status: Dict[str, str],
        point_prediction_error: Optional[float] = None,
        point_fit_error: Optional[float] = None,
    ) -> SMIDRecord:
        record = SMIDRecord(
            start_anom=float(start_anom),
            window_size=int(len(states)),
            accepted=accepted,
            reason=reason,
            fss_before=self._ranges_to_lists(fss_before),
            fss_candidate=self._ranges_to_lists(fss_candidate),
            fss_after=self._ranges_to_lists(fss_after),
            estimates_before={key: float(value) for key, value in estimates_before.items()},
            estimates_after={key: float(value) for key, value in estimates_after.items()},
            prediction_error=float(prediction_error),
            verification_error=verification_error,
            point_prediction_error=point_prediction_error,
            point_fit_error=point_fit_error,
            solve_status=solve_status,
        )
        self.records.append(record)
        return record

    def _normalize_controls(self, *args, **kwargs):
        return self._predictor()._normalize_controls(*args, **kwargs)

    def _copy_ranges(self, *args, **kwargs):
        return self._predictor()._copy_ranges(*args, **kwargs)

    def _ranges_to_lists(self, *args, **kwargs):
        return self._predictor()._ranges_to_lists(*args, **kwargs)

    def _predictor(self) -> ParameterPredictor:
        return ParameterPredictor(
            parameter_keys=self.parameter_keys,
            error_bound=self.error_bound,
            time_series_prior_weight=self.time_series_prior_weight,
            observability_absolute_threshold=self.observability_absolute_threshold,
            observability_relative_threshold=self.observability_relative_threshold,
            observability_unique_threshold=self.observability_unique_threshold,
        )

    def _window_backend(self):
        from orbexa.solvers.gekko_estimation import GekkoWindowEstimator
        return GekkoWindowEstimator(parameter_keys=self.parameter_keys, error_bound=self.error_bound,
                                    remote=self.remote, disp=self.disp, max_iter=self.max_iter)
