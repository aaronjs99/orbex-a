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

"""
ORBEX-A Estimation / Adaptation Module.

This module implements the Finite Set Statistics (FSS) based parameter 
estimation and adaptation logic for the Tube MPC controller.
"""

import logging
import time
import numpy as np
import queue
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional, Union
from gekko import GEKKO
from scipy.optimize import least_squares

from orbexa.utils import thread_worker
from orbexa.core.dynamics import orbital_ellp_drag

logger = logging.getLogger(__name__)


# =============================================================================
# Data Generation
# =============================================================================
def gen_adaptor_data(
    range_params: Dict,
    orbit_params: Dict,
    mean_motion: float,
    t_periapsis: float,
    *args,
    **kwargs,
) -> np.ndarray:
    """
    Generate synthetic orbit data for testing adaptation.
    """
    W = []

    # Note: Assuming alpha/beta are ignored for now as they were in dynamics
    # or passed via kwargs if dynamics supports them.
    # Passing them to args just in case.
    matrices, _, _ = orbital_ellp_drag(
        anom_step=range_params["dt"],
        mean_motion=mean_motion,
        eccentricity=orbit_params["eccentricity"],
        alpha=orbit_params.get("drag_alpha", 0.0) or 0.0,
        beta=orbit_params.get("drag_beta", 0.0) or 0.0,
        mu=orbit_params.get("mu", kwargs.get("mu", 3.986004418e14)),
        semi_major_axis=orbit_params.get("semi_major_axis"),
        specific_angular_momentum=orbit_params.get("specific_angular_momentum"),
    )
    A_act, B_act, _, _, d_act = matrices

    remote = kwargs.get("remote", False)
    solver = GEKKO(remote=remote)

    solver.time = np.linspace(
        0,
        range_params["dt"] * (range_params["data_range"] - 1),
        range_params["data_range"],
    )
    t = solver.Var(value=0.0)
    x_act = [
        solver.Var(value=orbit_params["x_0"][i], fixed_initial=True) for i in range(6)
    ]
    u_act = [solver.Param(value=orbit_params["u_t"][i]) for i in range(3)]

    # Equations
    eqs = []
    eqs.append(t.dt() == 1.0)

    # Dynamics loop
    a_val = A_act(t, t_periapsis, m=solver)
    d_val = d_act(t, t_periapsis, m=solver)

    for i in range(3):
        eqs.append(x_act[i + 0].dt() == x_act[i + 3])
        v_dot = 0
        for j in range(6):
            v_dot += a_val[i + 3][j] * x_act[j]

        eqs.append(x_act[i + 3].dt() == v_dot + u_act[i] + d_val[i + 3])

    solver.Equations(eqs)
    solver.options.IMODE = 6
    solver.options.SOLVER = 1
    solver.options.MAX_MEMORY = 512

    disp = kwargs.get("disp", False)
    solver.solve(disp=disp)

    W = np.array([x_act[i].value for i in range(6)])
    W = np.transpose(W)

    solver.cleanup()
    del solver
    return W


# =============================================================================
# Adaptation Core
# =============================================================================
def run_adaptor_op(
    operation: str,
    oper_iter: int,
    dt: float,
    t_s: float,
    u_t: List[np.ndarray],
    W: np.ndarray,
    D: float,
    p_range: List[List[float]],
    mean_motion: float,
    t_periapsis: float,
    *args,
    **kwargs,
) -> Tuple[Any, Any]:
    """Execute a single adaptation operation (Optimization or FSS check)."""

    w_0 = W[0]
    w_f = W[-1]
    flag = False

    solver = GEKKO(remote=kwargs.get("remote", False))
    solver.time = np.linspace(0, dt * (len(W) - 1), len(W))

    final = np.zeros(len(solver.time))
    final[-1] = 1
    final = solver.Param(value=final)

    t = solver.Var(value=0.0)
    x_est = [solver.Var(value=w_0[i], fixed_initial=True) for i in range(6)]
    u_act = [solver.Param(value=u_t[i]) for i in range(3)]

    # Estimation Parameters
    p_est = []
    for i in range(len(p_range)):
        p_est.append(
            solver.FV(
                value=np.mean(p_range[i]),
                lb=p_range[i][0],
                ub=p_range[i][1],
            )
        )
        p_est[i].STATUS = 1
    p_est[-1].STATUS = 0  # Beta fixed (assumed last param)

    matrices, _, _ = orbital_ellp_drag(
        anom_step=dt,
        mean_motion=mean_motion,
        eccentricity=p_est[0],
        alpha=p_est[1],
        beta=p_est[2],
        m=solver,
        mu=kwargs.get("mu", 3.986004418e14),
        semi_major_axis=kwargs.get("semi_major_axis"),
        specific_angular_momentum=kwargs.get("specific_angular_momentum"),
    )
    A_est, B_est, _, _, d_est = matrices

    # Equations
    eqs = []
    eqs.append(t.dt() == 1.0)

    a_val = A_est(t + t_s, t_periapsis, m=solver)
    d_val = d_est(t + t_s, t_periapsis, m=solver)

    for i in range(3):
        eqs.append(x_est[i + 0].dt() == x_est[i + 3])
        v_dot = 0
        for j in range(6):
            v_dot += a_val[i + 3][j] * x_est[j]

        eqs.append(x_est[i + 3].dt() == v_dot + u_act[i] + d_val[i + 3])

    # Error term
    sq_err = 0
    for i in range(6):
        sq_err += (w_f[i] - x_est[i]) ** 2

    if operation == "FSS":
        eqs.append(final * (sq_err - D**2) < 0)
        eqs.append(final * (sq_err + D**2) > 0)
    elif operation == "Optimal":
        eqs.append(final * sq_err < 4.0e-5)

    solver.Equations(eqs)
    solver.options.SOLVER = 3
    solver.options.IMODE = 5
    solver.options.MAX_TIME = 600

    output = (False, [])

    if operation == "FSS":
        solver.options.MAX_ITER = 250
        idx = oper_iter // 2
        p_est[idx].value = p_range[idx][oper_iter % 2]

        if oper_iter % 2 == 0:
            solver.Minimize(p_est[idx] * final)
        else:
            solver.Maximize(p_est[idx] * final)

        try:
            solver.solve(disp=kwargs.get("disp", False))
            output = oper_iter, p_est[idx].value[-1]
        except:
            output = oper_iter, p_range[idx][oper_iter % 2]

    elif operation == "Optimal":
        solver.options.MAX_ITER = 500
        solver.Minimize(final * sq_err)

        try:
            solver.solve(disp=kwargs.get("disp", False))
            est_vals = [p_est[i].value[-1] for i in range(len(p_range))]
            output = False, est_vals
        except:
            flag = True
            output = True, [0] * len(p_range)

    solver.cleanup()
    return output


def run_adaptation(
    init_params: Dict,
    range_params: Dict,
    u_t: List[np.ndarray],
    D: float,
    W: np.ndarray,
    mean_motion: float,
    t_periapsis: float,
    *args,
    **kwargs,
) -> Tuple[Dict, List, List]:
    """
    Main Adaptation Loop.
    """
    dt = range_params["dt"]
    data_range = range_params["data_range"]
    adaptation_range = range_params["adaptation_range"]

    num_params = len(init_params)
    estim_lists = [[] for _ in range(num_params)]
    range_lists = [[[], []] for _ in range(num_params)]
    keys = list(init_params.keys())

    # Initialization
    for i in range(num_params):
        range_lists[i][0] = [init_params[keys[i]][0]]
        range_lists[i][1] = [init_params[keys[i]][1]]

    if "estimates" in kwargs:
        for i in range(num_params):
            estim_lists[i] = [kwargs["estimates"][i]]
    else:
        for i in range(num_params):
            estim_lists[i] = [np.mean(range_lists[i])]

    p_range = [
        [range_lists[i][0][-1], range_lists[i][1][-1]] for i in range(num_params)
    ]

    use_threading = kwargs.get("threader", False)
    adaptation_iter = 0
    p_estim = [estim_lists[i][-1] for i in range(num_params)]

    for data_iter in range(1, data_range):
        # Update history
        p_estim = [estim_lists[i][-1] for i in range(num_params)]
        p_range = [
            [range_lists[i][0][-1], range_lists[i][1][-1]] for i in range(num_params)
        ]

        # Adaptation Trigger
        if data_iter % adaptation_range == 0 and data_iter != 0:
            adaptation_iter += 1
            oper_iter = 0
            est_results = []

            if use_threading:
                result_queue = queue.Queue()
                threads = []

            while oper_iter < 2 * (len(p_range) - 1):
                adaptor_args = {
                    "operation": "FSS",
                    "oper_iter": oper_iter,
                    "dt": dt,
                    "t_s": (data_iter - adaptation_range) * dt,
                    "u_t": [
                        u_t[i][data_iter - adaptation_range : data_iter]
                        for i in range(3)
                    ],
                    "W": W[data_iter - adaptation_range : data_iter],
                    "D": D,
                    "p_range": p_range,
                    "disp": kwargs.get("disp", False),
                    "mean_motion": mean_motion,
                    "t_periapsis": t_periapsis,
                    "remote": kwargs.get("remote", False),
                    "mu": kwargs.get("mu", 3.986004418e14),
                    "semi_major_axis": kwargs.get("semi_major_axis"),
                    "specific_angular_momentum": kwargs.get(
                        "specific_angular_momentum"
                    ),
                }

                if use_threading:
                    thread = threading.Thread(
                        target=thread_worker,
                        args=(result_queue, run_adaptor_op),
                        kwargs=adaptor_args,
                    )
                    thread.start()
                    threads.append(thread)
                else:
                    result = run_adaptor_op(**adaptor_args)
                    est_results.append(result)

                oper_iter += 1

            if use_threading:
                for thread in threads:
                    thread.join()
                while not result_queue.empty():
                    ok, payload = result_queue.get()
                    if ok:
                        est_results.append(payload)
                    else:
                        logger.debug("SMID threaded operation failed: %s", payload)

            p_xi = [[0.0, 0.0] for _ in range(len(p_range))]
            p_xi[-1] = p_range[-1].copy()

            for result in est_results:
                op_iter, val = result
                p_xi[op_iter // 2][op_iter % 2] = val

            for i in range(len(p_range)):
                p_range[i] = [
                    max(min(p_xi[i]), p_range[i][0]),
                    min(max(p_xi[i]), p_range[i][1]),
                ]

            flag, estimates = run_adaptor_op(
                operation="Optimal",
                oper_iter=1,
                dt=dt,
                t_s=(data_iter - adaptation_range) * dt,
                u_t=[
                    u_t[i][data_iter - adaptation_range : data_iter] for i in range(3)
                ],
                W=W[data_iter - adaptation_range : data_iter],
                D=D,
                p_range=p_range,
                disp=kwargs.get("disp", False),
                mean_motion=mean_motion,
                t_periapsis=t_periapsis,
                remote=kwargs.get("remote", False),
                mu=kwargs.get("mu", 3.986004418e14),
                semi_major_axis=kwargs.get("semi_major_axis"),
                specific_angular_momentum=kwargs.get("specific_angular_momentum"),
            )

            logger.debug(f"~~  Parameter Estimation : Iteration {adaptation_iter}  ~~")
            if not flag and all(
                p_range[i][0] <= estimates[i] <= p_range[i][1]
                for i in range(len(p_range))
            ):
                p_estim = estimates
                logger.debug("!!! Parameter Estimation : Success !!!")
            else:
                logger.debug("!!! Parameter Estimation : Failure !!!")

        for i in range(num_params):
            estim_lists[i].append(p_estim[i])
            range_lists[i][0].append(p_range[i][0])
            range_lists[i][1].append(p_range[i][1])

    FSS = {}
    for i in range(num_params):
        FSS[keys[i]] = p_range[i]

    return FSS, estim_lists, range_lists


@dataclass
class SMIDRecord:
    """Audit record for one Algorithm 1 set-membership update."""

    start_anom: float
    window_size: int
    accepted: bool
    reason: str
    fss_before: Dict[str, List[float]]
    fss_candidate: Dict[str, List[float]]
    fss_after: Dict[str, List[float]]
    estimates_before: Dict[str, float]
    estimates_after: Dict[str, float]
    prediction_error: float
    verification_error: Optional[float]
    solve_status: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_anom": float(self.start_anom),
            "window_size": int(self.window_size),
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "fss_before": self.fss_before,
            "fss_candidate": self.fss_candidate,
            "fss_after": self.fss_after,
            "estimates_before": self.estimates_before,
            "estimates_after": self.estimates_after,
            "prediction_error": float(self.prediction_error),
            "verification_error": (
                None
                if self.verification_error is None
                else float(self.verification_error)
            ),
            "solve_status": dict(self.solve_status),
        }


class SMIDAdaptor:
    """
    GEKKO implementation of the paper's set-membership identification update.

    The plant truth is not owned by this class.  It receives measured state and
    control windows, solves min/max feasible parameter bounds under the current
    FSS, verifies the candidate set against the terminal prediction error bound,
    and returns either the verified candidate or the untouched previous belief.
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
        ts_estimates, ts_error, ts_status = self._time_series_estimate(
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
            estimates_after = (
                self._blend_estimates(
                    estimates_before,
                    ts_estimates,
                    fss_before,
                    self.time_series_update_gain,
                )
                if ts_status.startswith("ok")
                else estimates_before
            )
            record = self._record(
                start_anom=start_anom,
                states=states,
                accepted=False,
                reason=(
                    "time_series_refined_below_threshold"
                    if ts_status.startswith("ok")
                    else "prediction_error_below_threshold"
                ),
                fss_before=fss_before,
                fss_candidate=fss_before,
                fss_after=fss_before,
                estimates_before=estimates_before,
                estimates_after=estimates_after,
                prediction_error=prediction_error,
                verification_error=ts_error if ts_status.startswith("ok") else None,
                solve_status={
                    "trigger": "skipped",
                    "time_series": ts_status,
                    "observability_held": ",".join(sorted(held_keys)) or "none",
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

        verified, verified_estimates, verification_error, verify_status = (
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
            accepted = True
            reason = "verified"
            fss_after = candidate
            base_estimates = (
                self._blend_estimates(
                    estimates_before,
                    ts_estimates,
                    candidate,
                    self.time_series_update_gain,
                )
                if ts_status.startswith("ok")
                else verified_estimates
            )
            estimates_after = {
                key: float(np.clip(verified_estimates[key], *candidate[key]))
                for key in self.parameter_keys
            }
            for key in self.parameter_keys:
                estimates_after[key] = float(
                    np.clip(
                        0.35 * estimates_after[key] + 0.65 * base_estimates[key],
                        *candidate[key],
                    )
                )
            for key in held_keys:
                estimates_after[key] = estimates_before[key]
        else:
            accepted = False
            reason = "verification_failed"
            fss_after = fss_before
            estimates_after = estimates_before

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

    def _blend_estimates(
        self,
        previous: Dict[str, float],
        refined: Dict[str, float],
        feasible_sets: Dict[str, Tuple[float, float]],
        gain: float,
    ) -> Dict[str, float]:
        blended = {}
        for key in self.parameter_keys:
            value = (1.0 - gain) * float(previous[key]) + gain * float(refined[key])
            blended[key] = float(np.clip(value, *feasible_sets[key]))
        return blended

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
        for idx in range(len(states) - 1):
            matrices, _, _ = orbital_ellp_drag(anom_step=anom_step, **params)
            A_func, B_func, _, _, d_func = matrices
            q_val = start_anom + idx * anom_step
            A_val = np.asarray(A_func(q_val, params.get("time_periapsis", 0.0)), dtype=float)
            B_val = np.asarray(B_func(), dtype=float)
            d_val = np.asarray(d_func(q_val, params.get("time_periapsis", 0.0)), dtype=float)
            predicted = predicted + (
                A_val @ predicted + B_val @ controls[idx] + d_val
            ) * anom_step
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
        span = np.maximum(upper - lower, 1.0e-15)
        z0 = np.array(
            [
                (float(estimates[key]) - lower[idx]) / span[idx]
                for idx, key in enumerate(self.parameter_keys)
            ],
            dtype=float,
        )
        z0 = np.clip(z0, 0.0, 1.0)
        residual_scale = max(self.error_bound, 1.0e-6)

        def unpack(z):
            values = lower + np.clip(z, 0.0, 1.0) * span
            return {
                key: float(values[idx]) for idx, key in enumerate(self.parameter_keys)
            }

        def residual(z):
            parameter_values = unpack(z)
            params = dict(dynamics_context)
            params.update(parameter_values)
            params["specific_angular_momentum"] = None
            matrices, _, _ = orbital_ellp_drag(anom_step=anom_step, **params)
            A_func, B_func, _, _, d_func = matrices
            B_val = np.asarray(B_func(), dtype=float)
            pieces = []
            for idx in range(len(states) - 1):
                q_val = start_anom + idx * anom_step
                A_val = np.asarray(
                    A_func(q_val, params.get("time_periapsis", 0.0)),
                    dtype=float,
                )
                d_val = np.asarray(
                    d_func(q_val, params.get("time_periapsis", 0.0)),
                    dtype=float,
                )
                predicted = states[idx] + (
                    A_val @ states[idx] + B_val @ controls[idx] + d_val
                ) * anom_step
                error = np.asarray(states[idx + 1] - predicted, dtype=float)
                pieces.append(error[:3] / residual_scale)
                pieces.append(2.0 * error[3:] / residual_scale)
            pieces.append(self.time_series_prior_weight * (np.asarray(z) - z0))
            return np.concatenate(pieces)

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
            refined = unpack(result.x)
            data_residual = lambda z: residual(z)[:- len(z0)]
            jacobian = self._finite_difference_jacobian(data_residual, result.x)
            observable, observability = self._observable_parameter_mask(jacobian)
            held = []
            for idx, key in enumerate(self.parameter_keys):
                if not observable[idx]:
                    refined[key] = float(estimates[key])
                    held.append(key)
            rms_error = float(
                np.sqrt(np.mean(np.square(residual(result.x)[:- len(z0)])))
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
                f"ok:nfev={result.nfev}:cost={result.cost:.3e}:held={held_text}:unique={score_text}",
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

    def _solve_parameter_bound(
        self,
        *,
        key: str,
        sense: str,
        feasible_sets: Dict[str, Tuple[float, float]],
        states: np.ndarray,
        controls: np.ndarray,
        start_anom: float,
        anom_step: float,
        dynamics_context: Dict[str, Any],
    ) -> Tuple[float, str]:
        lower, upper = feasible_sets[key]
        default_value = lower if sense == "min" else upper
        solver = None
        try:
            solver, p_vars, _, final_error = self._build_window_model(
                feasible_sets=feasible_sets,
                states=states,
                controls=controls,
                start_anom=start_anom,
                anom_step=anom_step,
                dynamics_context=dynamics_context,
            )
            solver.Equation(final_error <= self.error_bound**2)
            if sense == "min":
                solver.Minimize(p_vars[key])
            else:
                solver.Maximize(p_vars[key])
            solver.solve(disp=self.disp)
            value = float(p_vars[key].value[-1])
            solver.cleanup()
            return value, "ok"
        except Exception as exc:
            if solver is not None:
                try:
                    solver.cleanup()
                except Exception:
                    pass
            logger.debug("SMID %s %s bound failed: %s", key, sense, exc)
            return float(default_value), f"failed:{exc}"

    def _verify_candidate(
        self,
        *,
        feasible_sets: Dict[str, Tuple[float, float]],
        states: np.ndarray,
        controls: np.ndarray,
        start_anom: float,
        anom_step: float,
        dynamics_context: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, float], float, str]:
        solver = None
        try:
            solver, p_vars, x_est, final_error = self._build_window_model(
                feasible_sets=feasible_sets,
                states=states,
                controls=controls,
                start_anom=start_anom,
                anom_step=anom_step,
                dynamics_context=dynamics_context,
            )
            solver.Minimize(final_error)
            solver.solve(disp=self.disp)
            estimates = {key: float(p_vars[key].value[-1]) for key in self.parameter_keys}
            terminal = np.array([float(x_est[i].value[-1]) for i in range(6)])
            error = float(np.linalg.norm(terminal - states[-1]))
            solver.cleanup()
            return True, estimates, error, "ok"
        except Exception as exc:
            try:
                solver.cleanup()
            except Exception:
                pass
            logger.debug("SMID candidate verification failed: %s", exc)
            fallback = {
                key: float(np.mean(feasible_sets[key])) for key in self.parameter_keys
            }
            return False, fallback, float("inf"), f"failed:{exc}"

    def _build_window_model(
        self,
        *,
        feasible_sets: Dict[str, Tuple[float, float]],
        states: np.ndarray,
        controls: np.ndarray,
        start_anom: float,
        anom_step: float,
        dynamics_context: Dict[str, Any],
    ):
        solver = GEKKO(remote=self.remote)
        solver.time = np.linspace(0.0, anom_step * (len(states) - 1), len(states))
        solver.options.IMODE = 6
        solver.options.SOLVER = 3
        solver.options.MAX_ITER = self.max_iter
        solver.options.MAX_TIME = 30
        solver.options.MAX_MEMORY = 512
        solver.options.OTOL = 1.0e-6
        solver.options.RTOL = 1.0e-6

        tau = solver.Var(value=0.0)
        solver.Equation(tau.dt() == 1.0)
        x_est = [
            solver.Var(value=float(states[0, i]), fixed_initial=True)
            for i in range(6)
        ]
        u_param = [solver.Param(value=controls[:, i]) for i in range(3)]

        p_vars = {}
        for key in self.parameter_keys:
            lower, upper = feasible_sets[key]
            p_var = solver.FV(value=float(0.5 * (lower + upper)), lb=lower, ub=upper)
            p_var.STATUS = 1
            p_vars[key] = p_var

        params = dict(dynamics_context)
        params["eccentricity"] = p_vars["eccentricity"]
        params["alpha"] = p_vars["alpha"]
        params["beta"] = p_vars["beta"]
        params["specific_angular_momentum"] = None
        matrices, _, _ = orbital_ellp_drag(anom_step=anom_step, m=solver, **params)
        A_est, _, _, _, d_est = matrices
        q_expr = tau + float(start_anom)
        A_val = A_est(q_expr, params.get("time_periapsis", 0.0), m=solver, q=q_expr)
        d_val = d_est(q_expr, params.get("time_periapsis", 0.0), m=solver, q=q_expr)

        equations = []
        for axis in range(3):
            equations.append(x_est[axis].dt() == x_est[axis + 3])
            accel = 0
            for state_idx in range(6):
                accel += A_val[axis + 3][state_idx] * x_est[state_idx]
            equations.append(x_est[axis + 3].dt() == accel + u_param[axis] + d_val[axis + 3])
        solver.Equations(equations)

        final = np.zeros(len(states))
        final[-1] = 1.0
        final_param = solver.Param(value=final)
        sq_err = 0
        for idx in range(6):
            sq_err += (x_est[idx] - float(states[-1, idx])) ** 2
        final_error = solver.Intermediate(final_param * sq_err)
        return solver, p_vars, x_est, final_error

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
            solve_status=solve_status,
        )
        self.records.append(record)
        return record

    def _normalize_controls(self, controls: np.ndarray, length: int) -> np.ndarray:
        if controls.size == 0:
            return np.zeros((length, 3), dtype=float)
        if controls.ndim == 1:
            controls = controls.reshape(1, -1)
        if controls.shape[1] != 3:
            raise ValueError("SMID controls must have shape (N, 3)")
        if controls.shape[0] < length:
            pad = np.repeat(controls[-1:, :], length - controls.shape[0], axis=0)
            controls = np.vstack((controls, pad))
        return controls[:length]

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


if __name__ == "__main__":
    # Example usage for testing
    import numpy as np

    # Mock configuration for testing
    range_params = {"dt": 0.1, "data_range": 100, "adaptation_range": 10}

    # Mock measurement data
    W = [np.random.rand(10, 1) for _ in range(50)]

    print("Adaptor module test - placeholder implementation")
    print(f"Mock data shape: {len(W)} measurements")
