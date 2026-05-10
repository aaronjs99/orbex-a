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

        if prediction_error <= self.prediction_error_threshold:
            record = self._record(
                start_anom=start_anom,
                states=states,
                accepted=False,
                reason="prediction_error_below_threshold",
                fss_before=fss_before,
                fss_candidate=fss_before,
                fss_after=fss_before,
                estimates_before=estimates_before,
                estimates_after=estimates_before,
                prediction_error=prediction_error,
                verification_error=None,
                solve_status={"trigger": "skipped"},
            )
            return feasible_sets, estimates, record

        candidate: Dict[str, Tuple[float, float]] = {}
        solve_status: Dict[str, str] = {}
        for key in self.parameter_keys:
            lower, upper = fss_before[key]
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

        if verified and verification_error <= self.error_bound:
            accepted = True
            reason = "verified"
            fss_after = candidate
            estimates_after = {
                key: float(np.clip(verified_estimates[key], *candidate[key]))
                for key in self.parameter_keys
            }
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
