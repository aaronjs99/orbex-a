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

"""GEKKO window formulation and local candidate-fitting backend."""

import logging
import time
import numpy as np
import queue
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional, Union
from scipy.optimize import least_squares
from orbexa.core.dynamics import orbital_ellp_drag
from orbexa.utils.solver_options import gekko_memory_exponent

logger = logging.getLogger(__name__)

from gekko import GEKKO


class GekkoWindowEstimator:
    def __init__(self, *, parameter_keys, error_bound, remote, disp, max_iter, max_memory=4):
        self.parameter_keys = parameter_keys
        self.error_bound = error_bound
        self.remote = remote
        self.disp = disp
        self.max_iter = max_iter
        self.max_memory = gekko_memory_exponent(max_memory)

    def solve_parameter_bound(
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
            solver, p_vars, _, final_error = self.build_window_model(
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

    def verify_candidate(
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
            solver, p_vars, x_est, final_error = self.build_window_model(
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

    def build_window_model(
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
        solver.options.MAX_MEMORY = self.max_memory
        solver.options.OTOL = 1.0e-6
        solver.options.RTOL = 1.0e-6

        tau = solver.Var(value=0.0)
        solver.Equation(tau.dt() == 1.0)
        x_est = [
            solver.Var(value=float(states[0, i]), fixed_initial=True)
            for i in range(6)
        ]
        # GEKKO node k+1 drives recorded interval k; the initial node has no
        # preceding interval. Keep the numerical predictors interval-aligned.
        nodal_controls = np.vstack((controls[:1], controls))
        u_param = [solver.Param(value=nodal_controls[:, i]) for i in range(3)]

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
