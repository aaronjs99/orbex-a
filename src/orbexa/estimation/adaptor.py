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
from typing import List, Dict, Tuple, Any, Optional, Union
from gekko import GEKKO

from orbexa.utils import thread_worker
from orbexa.core.dynamics import orbital_ellp_undrag

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
    matrices, _, _ = orbital_ellp_undrag(
        dt=range_params["dt"],
        mean_motion=mean_motion,
        eccentricity=orbit_params["eccentricity"],
        alpha=orbit_params.get("drag_alpha"),
        beta=orbit_params.get("drag_beta"),
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

    solver = GEKKO(remote=True)
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

    matrices, _, _ = orbital_ellp_undrag(
        dt=dt,
        mean_motion=mean_motion,
        eccentricity=p_est[0],
        alpha=p_est[1],
        beta=p_est[2],
        m=solver,
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
                    est_results.append(result_queue.get())

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
            )

            logger.info(f"~~  Parameter Estimation : Iteration {adaptation_iter}  ~~")
            if not flag and all(estimates[i] not in p_range[i] for i in range(2)):
                p_estim = estimates
                logger.info("!!! Parameter Estimation : Success !!!")
            else:
                logger.warning("!!! Parameter Estimation : Failure !!!")

        for i in range(num_params):
            estim_lists[i].append(p_estim[i])
            range_lists[i][0].append(p_range[i][0])
            range_lists[i][1].append(p_range[i][1])

    FSS = {}
    for i in range(num_params):
        FSS[keys[i]] = p_range[i]

    return FSS, estim_lists, range_lists


if __name__ == "__main__":
    # Example usage for testing
    pass
