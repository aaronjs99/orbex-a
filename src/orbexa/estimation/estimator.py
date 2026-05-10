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

# PACKAGE IMPORTS
import time
import numpy as np
from gekko import GEKKO
from functools import partial
from matplotlib import pyplot as plt

import orbexa.params as p

np.random.seed(int(time.time()))


def estimator(t_stream, y_stream, model_functions, prev_theta=None):
    if prev_theta is None:
        prev_theta = np.zeros(len(model_functions))

    solver = GEKKO(remote=True)
    solver.time = t_stream

    T = solver.Param(value=t_stream)
    y = solver.Param(value=y_stream)

    t = solver.Var(value=t_stream[0])
    z = solver.Var(fixed_initial=False, value=y_stream[0])
    theta = [
        solver.FV(fixed_initial=False, value=prev_theta[i])
        for i in range(len(model_functions))
    ]

    solver.Equation(t.dt() == 1.0)
    solver.Equation(
        z
        == np.sum(
            [
                model_functions[i](t, m=solver) * theta[i]
                for i in range(len(model_functions))
            ]
        )
    )
    solver.Minimize(T * (y - z) ** 2)

    solver.options.IMODE = 6
    solver.options.SOLVER = 1
    solver.options.MAX_ITER = 1000
    solver.solve(disp=True)

    error = solver.options.OBJFCNVAL
    theta = np.array([theta[i].value[-1] for i in range(len(model_functions))])

    solver.cleanup()
    del solver
    return error, theta


def gen_y_stream(y_0, phi_0, omega_0, I, t_stream):
    y_stream = []

    solver = GEKKO(remote=True)
    solver.time = t_stream
    t = solver.Var(value=0.0)
    y = [solver.Var(value=y_0[i], fixed_initial=True) for i in range(3)]
    phi = [solver.Var(value=phi_0[i], fixed_initial=True) for i in range(3)]
    omega = [solver.Var(value=omega_0[i], fixed_initial=True) for i in range(3)]

    roll, pitch, yaw = phi
    rot_mat = [
        [
            solver.cos(yaw) * solver.cos(pitch),
            solver.cos(yaw) * solver.sin(pitch) * solver.sin(roll)
            - solver.sin(yaw) * solver.cos(roll),
            solver.cos(yaw) * solver.sin(pitch) * solver.cos(roll)
            + solver.sin(yaw) * solver.sin(roll),
        ],
        [
            solver.sin(yaw) * solver.cos(pitch),
            solver.sin(yaw) * solver.sin(pitch) * solver.sin(roll)
            + solver.cos(yaw) * solver.cos(roll),
            solver.sin(yaw) * solver.sin(pitch) * solver.cos(roll)
            - solver.cos(yaw) * solver.sin(roll),
        ],
        [
            -solver.sin(pitch),
            solver.cos(pitch) * solver.sin(roll),
            solver.cos(pitch) * solver.cos(roll),
        ],
    ]
    for i in range(3):
        for j in range(3):
            rot_mat[i][j] = solver.Intermediate(rot_mat[i][j])

    eqs = []
    eqs.append(t.dt() == 1.0)
    for i in range(3):
        eqs.append(y[i] == np.matmul(rot_mat, y)[i])
        eqs.append(phi[i].dt() == omega[i])
        eqs.append(
            omega[i].dt()
            == -np.matmul(np.linalg.inv(I), np.cross(omega, np.matmul(I, omega)))[i]
        )
    eqs = solver.Equations(eqs)
    solver.options.IMODE = 6
    solver.options.SOLVER = 3
    solver.options.MAX_MEMORY = 512
    solver.solve(disp=True, debug=2)

    y_stream = [y[i].value for i in range(3)]
    solver.cleanup()
    del solver
    return y_stream


if __name__ == "__main__":

    def f_1(t, *args, **kwargs):
        return 1.0

    def f_t(t, *args, **kwargs):
        return t

    def f_t2(t, *args, **kwargs):
        return t**2

    def f_t3(t, *args, **kwargs):
        return t**3

    def f_t4(t, *args, **kwargs):
        return t**4

    def f_exp(t, *args, **kwargs):
        if "m" in kwargs:
            solver = kwargs["m"]
            return solver.exp(t)
        return np.exp(t)

    def f_sin(t, *args, **kwargs):
        if "m" in kwargs:
            solver = kwargs["m"]
            return solver.sin(t)
        return np.sin(t)

    def f_cos(t, *args, **kwargs):
        if "m" in kwargs:
            solver = kwargs["m"]
            return solver.cos(t)
        return np.cos(t)

    y_0 = np.array([0.0, -0.8, 0.3])
    t_stream = np.linspace(0.0, 10.0, 501)
    y_stream = gen_y_stream(y_0, p.th_T0, p.w_T0, p.I_T0, t_stream)

    model_functions = np.array([f_1, f_t, f_t2, f_t3, f_t4, f_exp, f_sin, f_cos])
    error = np.array([0.0 for dim in range(len(y_stream))])
    estimate = np.array(
        [[0.0 for func in range(len(model_functions))] for dim in range(len(y_stream))]
    )
    for dim in range(len(y_stream)):
        error[dim], estimate[dim] = estimator(
            t_stream,
            y_stream[dim],
            model_functions,
            prev_theta=estimate[dim - 1] if dim > 0 else None,
        )

    z_stream = [
        [
            np.sum(
                [
                    model_functions[i](t) * estimate[dim][i]
                    for i in range(len(model_functions))
                ]
            )
            for t in t_stream
        ]
        for dim in range(len(y_stream))
    ]

    for dim in range(len(estimate)):
        print("Error          for Dimension ", dim, " : ", error[dim])
        print("Estimate Array for Dimension ", dim, " : ", estimate[dim])

    fig, ax = plt.subplots(len(y_stream), 1)
    for dim in range(len(y_stream)):
        ax[dim].plot(t_stream, y_stream[dim])
        ax[dim].plot(t_stream, z_stream[dim])
    plt.show()
