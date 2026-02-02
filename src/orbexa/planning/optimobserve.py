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

# PACKAGE IMPORTS
import math
import time
import numpy as np
from gekko import GEKKO
from functools import partial

try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL

    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    fmin = tpe = hp = None
    STATUS_OK = "ok"
    STATUS_FAIL = "fail"

import orbexa.core.params as p
from orbexa.estimation.enclosures import min_enclosing_ellipsoid

np.random.seed(1)


# FUNCTION DEFINITIONS
def calcShape(radius, target_shape, sphere_shape):
    kappa = math.exp(0.01 * radius) - 1
    shape = kappa * target_shape + (1 - kappa) * sphere_shape
    return shape


def calcSingleObservation(r1, r2, R, shape):
    r12 = np.subtract(r1, r2)
    return np.sum([r12[i] ** 2 for i in range(len(r12))])


def calcMutualObservation(theta_1, theta_2, phi_1, phi_2, *args, **kwargs):
    m = kwargs["m"]
    delTheta = theta_1 - theta_2
    delPhi = phi_1 - phi_2
    return (1 + m.cos(delTheta)) * (1 - m.sin(m.abs2(delPhi) / 4)) / 2


def calcTotalLocalObservation(numChasers, r, R, shape):
    observation = 0.0
    numDims = len(r) // numChasers
    for j in range(numChasers):
        for k in range(j + 1, numChasers):
            observation += (
                calcSingleObservation(
                    r[numDims * j : numDims * (j + 1)],
                    r[numDims * k : numDims * (k + 1)],
                    R,
                    shape,
                )
                / np.sum(R) ** 2
            )
    return observation


def calcTotalCrossObservation(numChasers, r1, r2, R1, R2, shape):
    observation = 0.0
    numDims = len(r1) // numChasers
    for i in range(numChasers):
        observation -= calcSingleObservation(
            r1[numDims * i : numDims * (i + 1)],
            r2[numDims * i : numDims * (i + 1)],
            R1,
            shape,
        )
    return observation


def calcTotalMutualObservation(numChasers, theta, phi, R, shape):
    observation = 0.0
    for i in range(numChasers):
        for j in range(i + 1, numChasers):
            observation -= (
                calcMutualObservation(theta[i], theta[j], phi[i], phi[j], m=GEKKO())
                / np.sum(R) ** 2
            )
    return observation


def observeOptimizer(
    hyperopt_params,
    numDims=3,
    Q=[1.00 for i in range(5)],
    r_0=None,
    numLevels=3,
    numChasers=1,
    startRadii=[100.00, 100.00, 100.00],
    finalRadii=[1.00, 1.00, 1.00],
    target_shape=[],
    *args,
    **kwargs,
):
    Rho_0 = hyperopt_params["Rho_0"]
    Q = hyperopt_params["Q"]
    if r_0 is None:
        r_0 = [0.0 for dim in range(numDims * numChasers * 1)]
        for chaser in range(numChasers):
            r_0[0 + numDims * chaser + 0] = startRadii[0]

    m = GEKKO(remote=True)

    Rho = [[0 for dim in range(numDims)] for i in range(numLevels)]
    for level in range(numLevels):
        for dim in range(numDims):
            if level == 0:
                Rho[level][dim] = m.Param(value=startRadii[dim])
            elif level == numLevels - 1:
                Rho[level][dim] = m.Param(value=finalRadii[dim])
            else:
                # if 'Rho_0' not in kwargs:
                #   Rho[level][dim] = m.Var  (value = (startRadii[dim] + 2*finalRadii[dim]) / 3)
                # else:
                Rho[level][dim] = m.Var(value=Rho_0[level][dim])

    # Generate theta_0 and phi_0 from r_0
    theta_0 = [0.0 for chaser in range(numChasers)]
    phi_0 = [0.0 for chaser in range(numChasers)]
    for chaser in range(numChasers):
        if numDims == 2:
            theta_0[chaser] = math.atan2(
                r_0[numDims * chaser + 1], r_0[numDims * chaser + 0]
            )
            phi_0[chaser] = math.atan2(
                r_0[numDims * chaser + 2], r_0[numDims * chaser + 0]
            )
        elif numDims == 3:
            theta_0[chaser] = math.atan2(
                r_0[numDims * chaser + 1], r_0[numDims * chaser + 0]
            )
            phi_0[chaser] = math.atan2(
                r_0[numDims * chaser + 2], r_0[numDims * chaser + 0]
            )
    theta = [m.Var() for i in range(numChasers * numLevels)]
    phi = [m.Var() for i in range(numChasers * numLevels)]
    # r = [m.Var() for i in range(numDims*numChasers*numLevels)]
    r = [0 for i in range(numDims * numChasers * numLevels)]
    for level, rho in enumerate(Rho):
        for chaser in range(numChasers):
            if level == 0:
                # for dim in range(numDims):
                #   r[numDims*numChasers*level + numDims*chaser + dim].lower = r_0[numDims*numChasers*level + numDims*chaser + dim] - 1e-3
                #   r[numDims*numChasers*level + numDims*chaser + dim].upper = r_0[numDims*numChasers*level + numDims*chaser + dim] + 1e-3
                m.Equation(theta[numChasers * level + chaser] == theta_0[chaser])
                m.Equation(phi[numChasers * level + chaser] == phi_0[chaser])
            # m.Equation(np.sum([(r[numDims*numChasers*level + numDims*chaser + dim]**2)/(rho[dim]**2)
            #                    for dim in range(numDims)]) - 1**2 <  1e-3)
            # m.Equation(np.sum([(r[numDims*numChasers*level + numDims*chaser + dim]**2)/(rho[dim]**2)
            #                    for dim in range(numDims)]) - 1**2 > -1e-3)
            if numDims == 2:
                r[numDims * numChasers * level + numDims * chaser + 0] = rho[0] * m.cos(
                    theta[numChasers * level + chaser]
                )
                r[numDims * numChasers * level + numDims * chaser + 1] = rho[1] * m.sin(
                    theta[numChasers * level + chaser]
                )
            elif numDims == 3:
                r[numDims * numChasers * level + numDims * chaser + 0] = (
                    rho[0]
                    * m.sin(phi[numChasers * level + chaser])
                    * m.cos(theta[numChasers * level + chaser])
                )
                r[numDims * numChasers * level + numDims * chaser + 1] = (
                    rho[1]
                    * m.sin(phi[numChasers * level + chaser])
                    * m.sin(theta[numChasers * level + chaser])
                )
                r[numDims * numChasers * level + numDims * chaser + 2] = rho[2] * m.cos(
                    phi[numChasers * level + chaser]
                )

    for level in range(numLevels - 1):
        for dim in range(numDims):
            m.Equation(Rho[level][dim] > Rho[level + 1][dim])

    totLocalObs = [
        m.Intermediate(
            calcTotalLocalObservation(
                numChasers,
                r[numDims * numChasers * i : numDims * numChasers * (i + 1)],
                Rho[i],
                target_shape,
            )
        )
        for i in range(numLevels)
    ]
    finLocalObs = m.Intermediate(
        calcTotalLocalObservation(
            numChasers,
            r[
                numDims
                * numChasers
                * (numLevels - 1) : numDims
                * numChasers
                * (numLevels - 0)
            ],
            Rho[numLevels - 1],
            target_shape,
        )
    )
    totCrossObs = [
        m.Intermediate(
            calcTotalCrossObservation(
                numChasers,
                r[numDims * numChasers * (i + 0) : numDims * numChasers * (i + 1)],
                r[numDims * numChasers * (i + 1) : numDims * numChasers * (i + 2)],
                Rho[i],
                Rho[i + 1],
                target_shape,
            )
        )
        for i in range(numLevels - 1)
    ]
    totMutualObs = [
        m.Intermediate(
            calcTotalMutualObservation(
                numChasers,
                theta[numChasers * i : numChasers * (i + 1)],
                phi[numChasers * i : numChasers * (i + 1)],
                Rho[i],
                target_shape,
            )
        )
        for i in range(numLevels)
    ]
    totCrossRad = [
        m.Intermediate(
            np.sum(
                [(Rho[level][dim] - Rho[level + 1][dim]) ** 2 for dim in range(numDims)]
            )
        )
        for level in range(numLevels - 1)
    ]
    totLocalRad = [
        m.Intermediate(
            np.sum(
                [
                    (Rho[level][dim % 3] - Rho[level][(dim + 1) % 3]) ** 2
                    for dim in range(numDims)
                ]
            )
        )
        for level in range(numLevels - 1)
    ]
    # m.Maximize(Q[0]*np.sum(totLocalObs) + Q[1]*finLocalObs + Q[2]*np.sum(totCrossObs) + Q[3]*np.sum(totCrossRad) + Q[4]*np.sum(totLocalRad))
    m.Maximize(
        Q[0] * np.sum(totLocalObs)
        + Q[1] * finLocalObs
        + Q[2] * np.sum(totCrossObs)
        + Q[3] * np.sum(totMutualObs)
        + Q[4] * np.sum(totCrossRad)
        + Q[5] * np.sum(totLocalRad)
    )

    m.options.SOLVER = 3
    m.options.IMODE = 3
    m.options.MAX_ITER = 500
    try:
        m.solve(disp=False, debug=2)
        status = STATUS_OK
        Rho = [[rho[dim].value[0] for dim in range(numDims)] for rho in Rho]
        theta = [theta_i.value[0] for theta_i in theta]
        phi = [phi_i.value[0] for phi_i in phi]
        # r   = [ r_i     .value[0]                            for r_i in   r]
        for level in range(numLevels):
            for chaser in range(numChasers):
                if numDims == 2:
                    r[numDims * numChasers * level + numDims * chaser + 0] = Rho[level][
                        0
                    ] * math.cos(theta[numChasers * level + chaser])
                    r[numDims * numChasers * level + numDims * chaser + 1] = Rho[level][
                        1
                    ] * math.sin(theta[numChasers * level + chaser])
                elif numDims == 3:
                    r[numDims * numChasers * level + numDims * chaser + 0] = (
                        Rho[level][0]
                        * math.sin(phi[numChasers * level + chaser])
                        * math.cos(theta[numChasers * level + chaser])
                    )
                    r[numDims * numChasers * level + numDims * chaser + 1] = (
                        Rho[level][1]
                        * math.sin(phi[numChasers * level + chaser])
                        * math.sin(theta[numChasers * level + chaser])
                    )
                    r[numDims * numChasers * level + numDims * chaser + 2] = Rho[level][
                        2
                    ] * math.cos(phi[numChasers * level + chaser])
        observation = m.options.OBJFCNVAL
    except Exception as e:
        print("!!! Failed to solve : ", str(e), " !!!")
        status = STATUS_FAIL
        Rho = Rho_0
        r = [0.0 for dim in range(numDims * numChasers * numLevels)]
        observation = 0.0

    m.cleanup()
    del m

    return {"loss": -observation, "status": status, "x": (Rho, r, Q)}


# MAIN PROGRAM
if __name__ == "__main__":
    ## Initialize Parameters
    Q = [1.000e0, 0.000e-8, 1.000e-2, 1.000e-2, 0.000e-8, 0.000e-8]
    Q_range = [
        {"func": lambda l, x, y: x, "p1": 1.000e0, "p2": 0.000e0},
        {"func": lambda l, x, y: x, "p1": 0.000e0, "p2": 0.000e0},
        {"func": hp.uniform, "p1": 0.000e-1, "p2": 1.000e-1},
        {"func": lambda l, x, y: x, "p1": 0.000e0, "p2": 0.000e0},
        {"func": lambda l, x, y: x, "p1": 0.000e0, "p2": 0.000e0},
        {"func": lambda l, x, y: x, "p1": 0.000e0, "p2": 0.000e0},
    ]
    numDims = 3
    numLevels = 5
    numChasers = 4
    ## Initialize Initial and Final Radii
    targetPoints = []
    for x in [p.targetLimit["r_T"], 0.0, -p.targetLimit["r_T"]]:
        for y in [p.targetLimit["r_T"], 0.0, -p.targetLimit["r_T"]]:
            for z in [p.targetLimit["l_T"], 0.0, -p.targetLimit["l_T"]]:
                if (x != 0.0 or y != 0.0 or z != 0.0) and not (
                    x**2 + y**2 > p.targetLimit["r_T"] ** 2
                ):
                    targetPoints.append([x, y, z])
    startRadii = [
        137.50,
        137.50,
        137.50,
    ]
    Q_bar, center_bar = min_enclosing_ellipsoid(targetPoints)
    for dim in range(numDims):
        if finalRadii[dim] > startRadii[dim]:
            raise Exception("!!! Final radius is greater than start radius !!!")
        elif (
            finalRadii[dim]
            <= [p.targetLimit["r_T"], p.targetLimit["r_T"], p.targetLimit["l_T"]][dim]
        ):
            raise Exception(
                "!!! Final radius is less than or equal to target limit !!!"
            )
        elif startRadii[dim] <= 0:
            raise Exception("!!! Start radius is less than or equal to zero !!!")
    ## Initialize Initial Positions
    r_0 = [0.0 for dim in range(numDims * numChasers * 1)]
    for chaser in range(numChasers):
        if numDims == 2:
            q = [np.random.uniform(0, 2 * math.pi)]
            f = [lambda q: math.cos(q[0]), lambda q: math.sin(q[0])]
        elif numDims == 3:
            q = [np.random.uniform(0, 2 * math.pi), np.random.uniform(0, math.pi)]
            f = [
                lambda q: math.sin(q[1]) * math.cos(q[0]),
                lambda q: math.sin(q[1]) * math.sin(q[0]),
                lambda q: math.cos(q[1]),
            ]
        for dim in range(numDims):
            r_0[numDims * numChasers * 0 + numDims * chaser + dim] = startRadii[
                dim
            ] * f[dim](q)
    target_shape = []

    Rho_0_space = [
        [
            hp.uniform(
                "Rho_0[" + str(level) + "][" + str(dim) + "]",
                finalRadii[dim],
                startRadii[dim],
            )
            for dim in range(numDims)
        ]
        for level in range(numLevels)
    ]
    Q_space = [
        Q_i["func"]("Q[" + str(i) + "]", Q_i["p1"], Q_i["p2"])
        for i, Q_i in enumerate(Q_range)
    ]
    fn = partial(
        observeOptimizer,
        numDims=numDims,
        Q=Q,
        r_0=r_0,
        numLevels=numLevels,
        numChasers=numChasers,
        startRadii=startRadii,
        finalRadii=finalRadii,
    )
    best = fmin(
        fn=fn,
        space={
            "Rho_0": Rho_0_space,
            "Q": Q_space,
        },
        algo=tpe.suggest,
        max_evals=10,
    )
    Rho_0 = [[0 for dim in range(numDims)] for i in range(numLevels)]
    for level in range(numLevels):
        for dim in range(numDims):
            Rho_0[level][dim] = best["Rho_0[" + str(level) + "][" + str(dim) + "]"]
    for i in range(len(Q)):
        try:
            Q[i] = best["Q[" + str(i) + "]"]
        except:
            try:
                Q[i] = Q_range[i]["p1"]
            except:
                pass
    sol = fn(
        hyperopt_params={
            "Rho_0": Rho_0,
            "Q": Q,
        }
    )
    Rho, r, Q = sol["x"]
    observation = -sol["loss"]

    print()
    print("Optimization Parameters : ", Q)
    print()
    for i, rho in enumerate(Rho):
        print("Radii[", i, "]  : ", rho)
    print()
    print("Observation : ", observation)
    print()
    for rho in range(numLevels):
        for chaser in range(numChasers):
            print(
                "chaser[",
                chaser,
                "] at radius number ",
                rho,
                " : ",
                [
                    r[numDims * numChasers * rho + numDims * chaser + dim]
                    for dim in range(numDims)
                ],
            )
        print()

    x, y, z = [
        [
            [
                r[numDims * numChasers * rho + numDims * chaser + dim]
                for rho in range(numLevels)
            ]
            for chaser in range(numChasers)
        ]
        for dim in range(numDims)
    ]

    if numDims == 3:
        from orbitsim import simulate

        simulate(
            "../plots/optimobserve.html",
            x,
            y,
            z,
            labels=[chaser for chaser in range(numChasers)],
            numAgents=numChasers,
            shape=[
                {
                    "type": "ellipsoid",
                    "center": np.array([0, 0, 0]),
                    "radii": rho,
                    "opacity": 0.1,
                }
                for rho in Rho
            ],
            lines=True,
        )
