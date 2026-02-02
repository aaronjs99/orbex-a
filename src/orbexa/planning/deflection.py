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
import time
import numpy as np
from gekko import GEKKO

import orbexa.core.params as p
from orbexa.core.params import *
from orbexa.utils import (
    genSkewSymMat,
    cylinderRadialUpperConstraint,
    cylinderRadialLowerConstraint,
    cylinderAxialUpperConstraint,
    cylinderAxialLowerConstraint,
)
from orbexa.core.spacecraft import Target
from orbexa.visualization.orbitsim import deflection_plot


# FUNCTION DEFINITIONS
## Position and Force Optimization
def targetDeflect(target, *args, **kwargs):
    ### Unpack Parameters ###
    dt = kwargs["dt"]
    x_f = kwargs["x_f"]
    bounds = kwargs["bounds"]
    numSteps = kwargs["numSteps"]
    numChasers = kwargs["numChasers"]
    rLen, fLen = kwargs["rLen"], kwargs["fLen"]
    chaserMinDist = kwargs["chaserMinDist"]
    shapeParams = kwargs["shapeParams"]
    solverParams = {
        "remote": True,
        "disp": True,
        "maxIter": 3000,
        "comp_time": False,
        "no_soln_disp": True,
    }
    momInertia = target.getMomInertia()
    invInertia = np.linalg.inv(momInertia)
    Q = np.diag([0, 0, 0, 1, 1, 1]) * 1e3
    R = np.eye(3) * 1e-4
    # R = np.zeros((3, 3))
    sigma_F = 4e9
    choice_sum_slack = 1e-7
    choice_ind_slack = 1e-7

    x_0 = np.array(
        [list(target.currState), list(target.angularVelocity)]
    ).flatten()  # Target State

    ### Unpacking System Parameters
    timeSeq = np.linspace(0, numSteps * dt, numSteps)
    w = np.ones(numSteps)

    ## Initialize MPC ##
    if True:
        m = GEKKO(remote=solverParams["remote"])
        m.time = timeSeq
        w = np.ones(numSteps)
        final = np.zeros(numSteps)
        final[-1] = 1

    if "ellRadX" in shapeParams.keys():
        targetShape = "ellipsoid"
        ellRadX = shapeParams["ellRadX"]
        ellRadY = shapeParams["ellRadY"]
        ellRadZ = shapeParams["ellRadZ"]
        ellCenter = shapeParams["ellCenter"]
    elif "cylHeight" in shapeParams.keys():
        targetShape = "cylinder"
        cylHeight = shapeParams["cylHeight"]
        cylRadius = shapeParams["cylRadius"]
        cylCenter = shapeParams["cylCenter"]

    ### Declaration of Gekko Variables
    if True:
        eqs = []
        x = [
            m.Var(value=x_0[i], fixed_initial=True) for i in range(len(x_0))
        ]  # Target State
        f = [
            m.Var(value=0, fixed_initial=False) for i in range(fLen * numChasers)
        ]  # Chaser Force
        W = m.Param(value=w)
        final = m.Param(value=final)
        from orbexa.core.params import discretizeDockers

        if discretizeDockers:
            r = [
                m.Var(value=0.01, fixed_initial=False) for i in range(rLen * numChasers)
            ]  # Chaser Position
        else:
            r = [
                m.FV(value=1, fixed_initial=False) for i in range(rLen * numChasers)
            ]  # Chaser Position

    ## Constraint Equations ##
    if True:
        theta = x[0:3]
        omega = np.array(x[3:6])
        state = [r[agent * rLen : (agent + 1) * rLen] for agent in range(numChasers)]
        force = [f[agent * fLen : (agent + 1) * fLen] for agent in range(numChasers)]
        torque = np.array(
            [np.cross(state[agent], force[agent]) for agent in range(numChasers)]
        )

    ### Discretized Docking Constraints ###
    if discretizeDockers:
        if targetShape == "ellipsoid":
            from orbexa.core.params import ellDiscPoints as r_options
        elif targetShape == "cylinder":
            from orbexa.core.params import cylDiscPoints as r_options
        numOptions = len(r_options)
        r_choices = [
            m.FV(value=0.01, fixed_initial=False)
            for i in range(numOptions * numChasers)
        ]
        for agent in range(numChasers):
            r_choice = r_choices[agent * numOptions : (agent + 1) * numOptions]
            eqs.append(np.sum(r_choice) > 1.000 - choice_sum_slack)
            eqs.append(np.sum(r_choice) < 1.000 + choice_sum_slack)
            for j in range(len(r_choice)):
                eqs.append(r_choice[j] * r_choice[j] - r_choice[j] < choice_ind_slack)
                eqs.append(r_choice[j] * r_choice[j] - r_choice[j] > -choice_ind_slack)
            for i in range(len(state[agent])):
                eqs.append(
                    state[agent][i]
                    == np.sum(
                        [r_options[j][i] * r_choice[j] for j in range(numOptions)]
                    )
                )

    ### Input Bounds ###
    forceBounds = bounds
    for agent in range(len(force)):
        for i in range(fLen):
            if forceBounds[i]["lower"] != "-Inf":
                eqs.append(force[agent][i] > forceBounds[i]["lower"])
            if forceBounds[i]["upper"] != "+Inf":
                eqs.append(force[agent][i] < forceBounds[i]["upper"])

    ### Dynamics Constraints ###
    for i in range(3):
        eqs.append(theta[i].dt() == omega[i])
        eqs.append(
            omega[i].dt()
            == ((invInertia @ genSkewSymMat(omega)) @ (momInertia @ omega))[i]
            + (invInertia @ np.sum(torque, axis=0))[i]
        )

    ### Target Shape Constraints ###
    if not discretizeDockers:
        for geometryIneq in target.geometry["Ineqs"]:
            for agent in range(numChasers):
                eqs.append(geometryIneq(state[agent]) <= 0)

    ### Collision Avoidance Constraints ###
    for i in range(numChasers):
        for j in range(numChasers):
            if i > j and chaserMinDist != 0.0:
                eqs.append(
                    np.sum([(state[i][k] - state[j][k]) ** 2 for k in range(rLen)])
                    >= chaserMinDist**2
                )

    ## Objective Function Definition ##
    if True:
        intErrorArr = []
        finErrorArr = []
        for i in range(3):
            intErrorArr.append(x @ Q @ x)
        for agent in range(numChasers):
            intErrorArr.append(force[agent] @ R @ force[agent])
        finErrorArr.append(sigma_F * np.sum([omega[i] ** 2 for i in range(3)]))
        intError = np.sum(intErrorArr)
        finError = np.sum(finErrorArr)

    ## Solver Parameters ##
    if True:
        eqs = m.Equations(eqs)
        m.Minimize(W * intError + final * finError)
        m.options.OTOL = 1e-7
        m.options.RTOL = 1e-7
        m.options.IMODE = 6
        # m.options.REDUCE     =    1
        m.options.SOLVER = 1
        m.options.MAX_ITER = solverParams["maxIter"]
        # m.options.MAX_MEMORY =  512
        # m.options.DIAGLEVEL  =    0
        # m.options.COLDSTART  =    0
        # m.options.TIME_SHIFT =    0

    ## Solve MPC ##
    startTime = time.time()
    try:
        m.solve(disp=solverParams["disp"], debug=2)
        x = [x[i].value for i in range(len(x))]
        r = [r[i].value[-1] for i in range(len(r))]
        r = [r[rLen * agent : rLen * (agent + 1)] for agent in range(numChasers)]
        f = [f[i].value for i in range(len(f))]
        f = [f[fLen * agent : fLen * (agent + 1)] for agent in range(numChasers)]
    except:
        if solverParams["no_soln_disp"]:
            print("Optimization Solution Not Found")
            print("Constraints: ", (x_0, x_f))
            print("Bounds: ", bounds)
        x = [0 for i in range(len(x))]
        r = [[0 for i in range(rLen)] for agent in range(numChasers)]
        f = [[[0] for i in range(rLen)] for agent in range(numChasers)]
    stopTime = time.time()

    timing = stopTime - startTime
    if solverParams["comp_time"]:
        print("Solution Computation Time = ", str(timing), " s")

    theta = np.array(x[:3])
    omega = np.array(x[3:])
    newAngularPos = list(np.transpose(theta))
    newAngularVel = list(np.transpose(omega))
    target.updateState(newAngularPos=newAngularPos, newAngularVel=newAngularVel)

    print("Final Target Angular Position      : ", target.currState)
    print("Final Target Angular Velocity      : ", target.angularVelocity)
    print(
        "Final Target Angular Velocity Norm = ", np.linalg.norm(target.angularVelocity)
    )
    print("Chaser Positions :")
    for agent in range(numChasers):
        print("  Chaser " + str(agent + 1) + " : ")
        p_chaser = np.array(r[rLen * agent : rLen * (agent + 1)])
        p_target = cylDiscPoints[
            np.argmin([np.linalg.norm(np.subtract(p_chaser, p)) for p in cylDiscPoints])
        ]
        print("    Final    Chaser Position       : ", p_chaser)
        print("    Intended Chaser Position       : ", p_target)
        print(
            "    Error in Chaser Position       = ",
            np.linalg.norm(np.subtract(p_chaser, p_target)),
        )

    return target, x, r, f


# MAIN PROGRAM
if __name__ == "__main__":
    # dt         = (math.pi/200)     # Override over params.py
    # numChasers = 1
    rLen, fLen = 3, 3

    T = Target(
        {"name": "Target", "numStates": 3, "dt": dt, "initState": th_T0},
        {"angularVelocity": w_T0, "momInertia": I_T0},
    )
    targetShape = "cylinder"
    T.updateGeometry(
        geometryIneqs=[
            cylinderRadialUpperConstraint,
            cylinderRadialLowerConstraint,
            cylinderAxialUpperConstraint,
            cylinderAxialLowerConstraint,
        ],
    )

    initialTimeLapse = 100
    TStopSteps = 180
    TStopTime = TStopSteps * dt
    T.updateState(numSteps=initialTimeLapse)

    T, x, r, f = targetDeflect(
        target=T,
        dt=dt,
        x_f=np.array([0, 0, 0, 0, 0, 0]),
        bounds=(forceBounds),
        numSteps=TStopSteps,
        numChasers=numChasers,
        rLen=rLen,
        fLen=fLen,
        chaserMinDist=chaserMinDist,
        shapeParams={
            "cylHeight": targetLimit["l_T"],
            "cylRadius": targetLimit["r_T"],
            "cylCenter": targetCenter,
        },
    )

    deflection_plot(
        T,
        x,
        r,
        f,
        plotFlags={
            "plot_target": True,
            "plot_position": True,
            "plot_force": True,
        },
        numChasers=numChasers,
        dt=dt,
        numSteps=TStopSteps,
        shapeParams={
            "cylHeight": targetLimit["l_T"],
            "cylRadius": targetLimit["r_T"],
            "cylCenter": targetCenter,
        },
        targetShape=targetShape,
        initialTimeLapse=initialTimeLapse,
    )
