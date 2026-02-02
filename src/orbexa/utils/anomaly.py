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
import numpy as np
from matplotlib import pyplot as plt

from orbexa.core.params import *


## Calculate time from true anomaly
def convTh2T(th, eccentricity=nominal_orbit_params["eccentricity"], *args, **kwargs):
    t_0 = 0.0
    E_0 = 0.0
    try:
        while th > 2 * math.pi:
            th -= 2 * math.pi
            E_0 += 2 * math.pi
        while th < 0:
            th += 2 * math.pi
            E_0 -= 2 * math.pi
        sin_th = np.sin(th / 2)
        cos_th = np.cos(th / 2)
        E = 2 * np.arctan(
            np.sqrt((1 - eccentricity) / (1 + eccentricity)) * sin_th / cos_th
        )
        E -= 2 * math.pi * (sin_th < 0) * (cos_th < 0)
        E += 2 * math.pi * (sin_th > 0) * (cos_th < 0)
        E += E_0
    except:
        m = kwargs["m"]
        # rev_0 = m.Var(value = 0.0, fixed_initial = False, integer = True)
        # m.Equation(rev_0 <= (th/(2*math.pi)))
        # m.Equation(rev_0 > (th/(2*math.pi)-1))
        # E_0 = rev_0*2*math.pi
        # th_q1 = th - E_0
        # sin_th =  m.sin(th_q1/2)
        # cos_th =  m.cos(th_q1/2)
        sin_th = m.sin(th / 2)
        cos_th = m.cos(th / 2)
        E = 2 * m.atan(
            np.sqrt((1 - eccentricity) / (1 + eccentricity)) * sin_th / cos_th
        )
    try:
        M = E - eccentricity * np.sin(E)
    except:
        M = E - eccentricity * m.sin(E)
    t = t_p + t_0 + M / n

    returnVar = [t]
    if "returnM" in kwargs.keys() and kwargs["returnM"]:
        returnVar.append(M)
    if "returnE" in kwargs.keys() and kwargs["returnE"]:
        returnVar.append(E)
    if len(returnVar) == 1:
        return t
    return tuple(returnVar)


## Calculate the time derivative of true anomaly given true anomaly
def calcDThDT(th, eccentricity=nominal_orbit_params["eccentricity"], *args, **kwargs):
    t, M, E = convTh2T(th, eccentricity, returnM=True, returnE=True, *args, **kwargs)
    dMdt = n
    try:
        dEdt = dMdt / (1 - eccentricity * np.cos(E))
        dthdt = (
            dEdt
            * np.sqrt((1 + eccentricity) / (1 - eccentricity))
            * ((np.cos(th / 2) / np.cos(E / 2)) ** 2)
        )
    except:
        m = kwargs["m"]
        dEdt = dMdt / (1 - eccentricity * m.cos(E))
        dthdt = (
            dEdt
            * np.sqrt((1 + eccentricity) / (1 - eccentricity))
            * ((m.cos(th / 2) / m.cos(E / 2)) ** 2)
        )
    return dthdt


## Calculate the true anomaly derivative of time given true anomaly
def calcDTDTh(th, eccentricity=nominal_orbit_params["eccentricity"], *args, **kwargs):
    return 1 / calcDThDT(th, eccentricity, *args, **kwargs)


if __name__ == "__main__":
    th_0 = -3.00 * math.pi
    th_f = 3.00 * math.pi
    num_mpc_steps = 2000
    eccentricity = 0.1

    th_stream = np.linspace(th_0, th_f, num_mpc_steps)
    t_stream = np.array([convTh2T(th, eccentricity) for th in th_stream])
    dthdt_stream = np.array([calcDThDT(th, eccentricity) for th in th_stream])
    dtdth_stream = np.array([calcDTDTh(th, eccentricity) for th in th_stream])

    plt.plot(th_stream, t_stream)
    plt.show()
    plt.plot(th_stream, dthdt_stream)
    plt.show()
    plt.plot(th_stream, dtdth_stream)
    plt.show()
