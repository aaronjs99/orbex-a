# /***********************************************************
# *                                                         *
# * Copyright (c) 2022                                      *
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

# PARAMETERS
## Simulation Parameters
decoupledMode     = True
debug_section     = True
debug_plots       = False

## Periapsis Time Anomalies ##
t_p               = 0.0
E_p               = 0.0
M_p               = 0.0
q_p               = 0.0

## General Orbital Parameters
dt                = (math.pi/20)/200
a                 = (6378.1363 + 300.00) * 1000
mu                = 3.986004418e+14
n                 = np.sqrt(mu/a**3)

## Orbital Parameters for MPC
actOrbitParams  = {'eccentricity' :  0.125,
                   'drag_alpha'   :  1.300e-7,
                   'drag_beta'    :  2.600e-7,}
initAdaptParams = {'eccentricity' : [0.000,   0.600  ],
                   'drag_alpha'   : [0.00e-7, 9.00e-7],
                   'drag_beta'    : [2.60e-7, 2.60e-7],}
nomOrbitParams  = {'eccentricity' :  0.000,
                   'drag_alpha'   :  0.000e-7,
                   'drag_beta'    :  2.600e-7,}
# initAdaptParams = {'eccentricity' : [1.1641033581e-1, 1.2533413227e-1],
#                    'drag_alpha'   : [1.2845496261e-7, 1.3231784092e-7],
#                    'drag_beta'    : [2.6000000000e-7, 2.6000000000e-7],}
# nomOrbitParams  = {'eccentricity' :  1.2502735059e-1,
#                    'drag_alpha'   :  1.2999902313e-7,
#                    'drag_beta'    :  2.6000000000e-7,}

## Chaser General Parameters
numChasers        =    1
initialTimeLapse  =  100
neighborMaxDist   = 1800

totalTime         =  100
numUpdateSteps    =  200
iterTime          = numUpdateSteps*dt
numMPCSteps       =  {'rendezvous': 80,
                      'docking'   : 40}
numActSteps       =  {'rendezvous': 20,
                      'docking'   :  2}

## Chaser Bounds
stateBounds       = [{"lower": "-Inf", "upper": "+Inf",}
                     for i in range(3)]
stateBounds.extend ([{"lower": "-Inf", "upper": "+Inf",}
                     for i in range(3)])
inputBounds       = [{"lower":   -1e5, "upper":    1e5,}
                     for i in range(3)]
forceBounds       = [{"lower":   -7e3, "upper":    7e3,}
                     for i in range(3)]
goalBounds        = (425.0, 475.0)

## Target Positional and Inertial Parameters
th_T0             = np.array( [  0.000,   0.200, - 0.100])
w_T0              = np.array( [- 0.040,   0.100, - 0.020])    # d(theta)/d(true_anomaly)
# th_T0             = np.array( [  0.400,   0.700,   0.100])
# w_T0              = np.array( [- 0.005,   0.000, - 0.010])  # d(theta)/d(true_anomaly)
# w_T0              = np.array( [- 0.100,   0.000, - 0.050]) 
I_T0              = np.array([[800.000,   2.000,   5.000,],
                              [  4.000, 450.000,   3.000,],
                              [  6.000,   8.000, 200.000,]])
pyramidalLimit    = {'mu_x' : 0.00,
                     'mu_y' : 0.00,}
radialLimit       =  1.0
targetShape       = 'cylinder'
if   targetShape == 'cylinder':
  targetLimit     = {'l_T'  : 0.60,
                     'r_T'  : 0.80,}
elif targetShape == 'ellipsoid':
  targetLimit     = {'r_Tx' : 0.30,
                     'r_Ty' : 0.25,
                     'r_Tz' : 0.40,}
targetCenter      = np.array([0, 0, 0])

## Target Deflection Parameters
TStopTime         =    5.0
chaserMinDist     =    0.0
discretizeDockers =   True
cylDiscPoints     =  []
for length in np.linspace(-targetLimit['l_T'], targetLimit['l_T'], 2):
  for theta in np.arange(0, 2*np.pi, np.pi/8):
    cylDiscPoints.append([targetCenter[0] + targetLimit['r_T']*np.cos(theta),
                          targetCenter[1] + targetLimit['r_T']*np.sin(theta),
                          targetCenter[2] + length])

## Tube MPC Parameters
tubeMPC           = {"runTube"      :  True,
                     "alpha_0"      : [   0.25,
                                          0.25,
                                          0.25,],
                     "omega_0"      : [   0.00,
                                          0.00,
                                          0.00,],  # Redefined for each iteration
                     "phi_0"        : [   0.50,
                                          0.50,
                                          0.50,],
                     "eta"          : [   0.10,
                                          0.10,
                                          0.10,],
                     "Lambda"       : [  60.00,
                                         60.00,
                                         60.00,],
                     "v_0"          : [   0.00,
                                          0.00,
                                          0.00,],
                     "alpha_range"  :  {"lower": 1.00e-2,
                                        "upper": 1.00e+1,}}