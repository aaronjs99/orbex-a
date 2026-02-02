#!/usr/bin/env python3
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
ORBEX-A Entry Point

This script serves as the main entry point for running ADTMPC simulations.
Run with --help to see available options.
"""

import numpy as np
from orbexa.control.mpc import MPCController
from orbexa.core import params as p

def main():
    print("Initializing ORBEX-A Simulation...")
    
    # Initialize Controller
    controller = MPCController()
    
    # Setup Parameters
    # Using defaults from params.py
    
    # Initial State (6D: pos, vel) - slightly offset from target
    # Target is usually at origin or moving. 
    # Let's set chaser at [-50, 0, 0] relative
    X_0 = np.array([-50.0, 10.0, 5.0, 0.05, -0.01, 0.01])
    
    # Target State (Goal)
    # R-bar approach: target at origin
    X_f = np.zeros(6)
    
    # Initial Control
    U_0 = np.zeros(3)
    
    # Run Mission
    print("Starting Mission Loop...")
    try:
        history = controller.run_mission(
            operation="rendezvous",
            dt=p.dt,
            t_0=0.0,
            num_chasers=1,
            num_mpc_steps=p.numMPCSteps["rendezvous"],
            num_act_steps=p.numActSteps["rendezvous"],
            X_0=X_0,
            f_X_f=X_f,
            U_0=U_0,
            act_orbit_params=p.actOrbitParams,
            nom_orbit_params=p.nomOrbitParams,
            bounds=(p.stateBounds, p.inputBounds),
            max_mission_steps=5, # Short run for verification
            disp=True
        )
        print("Simulation Completed Successfully.")
        print(f"Steps simulated: {len(history['time'])}")
        
    except Exception as e:
        print(f"Simulation Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
