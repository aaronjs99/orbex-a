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
import numpy as np
from gekko import GEKKO
from typing import Dict, List, Optional, Any, Union

from orbexa.utils import (
    gen_skew_sym_mat,
)
from orbexa.core.spacecraft import Target


# FUNCTION DEFINITIONS
## Position and Force Optimization
def target_deflect(
    target: Target,
    discretize_dockers: bool = True,
    ellipsoid_points: Optional[List[List[float]]] = None,
    cylinder_points: Optional[List[List[float]]] = None,
    *args,
    **kwargs,
):
    """
    Calculate target deflection strategy.

    Args:
        target: Target spacecraft object.
        discretize_dockers: Whether to optimize over discrete docking points.
        ellipsoid_points: List of available docking points on ellipsoid.
        cylinder_points: List of available docking points on cylinder.
        **kwargs: Must contain dt, x_f, bounds, numSteps, etc.
    """
    ### Unpack Parameters ###
    dt = kwargs["dt"]
    x_f = kwargs["x_f"]
    bounds = kwargs["bounds"]
    num_steps = kwargs["numSteps"]
    num_chasers = kwargs["num_chasers"]
    r_len, f_len = kwargs["rLen"], kwargs["fLen"]
    # chaser_min_dist = kwargs["chaser_min_dist"] # Unused?
    shape_params = kwargs["shapeParams"]
    solver_params = {
        "remote": kwargs.get("remote", False),
        "disp": kwargs.get("disp", False),
        "maxIter": kwargs.get("maxIter", 3000),
        "comp_time": False,
        "no_soln_disp": True,
    }
    mom_inertia = target.get_mom_inertia()
    inv_inertia = np.linalg.inv(mom_inertia)

    # Q and R matrices
    # Q = np.diag([0, 0, 0, 1, 1, 1]) * 1e3
    # R = np.eye(3) * 1e-4

    choice_sum_slack = 1e-7
    choice_ind_slack = 1e-7

    x_0 = np.concatenate((target.curr_state, target.angular_velocity))

    ### Unpacking System Parameters
    time_seq = np.linspace(0, num_steps * dt, num_steps)
    w = np.ones(num_steps)

    ## Initialize MPC ##
    solver = GEKKO(remote=solver_params["remote"])
    solver.time = time_seq
    w = np.ones(num_steps)
    final = np.zeros(num_steps)
    final[-1] = 1

    if "ellRadX" in shape_params.keys():
        target_shape = "ellipsoid"
        # ell_rad_x = shape_params["ellRadX"]
    elif "cylHeight" in shape_params.keys():
        target_shape = "cylinder"
        # cyl_height = shape_params["cylHeight"]

    ### Declaration of Gekko Variables
    eqs = []
    target_state = [
        solver.Var(value=x_0[i], fixed_initial=True) for i in range(len(x_0))
    ]  # Target State
    chaser_forces_flat = [
        solver.Var(value=0, fixed_initial=False) for i in range(f_len * num_chasers)
    ]  # Chaser Force
    W = solver.Param(value=w)
    final = solver.Param(value=final)

    if discretize_dockers:
        chaser_positions_flat = [
            solver.Var(value=0.01, fixed_initial=False)
            for i in range(r_len * num_chasers)
        ]  # Chaser Position
    else:
        chaser_positions_flat = [
            solver.FV(value=1, fixed_initial=False) for i in range(r_len * num_chasers)
        ]  # Chaser Position

    ## Constraint Equations ##
    # theta = x[0:3]
    # omega = np.array(x[3:6]) # Not directly usable like this with Gekko lists

    # Organize vars per agent
    chaser_positions = [
        chaser_positions_flat[agent * r_len : (agent + 1) * r_len]
        for agent in range(num_chasers)
    ]
    chaser_forces = [
        chaser_forces_flat[agent * f_len : (agent + 1) * f_len]
        for agent in range(num_chasers)
    ]

    # Torque calculation requires cross product
    # Manual cross product for GEKKO vars
    # torque = r x f
    torques = []
    for agent in range(num_chasers):
        rx, ry, rz = (
            chaser_positions[agent][0],
            chaser_positions[agent][1],
            chaser_positions[agent][2],
        )
        fx, fy, fz = (
            chaser_forces[agent][0],
            chaser_forces[agent][1],
            chaser_forces[agent][2],
        )
        tx = ry * fz - rz * fy
        ty = rz * fx - rx * fz
        tz = rx * fy - ry * fx
        torques.append([tx, ty, tz])

    # Sum of torques
    total_torque = [0, 0, 0]
    for i in range(3):
        for agent in range(num_chasers):
            total_torque[i] += torques[agent][i]

    ### Discretized Docking Constraints ###
    if discretize_dockers:
        r_options = []
        if target_shape == "ellipsoid":
            r_options = ellipsoid_points if ellipsoid_points else []
        elif target_shape == "cylinder":
            r_options = cylinder_points if cylinder_points else []

        if not r_options:
            raise ValueError("Discrete dockers requested but no points provided.")

        num_options = len(r_options)
        r_choices = [
            solver.FV(value=0.01, fixed_initial=False)
            for i in range(num_options * num_chasers)
        ]

        for agent in range(num_chasers):
            r_choice = r_choices[agent * num_options : (agent + 1) * num_options]

            # Sum of choices should be 1 (select exactly one point)
            eqs.append(solver.sum(r_choice) > 1.000 - choice_sum_slack)
            eqs.append(solver.sum(r_choice) < 1.000 + choice_sum_slack)

            # Binary relaxation constraint (x^2 ≈ x -> x near 0 or 1)
            for j in range(len(r_choice)):
                eqs.append(r_choice[j] * r_choice[j] - r_choice[j] < choice_ind_slack)
                eqs.append(r_choice[j] * r_choice[j] - r_choice[j] > -choice_ind_slack)

            # Position definition
            for i in range(3):  # x, y, z
                # chaser_positions[agent][i] = sum(option[j][i] * choice[j])
                pos_val = 0
                for j in range(num_options):
                    pos_val += r_options[j][i] * r_choice[j]
                eqs.append(chaser_positions[agent][i] == pos_val)

    # Force bounds
    if bounds:
        for idx, force_var in enumerate(chaser_forces_flat):
            bound = bounds[idx % len(bounds)]
            if bound.get("lower") not in ["-Inf", None, float("-inf")]:
                force_var.lower = bound["lower"]
            if bound.get("upper") not in ["+Inf", None, float("inf")]:
                force_var.upper = bound["upper"]

    # Euler rotational dynamics: theta_dot = omega,
    # I omega_dot = tau - omega x I omega.
    omega = target_state[3:6]
    inertia_omega = [
        sum(mom_inertia[i, j] * omega[j] for j in range(3)) for i in range(3)
    ]
    gyroscopic = [
        omega[1] * inertia_omega[2] - omega[2] * inertia_omega[1],
        omega[2] * inertia_omega[0] - omega[0] * inertia_omega[2],
        omega[0] * inertia_omega[1] - omega[1] * inertia_omega[0],
    ]
    angular_accel_rhs = [
        total_torque[i] - gyroscopic[i] for i in range(3)
    ]
    angular_accel = [
        sum(inv_inertia[i, j] * angular_accel_rhs[j] for j in range(3))
        for i in range(3)
    ]

    for i in range(3):
        eqs.append(target_state[i].dt() == target_state[i + 3])
        eqs.append(target_state[i + 3].dt() == angular_accel[i])

    terminal_error = sum((target_state[i] - x_f[i]) ** 2 for i in range(len(x_0)))
    force_effort = sum(force_i**2 for force_i in chaser_forces_flat)

    solver.Equations(eqs)
    solver.Minimize(final * terminal_error + 1.0e-4 * W * force_effort)
    solver.options.IMODE = 6
    solver.options.SOLVER = kwargs.get("solver", 3)
    solver.options.MAX_ITER = solver_params["maxIter"]

    try:
        solver.solve(disp=solver_params["disp"])
        success = solver.options.APPSTATUS == 1
        objective = solver.options.OBJFCNVAL
    except Exception as exc:
        solver.cleanup()
        return {
            "success": False,
            "message": str(exc),
            "target_state": None,
            "chaser_positions": None,
            "chaser_forces": None,
        }

    result = {
        "success": success,
        "objective": objective,
        "target_state": np.array([state_i.value for state_i in target_state]),
        "chaser_positions": np.array(
            [position_i.value for position_i in chaser_positions_flat]
        ),
        "chaser_forces": np.array([force_i.value for force_i in chaser_forces_flat]),
    }

    solver.cleanup()
    return result
