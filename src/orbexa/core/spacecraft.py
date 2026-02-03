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
ORBEX-A Spacecraft Classes.

This module defines the core agent classes:
- Spacecraft: Base class for all agents.
- Target: Passive object to be inspected/docked with.
- Chaser: Active agent performing rendezvous/docking.
"""

import math
import time
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from itertools import count
from scipy import optimize as opt
from scipy import integrate as intg
from typing import List, Dict, Any, Tuple, Optional, Union

# Core Imports
from orbexa.core.dynamics import cwh_equations
from orbexa.core.config import SimulationConfig
from orbexa.utils import gen_shape_data, gen_skew_sym_mat, calc_global_occlusion_cost
from orbexa.solvers import get_solver, MPCProblem
from orbexa.control.mpc_problem_builder import build_from_dynamics

logger = logging.getLogger(__name__)


random.seed(0)


# =============================================================================
# Helper Functions
# =============================================================================


def target_state_update_func(
    t, state, dt, mom_inertia, skew_sym_mat, torque_val, torque_type, *args, **kwargs
):
    """Dynamics function for target attitude propagation using solve_ivp."""
    angular_pos = state[:3]
    angular_vel = state[3:]

    if torque_type == "zero":
        torque = np.zeros(3)
    elif torque_type == "given":
        idx = min(int(np.floor(t / dt)), len(torque_val) - 1)
        torque = torque_val[idx]
    elif torque_type == "function":
        torque_val_eval = torque_val(state, mom_inertia)
        torque = torque_val_eval["torque"]
    else:
        torque = np.zeros(3)

    # Recomputing skew matrix for current angular velocity
    s_w = gen_skew_sym_mat(angular_vel)

    # Euler's rotation computations
    # dw/dt = I_inv * (torque - w x (I * w))
    i_inv = np.linalg.inv(mom_inertia)

    w_dot = np.matmul(
        np.matmul(i_inv, skew_sym_mat), np.matmul(mom_inertia, angular_vel)
    ) + np.matmul(i_inv, torque)

    return np.concatenate((angular_vel, w_dot))


# =============================================================================
# Class Definitions
# =============================================================================


class Spacecraft:
    """Base class for all spacecraft agents."""

    _ids = count(0)

    def __init__(self, config: SimulationConfig, **kwargs):
        """
        Initialize Spacecraft.

        Args:
            config: SimulationConfig object.
            **kwargs: Overrides for specific properties like 'name' or 'initState'.
        """
        self.config = config
        self.anom_step = config.anom_step

        self.id = next(self._ids)
        self.name = kwargs.get("name", "")
        self.num_states = kwargs.get("numStates", 6)
        self.init_state = kwargs.get("initState", np.zeros(self.num_states))

        self.curr_state = self.init_state
        self.state_history = [self.curr_state]

    def update_state(self, *args):
        """Update state. distinct for Target vs Chaser."""
        raise NotImplementedError("Spacecraft.update_state() is not implemented.")

    def plot_state_history(self, params: Dict, *args, **kwargs):
        """Plot the history of states."""
        ind_state_history = np.transpose(self.state_history)
        cmap = plt.cm.get_cmap("viridis", 256)

        plt.figure(figsize=(4, 3))
        # time_seq = [step * self.dt for step in range(len(ind_state_history[0]))]
        # Fixed: range gives ints, multiply by dt
        num_points = ind_state_history.shape[1]
        time_seq = np.linspace(0, num_points * self.dt, num_points)

        if params.get("sep_plots", False):
            for i in range(self.num_states):
                plt.subplot(self.num_states, 1, i + 1)
                label = f"$x_{i}$"
                plt.plot(time_seq, ind_state_history[i], c=cmap(96 * i), label=label)
                plt.legend()
                plt.ylabel("State History")
                plt.xlabel("Time")
        else:
            plt.subplot(1, 1, 1)
            for i in range(self.num_states):
                label = f"$x_{i}$"
                plt.plot(time_seq, ind_state_history[i], c=cmap(96 * i), label=label)
            plt.legend()
            plt.ylabel("State History")
            plt.xlabel("Time")

        if params.get("disp_plot", True) and "fLoc" not in kwargs:
            plt.show()
        elif "fLoc" in kwargs:
            plt.gcf().set_size_inches(10 * plt.gcf().get_size_inches())
            plt.tight_layout()
            plt.savefig(kwargs["fLoc"] + "target_ang_pos.png")
            plt.close()


class Target(Spacecraft):
    """Target spacecraft (passive object)."""

    _tids = count(0)

    def __init__(self, config: SimulationConfig, **kwargs):
        self.tid = next(self._tids) + 1
        super().__init__(config, **kwargs)

        self.observation_error_ang_pos = config.target.observation_error[
            "angular_position"
        ]
        self.observation_error_ang_vel = config.target.observation_error[
            "angular_velocity"
        ]

        # Default properties from config if not provided in kwargs
        self.angular_velocity = kwargs.get(
            "angularVelocity", config.target.initial_angular_velocity
        )
        self.mom_inertia = kwargs.get("momInertia", config.target.inertia)
        self.geometry = kwargs.get("geometry", {"Ineqs": [], "Eqs": []})

        if "dt" in kwargs:
            self.dt = kwargs["dt"]

        self.angular_velocity_history = [self.angular_velocity]

    def update_geometry(self, **kwargs):
        if "geometryIneqs" in kwargs:
            self.geometry["Ineqs"] = kwargs["geometryIneqs"]
        if "geometryEqs" in kwargs:
            self.geometry["Eqs"] = kwargs["geometryEqs"]
        return self.geometry

    def update_state(self, **kwargs):
        if "newAngularPos" in kwargs and "newAngularVel" in kwargs:
            new_angular_pos = kwargs["newAngularPos"]
            new_angular_vel = kwargs["newAngularVel"]

            # Update history
            self.angular_velocity_history.append(new_angular_vel)
            self.state_history.append(new_angular_pos)
            self.angular_velocity = new_angular_vel
            self.curr_state = new_angular_pos
        else:
            num_steps = kwargs.get("numSteps", 1)
            torque_val = kwargs.get(
                "torqueVal", [np.zeros(3) for _ in range(num_steps + 1)]
            )
            torque_type = kwargs.get("torqueType", "zero")

            # Integrate dynamics
            sol = intg.solve_ivp(
                fun=target_state_update_func,
                y0=np.concatenate((self.curr_state, self.angular_velocity)),
                t_span=[0, self.dt * num_steps],
                method="RK45",
                t_eval=np.arange(0, self.dt * num_steps, self.dt),
                args=(
                    self.dt,
                    self.mom_inertia,
                    gen_skew_sym_mat(self.angular_velocity),
                    torque_val,
                    torque_type,
                ),
            )

            if sol.success:
                new_state = sol.y.T  # (steps, 6)
                new_angular_pos = list(new_state[:, :3])
                new_angular_vel = list(new_state[:, 3:])

                self.angular_velocity_history.extend(new_angular_vel)
                self.state_history.extend(new_angular_pos)
            else:
                # Fallback
                self.angular_velocity_history.extend(
                    [self.angular_velocity] * num_steps
                )
                self.state_history.extend([self.curr_state] * num_steps)

            self.angular_velocity = self.angular_velocity_history[-1]
            self.curr_state = self.state_history[-1]

        return self.state_history

    def get_observed_state(self, t: float = 0.0):
        """Get state with observation noise."""
        idx = int(t / self.dt)
        if idx < len(self.state_history):
            state = self.state_history[idx]
        else:
            # Propagate if needed
            needed = idx - len(self.state_history) + 1
            self.update_state(numSteps=needed)
            state = self.state_history[idx]

        return state * random.gauss(1.00, self.observation_error_ang_pos)

    def get_observed_ang_vel(self, t: float = 0.0):
        """Get angular velocity with observation noise."""
        idx = int(t / self.dt)
        if idx < len(self.angular_velocity_history):
            ang_vel = self.angular_velocity_history[idx]
        else:
            needed = idx - len(self.angular_velocity_history) + 1
            self.update_state(numSteps=needed)
            ang_vel = self.angular_velocity_history[idx]

        return ang_vel * random.gauss(1.00, self.observation_error_ang_vel)

    def get_mom_inertia(self):
        return self.mom_inertia


class Chaser(Spacecraft):
    """Chaser agent performing control."""

    _cids = count(0)

    def __init__(self, config: SimulationConfig, repeat: bool = False, **kwargs):
        if not repeat:
            self.cid = next(self._cids) + 1
            super().__init__(config, **kwargs)

            self.mean_motion = config.orbit.mean_motion
            self.inputs = [np.zeros(3)]
            self.state_bounds = []
            self.input_bounds = []

            # Unpack bounds from config
            num_agents = int(len(self.init_state) / 6)
            for i in range(num_agents):
                self.state_bounds.extend(config.mpc.state_bounds)

            # Assuming 3 inputs per agent
            num_inputs = int(num_agents)
            for i in range(num_inputs):
                self.input_bounds.extend(config.mpc.input_bounds)

            self.goal_bounds = config.mpc.goal_bounds

            # Initialize Solver (default Gekko if not specified)
            self.solver = get_solver("gekko", config=None)

        self.neighbors = []
        self.t_n_info = {}
        self.g_n_info = {}
        self.r_n_info = {"len": 0}
        self.goal_state = None
        self.goal_locations = None
        self.occlusion = float("inf")

        self.target_observe_consensus = False
        self.neighbors_locn_consensus = False
        self.goal_calculate_consensus = False

    def reset_info(self):
        self.__init__(self.config, repeat=True)
        return self.goal_state

    def update_neighbors(self, operation: str, agent_list: List):
        if operation == "set":
            self.neighbors = agent_list.copy()
        elif operation == "append":
            for agent in agent_list:
                if agent not in self.neighbors:
                    self.neighbors.append(agent)
        elif operation == "remove":
            for agent in agent_list:
                if agent in self.neighbors:
                    self.neighbors.remove(agent)
        return self.neighbors

    def comm_data(self, type: str, info: Any, *args):
        """Handle communication data from neighbors."""
        if type == "target":
            neighbor = args[0]
            self.t_n_info[neighbor] = info
        elif type == "get_locations":
            self.r_n_info[self.id] = self.curr_state[:3]
            for agent in info.keys():
                if agent not in self.r_n_info.keys() and agent != "len":
                    self.r_n_info[agent] = info[agent]
                    self.neighbors_locn_consensus = False
            self.r_n_info["len"] = len(self.r_n_info.keys()) - 1
            if info["len"] == self.r_n_info["len"]:
                self.neighbors_locn_consensus = True
        elif type == "goal_list":
            if self.occlusion > info["occlusion"]:
                self.goal_locations = info["goalLocations"]
                self.occlusion = info["occlusion"]

            if self.occlusion > 400:
                self.goal_calculate_consensus = False
            else:
                self.goal_calculate_consensus = True
            return {"goalLocations": self.goal_locations, "occlusion": self.occlusion}
        elif type == "goal":
            pass
        else:
            raise ValueError("Invalid Type of Communication Data")

    def update_state(self, num_steps: int, *args):
        """Propagate chaser dynamics."""
        # Get CWH matrices
        matrices_tuple, _, _ = cwh_equations(
            self.anom_step,
            mean_motion=self.mean_motion,
            state_bounds=self.state_bounds,
            input_bounds=self.input_bounds,
        )
        A, B, state_cost_matrix, input_cost_matrix, d = matrices_tuple

        # Extend inputs if needed
        needed = num_steps - len(self.inputs)
        if needed > 0:
            self.inputs.extend([np.zeros_like(self.inputs[0]) for _ in range(needed)])

        for i in range(num_steps):
            if not self.inputs:
                break
            current_input = self.inputs[0]

            # x_next = A*x + B*u
            self.curr_state = np.dot(A, self.curr_state) + np.dot(B, current_input)

            self.inputs = self.inputs[1:]
            self.state_history.append(self.curr_state)

        return self.state_history

    def get_inputs(self):
        return self.inputs

    def set_inputs(self, inputs):
        self.inputs = inputs
        return self.inputs

    def get_observed_state(self):
        return self.curr_state * random.gauss(1.00, 0.001)

    def observe_target(self, target: Target):
        # Epsilon for observation consensus
        epsilon = 0.01

        t_state = target.get_observed_state()
        t_ang_vel = target.get_observed_ang_vel()

        self.t_state, self.t_ang_vel = t_state.copy(), t_ang_vel.copy()

        # Consensus average
        for info in self.t_n_info.values():
            self.t_state += info[0]  # state
            self.t_ang_vel += info[1]  # angVel

        count = len(self.t_n_info) + 1
        self.t_state /= count
        self.t_ang_vel /= count

        # Check consensus
        if (
            np.linalg.norm(t_state - self.t_state) < epsilon
            and np.linalg.norm(t_ang_vel - self.t_ang_vel) < epsilon
            and len(self.t_n_info) == len(self.neighbors)
        ):
            self.target_observe_consensus = True
        else:
            self.target_observe_consensus = False

        return (self.t_state, self.t_ang_vel)

    def calculate_goals(self):
        num_agents = len(self.r_n_info.keys()) - 1

        # Determine number of agents to initialize weights
        # Using defaults
        w_val = 1.0e3
        v_vals = [1.0e2, 1.0e6, 1.0e6]

        w = np.full(num_agents, w_val)
        v = v_vals

        try:
            x0 = self.goal_locations
            r_ids = list(self.r_n_info.keys())
        except:
            r_ids, x0 = [], []
            for agent in self.r_n_info.keys():
                if agent != "len":
                    r_ids.append(agent)
                    x0.append(self.r_n_info[agent])

        # Constraints
        constraints = []
        for agent in range(1, num_agents + 1):
            # Lower bound distance
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, ag=agent: (
                        np.linalg.norm(x[3 * ag - 3 : 3 * ag]) - self.goal_bounds[0]
                    ),
                }
            )
            # Upper bound distance
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, ag=agent: -(
                        np.linalg.norm(x[3 * ag - 3 : 3 * ag]) - self.goal_bounds[1]
                    ),
                }
            )

        constraints = tuple(constraints)

        # Initial guess optimization
        if x0 is None:
            x0 = [np.zeros(3) for _ in range(num_agents)]  # Fallback

        # Scale x0 to be within goal bounds
        avg_bound = sum(self.goal_bounds) / len(self.goal_bounds)
        x0_flat = []
        for x0i in x0:
            norm = np.linalg.norm(x0i)
            if norm > 1e-6:
                x0_flat.append(x0i * avg_bound / norm)
            else:
                x0_flat.append(np.array([avg_bound, 0, 0]))
        x0_flat = np.array(x0_flat).flatten()

        res = opt.minimize(
            calc_global_occlusion_cost,
            x0=x0_flat,
            args=(w, v, np.array(x0).flatten(), self.goal_bounds),
            options={"disp": False, "maxiter": 100},
            constraints=constraints,
        )

        self.goal_locations = res.x
        self.occlusion = calc_global_occlusion_cost(
            self.goal_locations, w, v, np.array(x0).flatten(), self.goal_bounds
        )
        return {"goalLocations": self.goal_locations, "occlusion": self.occlusion}

    def determine_goal_init(self, type: str, *args):
        if type == "pick_prelim_goal":
            self.goal_state = self.goal_locations[3 * self.id - 3 : 3 * self.id]
            return self.goal_state
        elif type == "create_agent":
            # Pass constructor for task allocation agent?
            task_alloc_class = args[0]
            self.task_alloc_agent = task_alloc_class(
                self.id - 1, self.curr_state[:3], self.goal_state, self.neighbors
            )
            return self.task_alloc_agent
        else:
            raise ValueError("Invalid Type of Goal Determination Command")

    def determine_inputs(self, num_steps: int):
        """Calculate optimal control inputs using the modular solver."""
        # Get dynamics matrices/functions from CWH
        matrices, _, _ = cwh_equations(
            self.anom_step,
            mean_motion=self.mean_motion,
            state_bounds=self.state_bounds,
            input_bounds=self.input_bounds,
            discretize_model=False,  # Continuous for GEKKO
        )
        A, B, Q, R, d = matrices

        # Build Problem
        problem = build_from_dynamics(
            A_func=A,
            B=B,
            Q=Q,
            R=R,
            d_func=d,
            x_0=self.curr_state,
            x_f=self.goal_state if self.goal_state is not None else np.zeros(6),
            start_anom=0,  # Relative time
            num_steps=num_steps,
            dt=self.dt,
            mean_motion=self.mean_motion,
            state_bounds=self.state_bounds,
            input_bounds=self.input_bounds,
        )

        # Solve
        result = self.solver.solve_problem(problem)

        if result.success and result.control_trajectory is not None:
            # Result inputs is shape (dimensions, steps)
            # We need list of inputs [u(0), u(1)...]
            # Transpose to (steps, dimensions)
            u_traj = result.control_trajectory.T
            self.inputs = [u_traj[i] for i in range(u_traj.shape[0])]
        else:
            logger.warning("Chaser solver failed. Keeping zero inputs.")
            self.inputs = [np.zeros(3) for _ in range(num_steps)]

        return self.inputs
