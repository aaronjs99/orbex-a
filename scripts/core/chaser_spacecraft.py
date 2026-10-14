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


"""Standalone chaser state, communication and explicitly selected solver services."""
from __future__ import annotations

from dataclasses import asdict
from itertools import count
import logging
from typing import TYPE_CHECKING, Any, List

import numpy as np
from orbexa.core.spacecraft import Spacecraft

if TYPE_CHECKING:
    from orbexa.core.config import SimulationConfig
    from orbexa.core.target_spacecraft import Target

logger = logging.getLogger(__name__)


class Chaser(Spacecraft):
    """Standalone CWH agent using physical seconds for every control interval.

    States are radial/transverse/normal positions in meters followed by velocities
    in m/s; CWH control inputs are accelerations in m/s^2. ``dt`` sets the seconds
    sample period for planning, propagation and plots. The legacy default keeps
    the numeric config.anom_step value as a seconds sample period. Callers can
    select a physical period explicitly; no time/anomaly scaling is applied.
    """

    _cids = count(0)

    def __init__(self, config: "SimulationConfig", repeat: bool = False, **kwargs):
        if not repeat:
            self.cid = next(self._cids) + 1
            super().__init__(config, **kwargs)
            if self.num_states != 6:
                raise ValueError("A standalone Chaser requires one six-state translational vector")

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

            # The solver remains injectable and inherits explicit local/remote policy.
            self.solver = kwargs.get("solver")
            if self.solver is None:
                from orbexa.solvers import get_solver

                self.solver = get_solver("gekko", config=asdict(config.solver))
            if not callable(getattr(self.solver, "solve_problem", None)):
                raise TypeError("Chaser solver must provide solve_problem(problem)")

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
        """Handle neighbor data or set this chaser's explicit local goal.

        ``goal`` accepts a finite three-value position or a complete six-state
        reference, stores a copy, and returns a copy. Position-only goals imply
        zero velocity when ``determine_inputs`` constructs its reference state.
        Other commands retain the existing neighbor-consensus protocol.
        """
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
            if args:
                raise ValueError("The goal command accepts only a local position or complete state")
            goal = np.asarray(info, dtype=float)
            if goal.shape not in ((3,), (self.num_states,)) or not np.all(np.isfinite(goal)):
                raise ValueError("Chaser goal must contain three finite position values or a finite complete state")
            self.goal_state = goal.copy()
            return self.goal_state.copy()
        else:
            raise ValueError("Invalid Type of Communication Data")

    def update_state(self, num_steps: int, *args):
        """Propagate exactly num_steps planned control intervals."""
        from orbexa.core.dynamics import cwh_equations

        if (isinstance(num_steps, bool) or not isinstance(num_steps, (int, np.integer))
                or num_steps < 0 or num_steps > len(self.inputs)):
            raise ValueError("Requested propagation exceeds the available control intervals")
        # Get CWH matrices
        matrices_tuple, _, _ = cwh_equations(
            self.dt,
            mean_motion=self.mean_motion,
            state_bounds=self.state_bounds,
            input_bounds=self.input_bounds,
        )
        A, B, state_cost_matrix, input_cost_matrix, d = matrices_tuple

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
        return self.curr_state * self.rng.gauss(1.00, 0.001)

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
        from scipy import optimize as opt
        from orbexa.utils.math_utils import calc_global_occlusion_cost

        if self.goal_bounds is None:
            raise ValueError("Standalone goal allocation requires explicit mpc.goal_bounds; docking missions use target geometry")
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
        """Plan num_steps control intervals using num_steps + 1 state samples."""
        from orbexa.core.dynamics import cwh_equations
        from orbexa.control.mpc_problem_builder import build_from_dynamics

        if isinstance(num_steps, bool) or not isinstance(num_steps, (int, np.integer)) or num_steps < 1:
            raise ValueError("At least one control interval is required")
        reference_state = np.zeros_like(self.curr_state, dtype=float)
        if self.goal_state is not None:
            goal = np.asarray(self.goal_state, dtype=float).reshape(-1)
            if len(goal) not in (3, len(reference_state)):
                raise ValueError("Chaser goal must contain a position or a complete state")
            reference_state[: len(goal)] = goal
        # Get dynamics matrices/functions from CWH
        matrices, _, _ = cwh_equations(
            self.dt,
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
            x_f=reference_state,
            start_anom=0,  # Relative time
            num_steps=num_steps + 1,
            dt=self.dt,
            mean_motion=self.mean_motion,
            use_anomaly_scaling=False,
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
