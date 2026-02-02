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
import numpy as np
import matplotlib.pyplot as plt
import logging
from itertools import count
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize as opt

from orbexa.utils import calc_local_occlusion_cost, calc_global_occlusion_cost
from orbexa.planning.taskalloc import gen_neighbors

logger = logging.getLogger(__name__)

np.random.seed(0)


# CLASS DEFINITIONS
class DistOptAgent:
    _dids = count(0)

    def __init__(self, state, w, v, N, *args):
        self.did = next(self._dids)
        self.t = 1  ### Time Step
        self.state = state  ### Current State
        self.w = w  ### Weight Vector for Neighbors in Declustering
        self.v = v  ### Weight Vector for Neighbors in Bounding
        self.N = N  ### Neighbors
        self.state_history = [state]  ### All States
        self.updateWV("only_neighbors")

    def update_state(self, states, fun=calc_local_occlusion_cost, *args):
        if len(args) == 0:
            args = [(self.w, self.v, states)]
        self.state = opt.minimize(
            fun,
            x0=self.state_history[-1],
            args=args[0],
            options={"disp": False},
            constraints=(
                {
                    "type": "ineq",
                    "fun": lambda state_candidate: -(
                        np.linalg.norm(
                            np.subtract(self.state_history[-1], state_candidate)
                        )
                        - 1.0
                    ),
                }
            ),
        ).x
        self.state_history.append(self.state)
        self.t += 1
        return self.state

    def updateWV(self, operation, *args):
        if operation == "setW":
            self.w = args[0]
        elif operation == "setV":
            self.v = args[0]
        elif operation == "setWV":
            self.w = args[0]
            self.v = args[1]
        elif operation == "only_neighbors":
            for i in range(len(self.w)):
                if i not in self.N and i != self.did:
                    self.w[i] = 0

    def update_n(self, operation, agent_list):
        if operation == "set":
            self.G = agent_list.copy()
        elif operation == "append":
            for agent in agent_list:
                if agent not in self.G:
                    self.G.append(agent)
        elif operation == "remove":
            for agent in agent_list:
                if agent in self.G:
                    self.G.remove(agent)
        return self.G


# MAIN FUNCTION
if __name__ == "__main__":
    # Initialize Agents
    # X = [np.array([ 5.0,  1.5]),
    #      np.array([-6.0,  2.0]),
    #      np.array([ 1.5, -8.0]),
    #      np.array([ 3.0,  8.5]),
    #      np.array([-2.5,  7.0]),
    #      np.array([ 3.0, -4.0]),
    #      np.array([ 7.0,  4.5]),
    #      np.array([ 2.5, -8.5])]
    states = [
        np.array(
            [
                np.random.normal(3.0, 4.0),
                np.random.normal(1.0, 0.5),
                np.random.normal(2.0, 8.0),
            ]
        )
        for i in range(8)
    ]
    num_agents = len(states)
    numStates = len(states[0])
    W = [
        np.array([1.0 / num_agents for ego in range(num_agents)])
        for agent in range(num_agents)
    ]
    V = [5.0 for agent in range(num_agents)]
    N = gen_neighbors("maxDist", states, 40, idOffset=False)
    agents = [DistOptAgent(states[i], W[i], V[i], N[i]) for i in range(num_agents)]

    logger.info(f"Initial States: {[agent.state for agent in agents]}")

    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    # Run Simulation
    for t in range(50):
        states_new = []
        for j in range(num_agents):
            state_j = agents[j].update_state(states)
            states_new.append(state_j)
        states = states_new.copy()

    # Plot Results
    for i in range(num_agents):
        ax1.scatter(
            agents[i].state_history[0][0],
            agents[i].state_history[0][1],
            label="Agent " + str(i),
        )
        ax1.plot(
            [
                agents[i].state_history[j][0]
                for j in range(len(agents[i].state_history))
            ],
            [
                agents[i].state_history[j][1]
                for j in range(len(agents[i].state_history))
            ],
            label="Agent " + str(i),
        )
        ax2.plot(
            list(range(len(agents[i].state_history))),
            [
                agents[i].state_history[j][0]
                for j in range(len(agents[i].state_history))
            ],
            label="Agent " + str(i),
        )
        ax3.scatter(
            agents[i].state_history[-1][0],
            agents[i].state_history[-1][1],
            label="Agent " + str(i),
        )
        ax4.plot(
            list(range(len(agents[i].state_history))),
            [
                agents[i].state_history[j][1]
                for j in range(len(agents[i].state_history))
            ],
            label="Agent " + str(i),
        )
    plt.legend()
    plt.show()

    # Plot Results in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(num_agents):
        ax.scatter(
            agents[i].state_history[0][0],
            agents[i].state_history[0][1],
            agents[i].state_history[0][2],
            label="Agent " + str(i),
        )
        ax.plot(
            [
                agents[i].state_history[j][0]
                for j in range(len(agents[i].state_history))
            ],
            [
                agents[i].state_history[j][1]
                for j in range(len(agents[i].state_history))
            ],
            [
                agents[i].state_history[j][2]
                for j in range(len(agents[i].state_history))
            ],
            label="Agent " + str(i),
        )
    plt.legend()
    plt.show()

    logger.info(f"Final States: {[agent.state for agent in agents]}")
