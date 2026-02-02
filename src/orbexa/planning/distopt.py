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
from itertools import count
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize as opt

from orbexa.utils import calcLocalOcclusion, calcGlobalOcclusion
from orbexa.planning.taskalloc import genNeighbors

np.random.seed(0)


# CLASS DEFINITIONS
class DistOptAgent:
    _dids = count(0)

    def __init__(self, x, w, v, N, *args):
        self.did = next(self._dids)
        self.t = 1  ### Time Step
        self.x = x  ### Current State
        self.w = w  ### Weight Vector for Neighbors in Declustering
        self.v = v  ### Weight Vector for Neighbors in Bounding
        self.N = N  ### Neighbors
        self.allx = [x]  ### All States
        self.updateWV("only_neighbors")

    def updateX(self, X, fun=calcLocalOcclusion, *args):
        if len(args) == 0:
            args = [(self.w, self.v, X)]
        self.x = opt.minimize(
            fun,
            x0=self.allx[-1],
            args=args[0],
            options={"disp": False},
            constraints=(
                {
                    "type": "ineq",
                    "fun": lambda x: -(
                        np.linalg.norm(np.subtract(self.allx[-1], x)) - 1.0
                    ),
                }
            ),
        ).x
        self.allx.append(self.x)
        self.t += 1
        return self.x

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

    def updateN(self, operation, agentList):
        if operation == "set":
            self.G = agentList.copy()
        elif operation == "append":
            for agent in agentList:
                if agent not in self.G:
                    self.G.append(agent)
        elif operation == "remove":
            for agent in agentList:
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
    X = [
        np.array(
            [
                np.random.normal(3.0, 4.0),
                np.random.normal(1.0, 0.5),
                np.random.normal(2.0, 8.0),
            ]
        )
        for i in range(8)
    ]
    numAgents = len(X)
    numStates = len(X[0])
    W = [
        np.array([1.0 / numAgents for ego in range(numAgents)])
        for agent in range(numAgents)
    ]
    V = [5.0 for agent in range(numAgents)]
    N = genNeighbors("maxDist", X, 40, idOffset=False)
    agents = [DistOptAgent(X[i], W[i], V[i], N[i]) for i in range(numAgents)]

    print("Initial States: ", [agent.x for agent in agents])

    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    # Run Simulation
    for t in range(50):
        X_new = []
        for j in range(numAgents):
            x_j = agents[j].updateX(X)
            X_new.append(x_j)
        X = X_new.copy()

    # Plot Results
    for i in range(numAgents):
        ax1.scatter(agents[i].allx[0][0], agents[i].allx[0][1], label="Agent " + str(i))
        ax1.plot(
            [agents[i].allx[j][0] for j in range(len(agents[i].allx))],
            [agents[i].allx[j][1] for j in range(len(agents[i].allx))],
            label="Agent " + str(i),
        )
        ax2.plot(
            list(range(len(agents[i].allx))),
            [agents[i].allx[j][0] for j in range(len(agents[i].allx))],
            label="Agent " + str(i),
        )
        ax3.scatter(
            agents[i].allx[-1][0], agents[i].allx[-1][1], label="Agent " + str(i)
        )
        ax4.plot(
            list(range(len(agents[i].allx))),
            [agents[i].allx[j][1] for j in range(len(agents[i].allx))],
            label="Agent " + str(i),
        )
    plt.legend()
    plt.show()

    # Plot Results in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(numAgents):
        ax.scatter(
            agents[i].allx[0][0],
            agents[i].allx[0][1],
            agents[i].allx[0][2],
            label="Agent " + str(i),
        )
        ax.plot(
            [agents[i].allx[j][0] for j in range(len(agents[i].allx))],
            [agents[i].allx[j][1] for j in range(len(agents[i].allx))],
            [agents[i].allx[j][2] for j in range(len(agents[i].allx))],
            label="Agent " + str(i),
        )
    plt.legend()
    plt.show()

    print("Final   States: ", [agent.x for agent in agents])
