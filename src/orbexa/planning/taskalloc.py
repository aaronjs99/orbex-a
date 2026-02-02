# /***********************************************************
# *                                                         *
# * Copyright (c) 2021                                      *
# *                                                         *
# * Dept. of Electrical Engineering                         *
# * Indian Institute of Technology Bombay                   *
# *                                                         *
# * Authors: Aaron John Sabu, Dwaipayan Mukherjee           *
# * Contact: aaronjs@ucla.edu, dm@ee.iitb.ac.in             *
# *                                                         *
# ***********************************************************/

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from orbexa.utils import calcDistance


# CLASS DEFINITIONS
class TaskAllocAgent:
    def __init__(self, id, r_0, r_T, G_w):
        self.id = id
        self.r_0 = r_0
        self.r_T = r_T
        self.G_w = G_w

    def getR_0(self):
        return self.r_0

    def getR_T(self):
        return self.r_T

    def getR_G(self, agents):
        R_0G = {}  # R_0G is a part of local info
        R_TG = {}  # R_TG is a part of local info
        for agent in agents:
            if agent.id in self.G_w:
                R_0G[agent.id] = agent.r_0.copy()
                R_TG[agent.id] = agent.r_T.copy()
        return R_0G, R_TG

    def commData(self, *args):
        raise NotImplementedError("commData() is not implemented")

    def checkIntersections(self, *args):
        raise NotImplementedError("checkIntersections() is not implemented")

    def greedySingleLoop(self, *args):
        raise NotImplementedError("greedySingleLoop() is not implemented")

    def disAuctionSingleLoop(self, agents):
        raise NotImplementedError("disAuctionSingleLoop() is not implemented")


class GreedyAgent(TaskAllocAgent):
    def __init__(self, id, r_0, r_T, G_w):
        super().__init__(id, r_0, r_T, G_w)

    def commData(self, agents, dir, agent_i, type, *args):
        if type == "new_goal":
            if dir == "send":
                agents[agent_i].r_T = args[0]
                agents[agent_i].checkIntersections(agents)
            elif dir == "recv":
                self.r_T = agents[agent_i].r_T
                self.checkIntersections(agents)
        return agents

    def checkIntersections(self, agents):
        self.N_int = []  ### list of neighbors that intersect
        R_0G, R_TG = self.getR_G(agents)
        for i in R_0G.keys():
            r_0i = R_0G[i]  ### initial position of neighboring agent i
            r_Ti = R_TG[i]  ### final   position of neighboring agent i
            straightDist = calcDistance(r_Ti, r_0i) + calcDistance(
                self.r_T, self.r_0
            )  ### sum of distances as calculated normally
            crossedDist = calcDistance(self.r_T, r_0i) + calcDistance(
                r_Ti, self.r_0
            )  ### sum of distances as calculated crossed
            if crossedDist < straightDist:
                self.N_int.append(i)
        self.bool_int = bool(self.N_int)
        return self.bool_int, self.N_int

    def greedySingleLoop(self, agents):
        totInteractions = 0
        self.checkIntersections(agents)
        if not self.bool_int:
            return agents, self, 0
        for i in self.N_int:
            ## --- BEGIN interaction between agent w and agent i --- ##
            r_Tn = self.r_T.copy()
            agents = self.commData(agents, "recv", i, "new_goal")
            agents = self.commData(agents, "send", i, "new_goal", r_Tn)
            totInteractions += 1
            ## --- END interaction between agent w and agent i --- ##
            if not self.bool_int:
                break
        return agents, self, totInteractions


class DisAucAgent(TaskAllocAgent):
    def __init__(self, id, r_0, r_T, G_w):
        super().__init__(id, r_0, r_T, G_w)

    def disAuctionSingleLoop(self, agents):
        self.a_prev = self.id
        return agents, self, 0


class CnsAucAgent(TaskAllocAgent):
    def __init__(self, id, r_0, r_T, G_w):
        super().__init__(id, r_0, r_T, G_w)

    def cnsAuctionSingleLoop(self, agents):
        return agents, self, 0


# FUNCTION DEFINITIONS
## Generate cost of moving from a particular location to a list of target locations
def genCost(egoLocation, targetLocations, minValue):
    return [
        (-(1 ** (minValue - 1)))
        * np.linalg.norm(np.subtract(egoLocation, targetLocations[i]))
        for i in range(len(targetLocations))
    ]


## Generate list of neighbors
def genNeighbors(type, R, k, idOffset=True):
    numAgents = len(R)
    N = [[] for i in range(numAgents)]
    try:
        if len(k) != numAgents:
            raise ValueError("Incorrect List Size for k")
    except TypeError:
        pass
    for i in range(numAgents):
        try:
            k_i = k[i]
        except:
            k_i = k
        R_i = list(R - R[i])
        D_i = np.array([np.linalg.norm(r_i) for r_i in R_i])
        if type == "kNN":  ## Generate list of neighbors based on k-Nearest Neighbors
            N[i] = np.setdiff1d(np.argpartition(D_i, k_i)[: k_i + 1], [i])
        elif (
            type == "maxDist"
        ):  ## Generate list of neighbors based on maximum communication distance maxDist
            N[i] = np.setdiff1d(np.nonzero(D_i <= k_i), [i])
        else:
            raise ValueError("Unknown Type of Neighbor Allocation")
        if idOffset:
            N[i] = np.add(N[i], 1)
    return N


## Check connectivity of graph
def isConnected(G):
    numAgents = len(G)
    metAgents = set()
    toVisit = set()
    visited = set()
    allAgents = range(len(G))
    metAgents.add(0)
    toVisit.add(0)
    while len(toVisit) != 0:
        w = list(toVisit)[0]
        if w not in visited:
            visited.add(w)
            metAgents.add(w)
            toVisit.remove(w)
            toVisit.update(
                [i for i in range(numAgents) if i in G[w] and i not in visited]
            )
    if len(metAgents) == numAgents:
        return True
    return False


## Standard Distributed Auction
def stdAuction(R_0, R_T, G):
    numAgents = len(R_0)
    for w in range(numAgents):
        R_Tn = [R_T[i] for i in range(numAgents) if G[w][i] > 0]
        R_0n = [R_0[i] for i in range(numAgents) if G[w][i] > 0]
        R_TnRange = [i for i in range(numAgents) if G[w][i] > 0]
        numNbrs = len(R_Tn)
        Auction_Values = [[] for i in range(numNbrs)]
        for nbr, r_0n in enumerate(R_0n):
            Auction_Values[nbr] = genCost(r_0n, R_Tn, True)
        row_ind, col_ind = linear_sum_assignment(Auction_Values)  # Hungarian Algorithm
        newR_Tn = []
        for i in range(numNbrs):
            newR_Tn.append(R_Tn[col_ind[i]])
        for newAgent, oldAgent in enumerate(R_TnRange):
            R_T[oldAgent] = newR_Tn[newAgent]
    return R_T


## Distributed Greedy Algorithm - 'Scalable Techniques for Autonomous Construction of a Paraboloidal Space Telescope in an Elliptic Orbit' - John Sabu, Mukherjee
def checkIntersections(w, R_0, R_T, G_w):
    ### w   - id of agent under consideration
    ### R_0 - initial positions of all agents
    ### R_T - final   positions of all agents
    ### G_w - neighborhood of agent w

    N_int = []  ### list of neighbors that intersect
    r_0w = R_0[w]  ### initial position of agent w
    r_Tw = R_T[w]  ### final   position of agent w
    for i, r_0i in enumerate(
        R_0
    ):  ### i = id of agent i, r_0i = Initial position of agent i
        if i in G_w:  ### if agent i is a neighbor of agent w
            r_Ti = R_T[i]  ### final position of agent i
            straightDist = calcDistance(r_Ti, r_0i) + calcDistance(
                r_Tw, r_0w
            )  ### sum of distances as calculated normally
            crossedDist = calcDistance(r_Tw, r_0i) + calcDistance(
                r_Ti, r_0w
            )  ### sum of distances as calculated crossed
            if crossedDist < straightDist:
                N_int.append(i)
    check = bool(N_int)
    return check, N_int


def greedy(R_0, R_T, G):
    numAgents = len(R_0)
    R_T = R_T.copy()
    totInteractions = 0
    while True:
        N_int = [[] for i in range(numAgents)]
        bool_int = [False for i in range(numAgents)]
        for w in range(numAgents):
            bool_int[w], N_int[w] = checkIntersections(w, R_0, R_T, G[w])
        if not bool(np.sum(bool_int)):
            break
        for w in range(numAgents):
            ## --- BEGIN operation on agent w --- ##
            if bool_int[w]:
                for i in N_int[w]:
                    ## --- BEGIN interaction between agent w and agent i --- ##
                    totInteractions += 1
                    R_T[w], R_T[i] = (
                        R_T[i],
                        R_T[w],
                    )  ### agent w and agent i switch goals
                    bool_int[w], N_int[w] = checkIntersections(
                        w, R_0, R_T, G[w]
                    )  ### agent w updates its list of intersecting neighbors
                    bool_int[i], N_int[i] = checkIntersections(
                        i, R_0, R_T, G[i]
                    )  ### agent i updates its list of intersecting neighbors
                    ## --- END   interaction between agent w and agent i --- ##
                    if not bool_int[w]:
                        break
            ## --- END   operation on agent w --- ##
    return R_T, totInteractions


### Distributed Auction Algorithm - 'A Distributed Auction Algorithm for the Assignment Problem' - Zavlanos, Spesivtsev, Pappas
def disAuction(R_0, R_T, G):
    if not isConnected(G):
        return R_T, -1

    totInteractions = 0
    G = G.copy()
    G = np.array(G, dtype=object)
    G = np.subtract(G, 1)
    numAgents = len(R_0)
    beta = [genCost(i, R_T, False) for i in R_0]
    p_prev = np.zeros((numAgents, numAgents))
    p_next = np.zeros((numAgents, numAgents))
    b_prev = np.zeros((numAgents, numAgents), dtype=int)
    b_next = np.zeros((numAgents, numAgents), dtype=int)
    a_prev = np.zeros(numAgents, dtype=int)
    a_next = np.zeros(numAgents, dtype=int)
    while len(a_prev) != len(set(a_prev)):
        for w in range(numAgents):
            for j in range(numAgents):
                p_next[w][j] = np.max(
                    [p_prev[k][j] for k in range(numAgents) if k in G[w]]
                )
                b_next[w][j] = int(
                    np.max(
                        [
                            b_prev[k][j]
                            for k in range(numAgents)
                            if k in G[w] and p_prev[k][j] == p_next[w][j]
                        ]
                    )
                )
                totInteractions += len(G[w])
            if (
                p_prev[w][a_prev[w]] <= p_next[w][a_prev[w]]
                and b_next[w][a_prev[w]] != w
            ):
                a_next[w] = np.argmax(
                    [(beta[w][k] - p_next[w][k]) for k in range(numAgents)]
                )
                b_next[w][a_next[w]] = w
                v_w = np.max([beta[w][k] - p_prev[w][k] for k in range(numAgents)])
                w_w = np.max(
                    [
                        beta[w][k] - p_prev[w][k]
                        for k in range(numAgents)
                        if k != a_next[w]
                    ]
                )
                epsilon = np.random.normal(
                    0.0, 0.05
                )  ####  epsilon-complementary slackness
                gamma_w = v_w - w_w + epsilon
                p_next[w][a_next[w]] = p_prev[w][a_next[w]] + gamma_w
            else:
                a_next[w] = a_prev[w]
        a_prev = a_next.copy()
        p_prev = p_next.copy()
        b_prev = b_next.copy()

    R_T_new = [R_T[a_w] for a_w in a_prev]
    return R_T_new, totInteractions


## 'Consensus-Based Decentralized Auctions for Robust Task Allocation' - Choi, Brunet, How
def selectTask(x_prev, y_prev, J_w, c_w, w_w):
    x_next = x_prev
    y_next = y_prev
    if np.sum(x_next) == 0:
        h_w = c_w >= y_next
        if any(h_w):
            J_w = np.argmax([h_w[k] * c_w[k] for k in range(len(y_next))])
            x_next[J_w] = 1
            y_next[J_w] = c_w[J_w]
    return x_next, y_next, J_w, w_w


def updateTask(w, x_prev, y, z_w, J_w, G_w):
    x_w = x_prev
    y_w = y[w]
    for j in range(len(y)):
        y_w[j] = np.max([y[k][j] for k in range(len(y)) if k in G_w])
    z_w[J_w] = np.argmax([y[k][J_w] for k in range(len(y)) if k in G_w])
    if not z_w[J_w] == w:
        x_w[J_w] = 0
    return x_w, y_w, z_w


def cnsAuction(R_0, R_T, G):
    if not isConnected(G):
        return R_T

    numAgents = len(R_0)
    c = [genCost(i, R_T, True) for i in R_0]
    x = np.zeros((numAgents, numAgents))
    y = np.zeros((numAgents, numAgents))
    t = np.zeros((numAgents, numAgents))
    z = np.zeros((numAgents, numAgents))
    w = np.zeros((numAgents, numAgents))
    J = [0 for i in range(numAgents)]
    while 0 in [(1 in x_i) for x_i in x]:
        for w in range(numAgents):
            x[w], y[w], J[w], w[w] = selectTask(x[w], y[w], J[w], c[w], w[w])
        for w in range(numAgents):
            x[w], y[w], z[w] = updateTask(w, x[w], y, z[w], J[w], G[w])
        print(z)
    return R_T


def hybAuction(R_0, R_T, G):
    if not isConnected(G):
        return greedy(R_0, R_T, G)
    else:
        return disAuction(R_0, R_T, G)


def functional_main():
    fig, axs = plt.subplots(2, 3)

    R_0 = [
        [4, 6],
        [7, 2],
        [-8, -5],
        [-8, 7],
        [4, -4],
        [2, -3],
        [-7, 1],
        [1, 1],
        [-3, -3],
        [-5, 5],
    ]
    R_T = [
        [-2, -1],
        [-5, -6],
        [7, -3],
        [-6, -3],
        [-9, 1],
        [2, 5],
        [-2, 7],
        [-5, 3],
        [4, 2],
        [6, -1],
    ]
    R_0 = [np.array(r_0) for r_0 in R_0]

    R_Tn = [R_T]
    totInteractions = [0]
    title = ["Original"]

    title.extend(
        [
            "Distributed Greedy  Algorithm with max. 4 neighbors",
            "Distributed Greedy  Algorithm with varying neighbors (large neighborhood)",
            "Distributed Greedy  Algorithm with varying neighbors (small neighborhood)",
        ]
    )
    title.extend(
        [
            "Distributed Auction Algorithm with max. 4 neighbors",
            "Distributed Auction Algorithm with varying neighbors (large neighborhood)",
            "Distributed Auction Algorithm with varying neighbors (small neighborhood)",
        ]
    )
    Gs = [
        genNeighbors("kNN", R_0, 4),
        genNeighbors("kNN", R_0, [4, 5, 5, 6, 4, 5, 3, 2, 4, 5]),
        genNeighbors("kNN", R_0, [2, 4, 4, 5, 3, 2, 2, 3, 3, 4]),
    ]

    for i in range(6):
        G = Gs[i % 3]
        if i < 3:
            out = greedy(R_0, R_T, G)
        else:
            out = disAuction(R_0, R_T, G)
        R_Tn.append(out[0])
        totInteractions.append(out[1])

    i, j = 0, 0
    for test in [0, 1, 3, 4, 5, 6]:
        dist = 0
        for pID, point in enumerate(R_0):
            axs[i, j].plot(
                [point[0], R_Tn[test][pID][0]], [point[1], R_Tn[test][pID][1]]
            )
            dist += np.linalg.norm(point - R_Tn[test][pID])
        axs[i, j].set_title(title[test], fontsize=7)
        print("R_Tn: ", R_Tn[test])
        print("dist: ", dist)
        print("totInteractions: ", totInteractions[test])
        print()
        j += 1
        if j == 3:
            i += 1
            j = 0
    plt.show()


def classed_main():
    fig, axs = plt.subplots(2, 3)

    R_0 = [
        [4, 6],
        [7, 2],
        [-8, -5],
        [-8, 7],
        [4, -4],
        [2, -3],
        [-7, 1],
        [1, 1],
        [-3, -3],
        [-5, 5],
    ]
    R_T = [
        [-2, -1],
        [-5, -6],
        [7, -3],
        [-6, -3],
        [-9, 1],
        [2, 5],
        [-2, 7],
        [-5, 3],
        [4, 2],
        [6, -1],
    ]
    R_0 = [np.array(r_0) for r_0 in R_0]
    numAgents = len(R_0)

    R_Tn = [R_T]
    totInteractions = [0]
    title = ["Original"]

    title.extend(
        [
            "Distributed Greedy  Algorithm with max. 4 neighbors",
            "Distributed Greedy  Algorithm with varying neighbors (large neighborhood)",
            "Distributed Greedy  Algorithm with varying neighbors (small neighborhood)",
        ]
    )
    title.extend(
        [
            "Distributed Auction Algorithm with max. 4 neighbors",
            "Distributed Auction Algorithm with varying neighbors (large neighborhood)",
            "Distributed Auction Algorithm with varying neighbors (small neighborhood)",
        ]
    )
    Gs = [
        genNeighbors("kNN", R_0, 4, idOffset=False),
        genNeighbors("kNN", R_0, [4, 5, 5, 6, 4, 5, 3, 2, 4, 5], idOffset=False),
        genNeighbors("kNN", R_0, [2, 4, 4, 5, 3, 2, 2, 3, 3, 4], idOffset=False),
    ]

    for i in range(6):
        G = Gs[i % 3]
        if i < 3:
            agents = [GreedyAgent(w, R_0[w], R_T[w], G[w]) for w in range(numAgents)]
        else:
            agents = [DisAucAgent(w, R_0[w], R_T[w], G[w]) for w in range(numAgents)]
        totInteractions.append(0)
        complete = False
        if i >= 3 and not isConnected(G):
            complete = True
            totInteractions[-1] = -1
        while not complete:
            for w in range(numAgents):
                if i < 3:
                    agents, agent_w, numInteractions = agents[w].greedySingleLoop(
                        agents
                    )
                else:
                    agents, agent_w, numInteractions = agents[w].disAuctionSingleLoop(
                        agents
                    )
                agents[w] = agent_w
                totInteractions[-1] += numInteractions
            if i < 3:
                complete = not (any([agent.bool_int for agent in agents]))
            else:
                a_prev = [agent.a_prev for agent in agents]
                complete = len(a_prev) == len(set(a_prev))
        R_Tn.append([agent.r_T for agent in agents])

    i, j = 0, 0
    for test in [0, 1, 3, 4, 5, 6]:
        dist = 0
        for pID, point in enumerate(R_0):
            axs[i, j].plot(
                [point[0], R_Tn[test][pID][0]], [point[1], R_Tn[test][pID][1]]
            )
            dist += np.linalg.norm(point - R_Tn[test][pID])
        axs[i, j].set_title(title[test], fontsize=7)
        print("R_Tn: ", R_Tn[test])
        print("dist: ", dist)
        print("totInteractions: ", totInteractions[test])
        print()
        j += 1
        if j == 3:
            i += 1
            j = 0
    plt.show()


if __name__ == "__main__":
    print("Functional  Implementation")
    functional_main()
    print("Class-based Implementation")
    classed_main()
