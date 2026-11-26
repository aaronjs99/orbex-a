# /***********************************************************
# *                                                         *
# * Copyright (c) 2021                                      *
# *                                                         *
# * Dept. of Electrical Engineering                         *
# * Indian Institute of Technology Bombay                   *
# *                                                         *
# * Authors: Aaron John Sabu                                *
# * Contact: aaronjs@ucla.edu                               *
# *                                                         *
# ***********************************************************/


"""demo.py: Optional allocation figures and explicit demonstration.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from orbexa.mission.coordination.allocation.contracts import Vec
from orbexa.mission.coordination.allocation.networks import KNNNetwork
from orbexa.mission.coordination.allocation.policies.auction import (
    DistributedAuctionPolicy,
)
from orbexa.mission.coordination.allocation.policies.greedy import GreedySwapPolicy
from orbexa.mission.coordination.allocation.system import (
    TaskAllocationSystem,
    make_agents,
)


def plot_assignments(starts: Sequence[Vec], goals: Sequence[Vec], title: str) -> None:
    """Plot a straight segment from each start to its goal and show the figure.

    Inputs:
        starts, goals: paired positions; the first two components are plotted.
        title: figure title.
    """
    plt.figure()
    for s, g in zip(starts, goals):
        s = np.asarray(s)
        g = np.asarray(g)
        plt.plot([s[0], g[0]], [s[1], g[1]])
    plt.title(title)
    plt.axis("equal")
    plt.show()


def demo_main() -> None:
    """Run the greedy and auction policies on a fixed 10-agent example and plot both.

    Each policy runs on a 4-nearest-neighbor network over the start positions, for up
    to 100 (greedy) or 400 (auction) iterations, stopping at the first converged
    iteration. Prints the iteration count on convergence and opens one plot window per
    policy.
    """
    start_positions = [
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
    target_positions = [
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

    agents = make_agents(start_positions, target_positions)
    tasks = [np.asarray(p, dtype=float) for p in target_positions]

    # Network examples
    knn_net = KNNNetwork([a.start_pos for a in agents], k=4)
    # Or fully connected for sanity:
    # n = len(agents)
    # full_net = StaticNeighborNetwork([[j for j in range(n) if j != i] for i in range(n)])

    # 1) Greedy policy
    greedy_sys = TaskAllocationSystem(
        agents=make_agents(start_positions, target_positions),
        tasks=tasks,
        network=knn_net,
        policy=GreedySwapPolicy(),
    )

    for it in range(100):
        interactions = greedy_sys.step_all()
        if greedy_sys.has_converged():
            print(
                f"[Greedy] converged in {it + 1} iterations; interactions this iter={interactions}"
            )
            break

    plot_assignments(
        [a.start_pos for a in greedy_sys.agents],
        greedy_sys.current_goals(),
        title="GreedySwapPolicy (kNN network)",
    )

    # 2) Distributed auction policy (may converge slowly depending on network)
    auction_sys = TaskAllocationSystem(
        agents=make_agents(start_positions, target_positions),
        tasks=tasks,
        network=knn_net,
        policy=DistributedAuctionPolicy(epsilon_std=0.01),
    )

    for it in range(400):
        interactions = auction_sys.step_all()
        if auction_sys.has_converged():
            print(
                f"[Auction] converged in {it + 1} iterations; interactions this iter={interactions}"
            )
            break

    plot_assignments(
        [a.start_pos for a in auction_sys.agents],
        auction_sys.current_goals(),
        title="DistributedAuctionPolicy (kNN network)",
    )
