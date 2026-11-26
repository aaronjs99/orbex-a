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


"""auction.py: Incremental auction policy with caller-owned stepping order.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from orbexa.mission.coordination.allocation import logger
from orbexa.mission.coordination.allocation.contracts import AgentState, Network, Vec


@dataclass
class DistributedAuctionPolicy:
    """
    Incremental distributed auction (Zavlanos-style) as a policy.

    External caller drives scheduling: step(agent_id) in whatever order.
    """

    epsilon_std: float = 0.05

    def initialize(
        self, agents: list[AgentState], tasks: Sequence[Vec], network: Network
    ) -> None:
        """Set each agent's initial assignment, prices and winners.

        Each agent receives assignment_idx = agent_id % number of tasks, zero prices
        and winners of -1 for every task. Logs a warning if the network is not
        connected. Raises ValueError if tasks is empty.
        """
        n_tasks = len(tasks)
        if n_tasks == 0:
            raise ValueError("tasks must be non-empty")

        for a in agents:
            a.assignment_idx = a.agent_id % n_tasks
            a.prices = np.zeros(n_tasks, dtype=float)
            a.winners = -np.ones(n_tasks, dtype=int)

        if not network.is_connected():
            logger.warning(
                "Network is not connected; distributed auction may not converge."
            )

    @staticmethod
    def _utility(start: Vec, task: Vec) -> float:
        """Return the negative Euclidean distance between start and task."""
        return -float(np.linalg.norm(np.asarray(start) - np.asarray(task)))

    def step(
        self,
        agents: list[AgentState],
        tasks: Sequence[Vec],
        network: Network,
        agent_id: int,
    ) -> int:
        """Run one auction update for agent_id; return the neighbor-interaction count.

        For each task the agent adopts the highest price among itself and its
        neighbors, breaking price ties by the larger winner id. If the winner of its
        current task is another agent, it rebids on the task with the largest utility
        minus price: that price rises by the margin over the second-best net value plus
        Gaussian noise of standard deviation epsilon_std, and the agent is recorded as
        winner. Finally goal_pos is set to the task at assignment_idx.

        Outputs:
            Mutates the agent's prices, winners, assignment_idx and goal_pos.
            The count adds the number of neighbors once per task.
        """
        a = agents[agent_id]
        assert a.prices is not None and a.winners is not None

        nbrs = network.neighbors(agent_id)
        interactions = 0

        # "Consensus" update: take max price per task among neighbors (incl. self)
        for j in range(len(tasks)):
            best_price = a.prices[j]
            best_winner = a.winners[j]

            for k in nbrs:
                ak = agents[k]
                if ak.prices is None or ak.winners is None:
                    continue
                if ak.prices[j] > best_price:
                    best_price = ak.prices[j]
                    best_winner = ak.winners[j]
                elif ak.prices[j] == best_price:
                    best_winner = max(best_winner, ak.winners[j])

            a.prices[j] = best_price
            a.winners[j] = best_winner
            interactions += len(nbrs)

        # Decide whether to rebid
        cur = a.assignment_idx
        cur_winner = a.winners[cur]
        if cur_winner != agent_id:
            utils = np.array(
                [self._utility(a.start_pos, t) for t in tasks], dtype=float
            )
            net = utils - a.prices
            new_task = int(np.argmax(net))

            best = float(net[new_task])
            if len(tasks) > 1:
                second_best = float(
                    np.max([net[k] for k in range(len(tasks)) if k != new_task])
                )
            else:
                second_best = best

            gamma = (best - second_best) + float(
                np.random.normal(0.0, self.epsilon_std)
            )

            a.assignment_idx = new_task
            a.winners[new_task] = agent_id
            a.prices[new_task] = a.prices[new_task] + gamma

        # Materialize goal position from assignment index
        a.goal_pos = np.asarray(tasks[a.assignment_idx], dtype=float)
        return interactions

    def has_converged(
        self, agents: list[AgentState], tasks: Sequence[Vec], network: Network
    ) -> bool:
        """Return True when no two agents hold the same assignment_idx."""
        assigns = [a.assignment_idx for a in agents]
        return len(assigns) == len(set(assigns))
