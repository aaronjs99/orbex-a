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


"""Incremental auction policy with caller-owned stepping order."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Sequence
from orbexa.planning.coordination.allocation import logger
from orbexa.planning.coordination.allocation.contracts import AgentState, Network, Vec


@dataclass
class DistributedAuctionPolicy:
    """
    Incremental distributed auction (Zavlanos-style) as a policy.

    External caller drives scheduling: step(agent_id) in whatever order.
    """

    epsilon_std: float = 0.05

    def initialize(
        self, agents: List[AgentState], tasks: Sequence[Vec], network: Network
    ) -> None:
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
        return -float(np.linalg.norm(np.asarray(start) - np.asarray(task)))

    def step(
        self,
        agents: List[AgentState],
        tasks: Sequence[Vec],
        network: Network,
        agent_id: int,
    ) -> int:
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
        self, agents: List[AgentState], tasks: Sequence[Vec], network: Network
    ) -> bool:
        assigns = [a.assignment_idx for a in agents]
        return len(assigns) == len(set(assigns))
