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


"""Local goal-swap policy with its original distance heuristic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence
from orbexa.planning.coordination.allocation.contracts import AgentState, Network, Vec
from orbexa.utils.geometry.algebra import calc_distance


@dataclass
class GreedySwapPolicy:
    """
    Greedy local swapping policy based on intersection heuristic.

    Note: This directly swaps neighbor goals (state mutation).
    For realism, replace this with message-based proposals.
    """

    def initialize(
        self, agents: List[AgentState], tasks: Sequence[Vec], network: Network
    ) -> None:
        return

    @staticmethod
    def _intersections_for(
        agent_w: AgentState, agents: List[AgentState], neighbors: Iterable[int]
    ) -> List[int]:
        w0 = agent_w.start_pos
        wT = agent_w.goal_pos

        intersecting: List[int] = []
        for i in neighbors:
            ai = agents[i]
            i0 = ai.start_pos
            iT = ai.goal_pos

            straight = calc_distance(iT, i0) + calc_distance(wT, w0)
            crossed = calc_distance(wT, i0) + calc_distance(iT, w0)
            if crossed < straight:
                intersecting.append(i)
        return intersecting

    def step(
        self,
        agents: List[AgentState],
        tasks: Sequence[Vec],
        network: Network,
        agent_id: int,
    ) -> int:
        agent = agents[agent_id]
        nbrs = network.neighbors(agent_id)

        agent.intersecting_neighbors = self._intersections_for(agent, agents, nbrs)
        interactions = 0

        for j in list(agent.intersecting_neighbors):
            tmp = agent.copy_goal()
            agent.goal_pos = agents[j].copy_goal()
            agents[j].goal_pos = tmp
            interactions += 1

            # Recompute and possibly early exit
            agent.intersecting_neighbors = self._intersections_for(agent, agents, nbrs)
            if not agent.intersecting_neighbors:
                break

        return interactions

    def has_converged(
        self, agents: List[AgentState], tasks: Sequence[Vec], network: Network
    ) -> bool:
        for w in range(len(agents)):
            if self._intersections_for(agents[w], agents, network.neighbors(w)):
                return False
        return True
