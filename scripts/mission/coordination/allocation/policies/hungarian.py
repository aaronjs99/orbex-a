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


"""hungarian.py: Local Hungarian reassignment with Euclidean travel costs.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from orbexa.mission.coordination.allocation.contracts import AgentState, Network, Vec
from scipy.optimize import linear_sum_assignment


@dataclass
class LocalHungarianPolicy:
    """
    Minimize total Euclidean travel within the current neighbor cluster.
    """

    def initialize(
        self, agents: list[AgentState], tasks: Sequence[Vec], network: Network
    ) -> None:
        return

    @staticmethod
    def _cost_matrix(starts: list[Vec], goals: list[Vec]) -> np.ndarray:
        C = np.zeros((len(starts), len(goals)))
        for i, s in enumerate(starts):
            for j, g in enumerate(goals):
                C[i, j] = np.linalg.norm(np.asarray(s) - np.asarray(g))
        return C

    def step(
        self,
        agents: list[AgentState],
        tasks: Sequence[Vec],
        network: Network,
        agent_id: int,
    ) -> int:
        w = agent_id
        nbrs = network.neighbors(w)
        if not nbrs:
            return 0

        cluster = sorted(set([w] + list(nbrs)))
        starts = [agents[i].start_pos for i in cluster]
        goals = [agents[i].goal_pos for i in cluster]

        C = self._cost_matrix(starts, goals)
        row_ind, col_ind = linear_sum_assignment(C)
        new_goals = [goals[j] for j in col_ind]

        for idx, agent_idx in enumerate(cluster):
            agents[agent_idx].goal_pos = np.asarray(new_goals[idx], dtype=float)

        return len(nbrs)

    def has_converged(
        self, agents: list[AgentState], tasks: Sequence[Vec], network: Network
    ) -> bool:
        # Heuristic policy; no strong convergence check.
        return False
