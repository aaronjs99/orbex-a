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


"""system.py: Agent construction and allocation-policy scheduling.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from orbexa.mission.coordination.allocation.contracts import (
    AgentState,
    Network,
    Policy,
    Vec,
)


@dataclass
class TaskAllocationSystem:
    """
    Holds agents + tasks + network + policy.
    External code calls step_all() or step_agent() as desired.
    """

    agents: list[AgentState]
    tasks: list[Vec]
    network: Network
    policy: Policy

    def __post_init__(self) -> None:
        self.policy.initialize(self.agents, self.tasks, self.network)

    def step_agent(self, agent_id: int) -> int:
        return self.policy.step(self.agents, self.tasks, self.network, agent_id)

    def step_all(self, order: Sequence[int] | None = None) -> int:
        if order is None:
            order = list(range(len(self.agents)))
        total = 0
        for i in order:
            total += self.step_agent(int(i))
        return total

    def has_converged(self) -> bool:
        return self.policy.has_converged(self.agents, self.tasks, self.network)

    def current_goals(self) -> list[Vec]:
        return [a.goal_pos for a in self.agents]


def make_agents(
    start_positions: Sequence[Sequence[float]], initial_goals: Sequence[Sequence[float]]
) -> list[AgentState]:
    if len(start_positions) != len(initial_goals):
        raise ValueError("start_positions and initial_goals must have same length")

    agents: list[AgentState] = []
    for i, (s, g) in enumerate(zip(start_positions, initial_goals)):
        agents.append(
            AgentState(
                agent_id=i,
                start_pos=np.asarray(s, dtype=float),
                goal_pos=np.asarray(g, dtype=float),
            )
        )
    return agents
