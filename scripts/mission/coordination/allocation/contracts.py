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


"""contracts.py: Typed agent memory and policy/network contracts.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

Vec = np.ndarray  # shape (d,)


@dataclass
class AgentState:
    """Local agent state + per-algorithm memory."""

    agent_id: int
    start_pos: Vec
    goal_pos: Vec

    # Greedy memory
    intersecting_neighbors: list[int] = field(default_factory=list)

    # Distributed auction memory
    assignment_idx: int = 0
    prices: np.ndarray | None = None  # shape (N_tasks,)
    winners: np.ndarray | None = None  # shape (N_tasks,) int agent ids

    def copy_goal(self) -> Vec:
        """Return an independent copy of goal_pos."""
        return np.array(self.goal_pos, copy=True)


class Network(Protocol):
    """Protocol for a communication topology over integer agent ids."""

    def neighbors(self, agent_id: int) -> list[int]:
        """Return the ids of the agents that agent_id can communicate with."""
        ...

    def is_connected(self) -> bool:
        """Return whether the communication graph is connected."""
        ...


class Policy(Protocol):
    """Protocol for a task-allocation policy that updates one agent per step."""

    def initialize(
        self, agents: list[AgentState], tasks: Sequence[Vec], network: Network
    ) -> None:
        """Prepare per-agent policy memory before any step is taken."""
        ...

    def step(
        self,
        agents: list[AgentState],
        tasks: Sequence[Vec],
        network: Network,
        agent_id: int,
    ) -> int:
        """One local update for agent_id; returns an interaction count for metrics."""
        ...

    def has_converged(
        self, agents: list[AgentState], tasks: Sequence[Vec], network: Network
    ) -> bool:
        """Return whether the policy's convergence criterion currently holds."""
        ...
