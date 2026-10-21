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


"""Typed agent memory and policy/network contracts."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence


Vec = np.ndarray  # shape (d,)


@dataclass
class AgentState:
    """Local agent state + per-algorithm memory."""

    agent_id: int
    start_pos: Vec
    goal_pos: Vec

    # Greedy memory
    intersecting_neighbors: List[int] = field(default_factory=list)

    # Distributed auction memory
    assignment_idx: int = 0
    prices: Optional[np.ndarray] = None  # shape (N_tasks,)
    winners: Optional[np.ndarray] = None  # shape (N_tasks,) int agent ids

    def copy_goal(self) -> Vec:
        return np.array(self.goal_pos, copy=True)


class Network(Protocol):
    def neighbors(self, agent_id: int) -> List[int]: ...

    def is_connected(self) -> bool: ...


class Policy(Protocol):
    def initialize(
        self, agents: List[AgentState], tasks: Sequence[Vec], network: Network
    ) -> None: ...

    def step(
        self,
        agents: List[AgentState],
        tasks: Sequence[Vec],
        network: Network,
        agent_id: int,
    ) -> int:
        """One local update for agent_id; returns an interaction count for metrics."""
        ...

    def has_converged(
        self, agents: List[AgentState], tasks: Sequence[Vec], network: Network
    ) -> bool: ...
