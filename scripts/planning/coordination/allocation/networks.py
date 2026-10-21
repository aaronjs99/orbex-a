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


"""Explicit static, nearest-neighbor and radius communication networks."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Sequence, Tuple
from orbexa.planning.coordination.allocation.contracts import Vec


@dataclass
class StaticNeighborNetwork:
    """
    Static adjacency list network.

    neighbors_by_agent[i] = list of agent ids agent i can communicate with.
    """

    neighbors_by_agent: List[List[int]]

    def neighbors(self, agent_id: int) -> List[int]:
        return list(self.neighbors_by_agent[agent_id])

    def is_connected(self) -> bool:
        n = len(self.neighbors_by_agent)
        if n == 0:
            return True

        visited = set()
        stack = [0]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for v in self.neighbors_by_agent[u]:
                if v not in visited:
                    stack.append(v)
        return len(visited) == n


@dataclass
class KNNNetwork:
    """kNN network computed from provided positions."""

    positions: Sequence[Vec]
    k: int
    include_self: bool = False

    def neighbors(self, agent_id: int) -> List[int]:
        x = np.asarray(self.positions[agent_id], dtype=float)
        dists: List[Tuple[int, float]] = []
        for j, p in enumerate(self.positions):
            if not self.include_self and j == agent_id:
                continue
            pj = np.asarray(p, dtype=float)
            dists.append((j, float(np.linalg.norm(pj - x))))
        dists.sort(key=lambda t: t[1])
        return [j for j, _ in dists[: self.k]]

    def is_connected(self) -> bool:
        adj = [self.neighbors(i) for i in range(len(self.positions))]
        return StaticNeighborNetwork(adj).is_connected()


@dataclass
class RadiusNetwork:
    """Radius-based network computed from provided positions."""

    positions: Sequence[Vec]
    radius: float
    include_self: bool = False

    def neighbors(self, agent_id: int) -> List[int]:
        x = np.asarray(self.positions[agent_id], dtype=float)
        out: List[int] = []
        for j, p in enumerate(self.positions):
            if not self.include_self and j == agent_id:
                continue
            pj = np.asarray(p, dtype=float)
            if np.linalg.norm(pj - x) <= self.radius:
                out.append(j)
        return out

    def is_connected(self) -> bool:
        adj = [self.neighbors(i) for i in range(len(self.positions))]
        return StaticNeighborNetwork(adj).is_connected()
