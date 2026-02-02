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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol, Sequence, Tuple

import logging
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment

from orbexa.utils import calc_distance

logger = logging.getLogger(__name__)
Vec = np.ndarray  # shape (d,)


# ============================================================
# Agent state (no neighbor graph stored here)
# ============================================================


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


# ============================================================
# Network model (this is the graph; "neighbor graph" is redundant)
# ============================================================


class Network(Protocol):
    def neighbors(self, agent_id: int) -> List[int]: ...

    def is_connected(self) -> bool: ...


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


# ============================================================
# Algorithms as policies (external "world" calls step())
# ============================================================


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


@dataclass
class LocalHungarianPolicy:
    """
    Local neighbor-cluster Hungarian reassignment heuristic.

    This mirrors your old "standard_auction_assignment" flavor, but as a policy.
    """

    def initialize(
        self, agents: List[AgentState], tasks: Sequence[Vec], network: Network
    ) -> None:
        return

    @staticmethod
    def _cost_matrix(starts: List[Vec], goals: List[Vec]) -> np.ndarray:
        C = np.zeros((len(starts), len(goals)))
        for i, s in enumerate(starts):
            for j, g in enumerate(goals):
                C[i, j] = -np.linalg.norm(np.asarray(s) - np.asarray(g))
        return C

    def step(
        self,
        agents: List[AgentState],
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
        self, agents: List[AgentState], tasks: Sequence[Vec], network: Network
    ) -> bool:
        # Heuristic policy; no strong convergence check.
        return False


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


# ============================================================
# Orchestrator (not "world": just glue for stepping/scheduling)
# ============================================================


@dataclass
class TaskAllocationSystem:
    """
    Holds agents + tasks + network + policy.
    External code calls step_all() or step_agent() as desired.
    """

    agents: List[AgentState]
    tasks: List[Vec]
    network: Network
    policy: Policy

    def __post_init__(self) -> None:
        self.policy.initialize(self.agents, self.tasks, self.network)

    def step_agent(self, agent_id: int) -> int:
        return self.policy.step(self.agents, self.tasks, self.network, agent_id)

    def step_all(self, order: Optional[Sequence[int]] = None) -> int:
        if order is None:
            order = list(range(len(self.agents)))
        total = 0
        for i in order:
            total += self.step_agent(int(i))
        return total

    def has_converged(self) -> bool:
        return self.policy.has_converged(self.agents, self.tasks, self.network)

    def current_goals(self) -> List[Vec]:
        return [a.goal_pos for a in self.agents]


# ============================================================
# Convenience: construction + plotting (tests only)
# ============================================================


def make_agents(
    start_positions: Sequence[Sequence[float]], initial_goals: Sequence[Sequence[float]]
) -> List[AgentState]:
    if len(start_positions) != len(initial_goals):
        raise ValueError("start_positions and initial_goals must have same length")

    agents: List[AgentState] = []
    for i, (s, g) in enumerate(zip(start_positions, initial_goals)):
        agents.append(
            AgentState(
                agent_id=i,
                start_pos=np.asarray(s, dtype=float),
                goal_pos=np.asarray(g, dtype=float),
            )
        )
    return agents


def plot_assignments(starts: Sequence[Vec], goals: Sequence[Vec], title: str) -> None:
    plt.figure()
    for s, g in zip(starts, goals):
        s = np.asarray(s)
        g = np.asarray(g)
        plt.plot([s[0], g[0]], [s[1], g[1]])
    plt.title(title)
    plt.axis("equal")
    plt.show()


def demo_main() -> None:
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
                f"[Greedy] converged in {it+1} iterations; interactions this iter={interactions}"
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
                f"[Auction] converged in {it+1} iterations; interactions this iter={interactions}"
            )
            break

    plot_assignments(
        [a.start_pos for a in auction_sys.agents],
        auction_sys.current_goals(),
        title="DistributedAuctionPolicy (kNN network)",
    )


if __name__ == "__main__":
    demo_main()
