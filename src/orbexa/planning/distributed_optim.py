# /***********************************************************
# *                                                         *
# * Copyright (c) 2026                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu, Brett T. Lopez                *
# * Contact: {aaronjs, btlopez}@ucla.edu                    *
# *                                                         *
# ***********************************************************/

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as opt

from orbexa.utils import calc_global_occlusion_cost, calc_local_occlusion_cost

logger = logging.getLogger(__name__)


# -----------------------------
# Neighbor policies + network
# -----------------------------
class NeighborPolicy(Protocol):
    def neighbors(self, states: Sequence[np.ndarray]) -> List[List[int]]: ...


@dataclass(frozen=True)
class MaxDistanceNeighbors:
    """Adjacency list where i connects to all j with ||x_i - x_j|| <= max_distance."""

    max_distance: float

    def neighbors(self, states: Sequence[np.ndarray]) -> List[List[int]]:
        x = np.asarray(states, dtype=float)
        num_agents = x.shape[0]
        out: List[List[int]] = [[] for _ in range(num_agents)]
        for i in range(num_agents):
            deltas = x - x[i]
            dists = np.linalg.norm(deltas, axis=1)
            nbrs = np.where(dists <= self.max_distance)[0].tolist()
            out[i] = [j for j in nbrs if j != i]
        return out


@dataclass
class AgentStateNetwork:
    """Owns agent state vectors and a communication graph (adjacency list)."""

    states: List[np.ndarray]
    neighbor_policy: NeighborPolicy
    neighbors: List[List[int]] = field(init=False)

    def __post_init__(self) -> None:
        self.states = [np.asarray(s, dtype=float).copy() for s in self.states]
        self.neighbors = self.neighbor_policy.neighbors(self.states)

    @property
    def num_agents(self) -> int:
        return len(self.states)

    def recompute_neighbors(self) -> None:
        self.neighbors = self.neighbor_policy.neighbors(self.states)

    def set_state(self, agent_id: int, state: np.ndarray) -> None:
        self.states[agent_id] = np.asarray(state, dtype=float).copy()

    def snapshot_states(self) -> List[np.ndarray]:
        return [s.copy() for s in self.states]

    def evaluate_global_cost(
        self,
        global_cost: Callable[..., float],
        *,
        states_snapshot: Optional[Sequence[np.ndarray]] = None,
    ) -> float:
        """
        Optional helper for logging or analysis.

        We avoid assuming a specific global_cost signature, so we pass only the snapshot.
        If your global cost expects more args, wrap it before passing in.
        """
        if states_snapshot is None:
            states_snapshot = self.snapshot_states()
        return float(global_cost(states_snapshot))


# -----------------------------
# Agent + algorithm
# -----------------------------
@dataclass
class DistributedOptimizationAgent:
    agent_id: int
    network: AgentStateNetwork
    neighbor_weight: np.ndarray
    bound_weight: float
    step_limit: float = 1.0
    renormalize_neighbor_weights: bool = False

    state_history: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = self.network.num_agents
        w = np.asarray(self.neighbor_weight, dtype=float).reshape(-1)
        if w.shape[0] != n:
            raise ValueError(f"neighbor_weight must have length {n}, got {w.shape[0]}.")
        self.neighbor_weight = w

        self.state_history.append(self.state.copy())
        self._mask_weights_to_neighbors()

    @property
    def state(self) -> np.ndarray:
        return self.network.states[self.agent_id]

    def _mask_weights_to_neighbors(self) -> None:
        nbrs = set(self.network.neighbors[self.agent_id])

        # Zero out non-neighbors (and leave self weight untouched, if you want it)
        for j in range(self.network.num_agents):
            if j != self.agent_id and j not in nbrs:
                self.neighbor_weight[j] = 0.0

        if self.renormalize_neighbor_weights:
            s = float(np.sum(self.neighbor_weight))
            if s > 0.0:
                self.neighbor_weight /= s

    def _movement_constraint(self, x_candidate: np.ndarray) -> float:
        # constraint: step_limit - ||x_candidate - x_prev|| >= 0
        x_prev = self.state_history[-1]
        return self.step_limit - float(np.linalg.norm(x_candidate - x_prev))

    def step(
        self,
        local_cost: Callable[..., float],
        *,
        states_snapshot: Optional[Sequence[np.ndarray]] = None,
        recompute_neighbors: bool = False,
    ) -> np.ndarray:
        """
        One local optimization step.

        Expected local_cost signature:
            local_cost(x, w, v, X)

        Args:
            local_cost: objective function
            states_snapshot: optional frozen copy of global states for this iteration
            recompute_neighbors: if True, update comm graph before masking weights
        """
        if recompute_neighbors:
            self.network.recompute_neighbors()

        self._mask_weights_to_neighbors()

        if states_snapshot is None:
            states_snapshot = self.network.snapshot_states()

        result = opt.minimize(
            local_cost,
            x0=self.state_history[-1],
            args=(self.neighbor_weight, self.bound_weight, states_snapshot),
            options={"disp": False},
            constraints=({"type": "ineq", "fun": self._movement_constraint},),
        )
        x_new = np.asarray(result.x, dtype=float).copy()

        self.network.set_state(self.agent_id, x_new)
        self.state_history.append(x_new.copy())
        return x_new


# -----------------------------
# Demo / visualization
# -----------------------------
def run_demo(
    *,
    num_agents: int = 8,
    num_steps: int = 50,
    seed: int = 0,
    max_distance: float = 40.0,
    bound_weight: float = 5.0,
    step_limit: float = 1.0,
    recompute_neighbors_each_step: bool = True,
    local_cost: Callable[..., float] = calc_local_occlusion_cost,
    global_cost: Optional[Callable[..., float]] = None,
    renormalize_neighbor_weights: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(seed)
    initial_states = [
        np.array(
            [
                rng.normal(3.0, 4.0),
                rng.normal(1.0, 0.5),
                rng.normal(2.0, 8.0),
            ],
            dtype=float,
        )
        for _ in range(num_agents)
    ]

    network = AgentStateNetwork(
        states=initial_states,
        neighbor_policy=MaxDistanceNeighbors(max_distance=max_distance),
    )

    weights = [
        np.ones(num_agents, dtype=float) / float(num_agents) for _ in range(num_agents)
    ]
    agents = [
        DistributedOptimizationAgent(
            agent_id=i,
            network=network,
            neighbor_weight=weights[i],
            bound_weight=bound_weight,
            step_limit=step_limit,
            renormalize_neighbor_weights=renormalize_neighbor_weights,
        )
        for i in range(num_agents)
    ]

    logger.info("Initial states: %s", [a.state for a in agents])

    for t in range(num_steps):
        if recompute_neighbors_each_step:
            network.recompute_neighbors()

        snapshot = network.snapshot_states()

        neighbor_reads = 0
        for agent in agents:
            neighbor_reads += len(network.neighbors[agent.agent_id])
            agent.step(local_cost, states_snapshot=snapshot)

        if (t + 1) % 10 == 0:
            msg = f"[DistOpt] step={t+1}/{num_steps}; neighbor-reads={neighbor_reads}"
            if global_cost is not None:
                try:
                    # If calc_global_occlusion_cost wants a different signature,
                    # pass a wrapper in run_demo instead.
                    msg += f"; global_cost={network.evaluate_global_cost(global_cost, states_snapshot=snapshot):.3f}"
                except TypeError:
                    msg += "; global_cost=<signature-mismatch>"
            logger.info(msg)

    logger.info("Final states: %s", [a.state for a in agents])

    # 2D plots
    fig, axs = plt.subplots(2, 2)
    ax_xy0, ax_x_t, ax_xyf, ax_y_t = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    for i, agent in enumerate(agents):
        hist = np.asarray(agent.state_history)
        ax_xy0.scatter(hist[0, 0], hist[0, 1], label=f"Agent {i}")
        ax_xy0.plot(hist[:, 0], hist[:, 1])
        ax_xyf.scatter(hist[-1, 0], hist[-1, 1], label=f"Agent {i}")

        ax_x_t.plot(np.arange(hist.shape[0]), hist[:, 0], label=f"Agent {i}")
        ax_y_t.plot(np.arange(hist.shape[0]), hist[:, 1], label=f"Agent {i}")

    ax_xy0.set_title("Trajectories (x-y)")
    ax_xyf.set_title("Final (x-y)")
    ax_x_t.set_title("x over time")
    ax_y_t.set_title("y over time")
    ax_xy0.legend()

    plt.tight_layout()
    plt.show()

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i, agent in enumerate(agents):
        hist = np.asarray(agent.state_history)
        ax.scatter(hist[0, 0], hist[0, 1], hist[0, 2], label=f"Agent {i}")
        ax.plot(hist[:, 0], hist[:, 1], hist[:, 2])
    ax.set_title("3D trajectories")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # Keep default behavior identical to your successful run.
    run_demo(global_cost=None)

    # If you later want to log global cost and its signature matches,
    # you can switch to:
    # run_demo(global_cost=calc_global_occlusion_cost)
