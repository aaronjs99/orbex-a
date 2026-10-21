# /***********************************************************
# *                                                         *
# * Copyright (c) 2026                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu                                *
# * Contact: aaronjs@ucla.edu                               *
# *                                                         *
# ***********************************************************/

"""Numerical local and fleet visibility objectives."""

from typing import Optional
import numpy as np


def calc_local_occlusion_cost(
    state: np.ndarray,
    w: np.ndarray,
    v: float,
    states_list: np.ndarray,
    limits: Optional[dict] = None,
) -> float:
    """
    Estimate occlusion cost (to be minimized).
    Higher cost = Closer to neighbors or violating bounds.

    Cost = Sum(w_i / distance_i) + Penalty(bounds)
    """
    state = np.asarray(state)

    # 1. Declustering Reward (Weighted Sum of Distances)
    # We want to maximize this.
    dist_reward = 0.0
    for j in range(len(states_list)):
        dist = np.linalg.norm(state - states_list[j])
        dist_reward += w[j] * dist

    # 2. Bounding Penalty
    # We want to minimize this.
    penalty = 0.0
    norm_state = np.linalg.norm(state)

    # Hardcoded bounds from original code (9, 11) - ideally should come from 'limits'
    lower_bound = 9.0
    upper_bound = 11.0

    if norm_state < lower_bound:
        penalty += v * (lower_bound - norm_state) ** 2
    elif norm_state > upper_bound:
        penalty += v * (norm_state - upper_bound) ** 2

    # Total Cost = Penalty - Reward
    # Normalizer from original code: 2 * len(states_list)^2
    normalizer = 2 * (len(states_list) ** 2) if len(states_list) > 0 else 1.0

    cost = (penalty - dist_reward) / normalizer
    return cost

def calc_global_occlusion_cost(
    states_list: np.ndarray,
    W: np.ndarray,
    V: np.ndarray,
    states_0_all: np.ndarray,
    B: np.ndarray,
) -> float:
    """
    Global occlusion cost (to be minimized).
    """
    num_agents = len(states_0_all) // 3
    dist_reward = 0.0
    travel_cost = 0.0
    bound_penalty = 0.0

    for agent_idx in range(num_agents):
        state_agent = states_list[3 * agent_idx : 3 * agent_idx + 3]

        # Declustering (Reward)
        for i in range(num_agents):
            if i != agent_idx:
                state_other = states_list[3 * i : 3 * i + 3]
                # Original used W[i], let's stick to that
                dist_reward += W[i] * np.linalg.norm(state_agent - state_other)

        # Travel Minimization (Cost)
        state_0_agent = states_0_all[3 * agent_idx : 3 * agent_idx + 3]
        travel_cost += V[0] * np.linalg.norm(state_agent - state_0_agent)

        # Bounding (Penalty/Cost)
        norm_state_agent = np.linalg.norm(state_agent)
        if norm_state_agent < B[0]:
            bound_penalty += V[1] * (B[0] - norm_state_agent) ** 2
        elif norm_state_agent > B[1]:
            bound_penalty += V[2] * (norm_state_agent - B[1]) ** 2

    total_cost = (travel_cost + bound_penalty) - dist_reward

    return total_cost / (2 * (num_agents**2))
