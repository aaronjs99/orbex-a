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


"""Abstract spacecraft base and lazy compatibility facade for the concrete agents."""
from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import count
import random
from typing import TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:
    from orbexa.core.config import SimulationConfig
    from orbexa.core.target_spacecraft import Target
    from orbexa.core.chaser_spacecraft import Chaser


def target_state_update_func(t, state, dt, mom_inertia, skew_sym_mat, torque_val, torque_type, *args, **kwargs):
    """Legacy six-state Euler RHS with current body rates and correct kinematics.

    The cached skew matrix is accepted for compatibility and ignored. A singular
    Euler chart rejects explicitly; Target itself propagates a quaternion.
    """
    from orbexa.core.attitude import (finite_vector, evaluate_torque,
                                      body_rates_to_euler_rates, body_rate_derivative, inertia_matrix)

    state = finite_vector(state, "Attitude state", 6)
    inertia = inertia_matrix(mom_inertia)
    torque = evaluate_torque(t, state, dt, inertia, torque_val, torque_type)
    return np.concatenate((body_rates_to_euler_rates(state[:3], state[3:]),
                           body_rate_derivative(state[3:], inertia, torque)))


class Spacecraft(ABC):
    """Base class for all spacecraft agents."""

    _ids = count(0)

    def __init__(self, config: "SimulationConfig", **kwargs):
        """
        Initialize Spacecraft.

        Args:
            config: SimulationConfig object.
            **kwargs: Overrides for specific properties like 'name' or 'initState'.
        """
        self.config = config
        self.anom_step = config.anom_step

        self.id = next(self._ids)
        self.name = kwargs.get("name", "")
        self.num_states = kwargs.get("numStates", 6)
        if (isinstance(self.num_states, bool) or not isinstance(self.num_states, (int, np.integer))
                or self.num_states < 1):
            raise ValueError("numStates must be a positive integer")
        from orbexa.core.attitude import finite_vector

        self.init_state = finite_vector(kwargs.get("initState", np.zeros(self.num_states)), "Initial state", self.num_states)
        raw_dt = kwargs.get("dt", self.anom_step)
        try:
            dt = np.asarray(raw_dt, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("dt must be a finite positive scalar") from error
        if isinstance(raw_dt, (bool, np.bool_)) or dt.shape != () or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive scalar")
        self.dt = float(dt)
        self.rng = kwargs.get("rng", random.Random(int(config.seed) + self.id))
        self.curr_state = self.init_state.copy()
        self.state_history = [self.curr_state.copy()]

    @abstractmethod
    def update_state(self, *args, **kwargs):
        """Advance this spacecraft's explicit state model."""

    def plot_state_history(self, params: Dict, *args, **kwargs):
        """Plot the history of states."""
        import matplotlib.pyplot as plt

        ind_state_history = np.transpose(self.state_history)
        cmap = plt.get_cmap("viridis", 256)

        plt.figure(figsize=(4, 3))
        num_points = ind_state_history.shape[1]
        time_seq = np.arange(num_points) * self.dt

        if params.get("sep_plots", False):
            for i in range(self.num_states):
                plt.subplot(self.num_states, 1, i + 1)
                label = f"$x_{i}$"
                plt.plot(time_seq, ind_state_history[i], c=cmap(96 * i), label=label)
                plt.legend()
                plt.ylabel("State History")
                plt.xlabel("Time")
        else:
            plt.subplot(1, 1, 1)
            for i in range(self.num_states):
                label = f"$x_{i}$"
                plt.plot(time_seq, ind_state_history[i], c=cmap(96 * i), label=label)
            plt.legend()
            plt.ylabel("State History")
            plt.xlabel("Time")

        if params.get("disp_plot", True) and "fLoc" not in kwargs:
            plt.show()
        elif "fLoc" in kwargs:
            plt.gcf().set_size_inches(10 * plt.gcf().get_size_inches())
            plt.tight_layout()
            plt.savefig(kwargs["fLoc"] + "target_ang_pos.png")
            plt.close()



def __getattr__(name):
    if name == "Target":
        from orbexa.core.target_spacecraft import Target
        globals()[name] = Target
        return Target
    if name == "Chaser":
        from orbexa.core.chaser_spacecraft import Chaser
        globals()[name] = Chaser
        return Chaser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Spacecraft", "Target", "Chaser", "target_state_update_func"]
