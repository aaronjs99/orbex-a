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

"""Persistent nominal centers and tube envelopes on the shared fleet clock."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from orbexa.core.config import SimulationConfig
    from orbexa.simulation.mission_models import ChaserConfig


@dataclass
class CoupledMissionState:
    anomaly: float
    states: Dict[str, np.ndarray]
    centers: Dict[str, np.ndarray]
    phi: Dict[str, np.ndarray]
    omega: Dict[str, np.ndarray]
    bandwidth: Dict[str, np.ndarray]

    @classmethod
    def initialize(cls, chasers: Sequence[ChaserConfig], config: SimulationConfig) -> "CoupledMissionState":
        ids = [c.chaser_id for c in chasers]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("A mission fleet requires nonempty, unique chaser identities")
        states = {c.chaser_id: np.array(c.initial_state, dtype=float) for c in chasers}
        return cls(anomaly=0.0, states=states,
                   centers={key: value.copy() for key, value in states.items()},
                   phi={key: np.array(config.tube.boundary_layer_0, dtype=float) for key in states},
                   omega={key: np.zeros(3) for key in states},
                   bandwidth={key: np.array(config.tube.bandwidth_0, dtype=float) for key in states})

    def commit(self, anomaly: float, *, states: Dict[str, np.ndarray], centers: Dict[str, np.ndarray],
               phi: Dict[str, np.ndarray], omega: Dict[str, np.ndarray], bandwidth: Dict[str, np.ndarray]) -> None:
        """Commit one jointly proposed dense clock step after all inputs are feasible."""
        self.states, self.centers = states, centers
        self.phi, self.omega, self.bandwidth = phi, omega, bandwidth
        self.anomaly = float(anomaly)
