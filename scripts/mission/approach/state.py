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

"""state.py: Persistent nominal centers and tube envelopes on the shared fleet clock.

Copyright: Aaron John Sabu
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from orbexa.mission.approach.arrival.policy import ArrivalCommitment
    from orbexa.mission.approach.models import ChaserConfig
    from orbexa.models.configuration.model import SimulationConfig


@dataclass
class CoupledMissionState:
    anomaly: float
    states: dict[str, np.ndarray]
    centers: dict[str, np.ndarray]
    phi: dict[str, np.ndarray]
    omega: dict[str, np.ndarray]
    bandwidth: dict[str, np.ndarray]
    arrival_commitment: ArrivalCommitment | None = None

    @classmethod
    def initialize(
        cls, chasers: Sequence[ChaserConfig], config: SimulationConfig
    ) -> CoupledMissionState:
        ids = [c.chaser_id for c in chasers]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(
                "A mission fleet requires nonempty, unique chaser identities"
            )
        states = {c.chaser_id: np.array(c.initial_state, dtype=float) for c in chasers}
        return cls(
            anomaly=0.0,
            states=states,
            centers={key: value.copy() for key, value in states.items()},
            phi={
                key: np.array(config.tube.boundary_layer_0, dtype=float)
                for key in states
            },
            omega={key: np.zeros(3) for key in states},
            bandwidth={
                key: np.array(config.tube.bandwidth_0, dtype=float) for key in states
            },
        )

    def commit(
        self,
        anomaly: float,
        *,
        states: dict[str, np.ndarray],
        centers: dict[str, np.ndarray],
        phi: dict[str, np.ndarray],
        omega: dict[str, np.ndarray],
        bandwidth: dict[str, np.ndarray],
    ) -> None:
        """Commit one jointly proposed dense clock step after all inputs are feasible."""
        self.states, self.centers = states, centers
        self.phi, self.omega, self.bandwidth = phi, omega, bandwidth
        self.anomaly = float(anomaly)
