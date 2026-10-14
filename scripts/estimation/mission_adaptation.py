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

"""Dense measurement windows and ordered membership/point-estimate updates."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Tuple, TYPE_CHECKING
import numpy as np

from orbexa.estimation.belief import ParameterBelief

if TYPE_CHECKING:
    from orbexa.core.config import SimulationConfig
    from orbexa.core.orbital_parameters import OrbitalPlantTruth


class PointEstimator(Protocol):
    """Point-update API; returned feasible sets are deliberately not authoritative."""

    def update(self, *, feasible_sets: Dict[str, Tuple[float, float]],
               estimates: Dict[str, float], states: np.ndarray, controls: np.ndarray,
               start_anom: float, anom_step: float, dynamics_context: Dict[str, Any]) -> Tuple[Any, Dict[str, float], Any]: ...


def default_point_estimator(config: SimulationConfig) -> PointEstimator:
    from orbexa.estimation.adaptor import SMIDAdaptor
    policy = config.adaptation
    return SMIDAdaptor(error_bound=policy.error_bound,
                       prediction_error_threshold=policy.prediction_error_threshold,
                       max_iter=policy.max_iter, minimum_widths=dict(policy.minimum_widths))


@dataclass
class AdaptationOutcome:
    consistent: bool = True
    messages: List[str] = field(default_factory=list)


class MissionAdaptation:
    """Only interval contraction can remove values from the parameter box."""

    def __init__(self, config: SimulationConfig, truth: OrbitalPlantTruth,
                 point_estimator: PointEstimator, initial_state: np.ndarray, initial_anomaly: float = 0.0):
        self.config = config
        self.truth = truth
        self.point_estimator = point_estimator
        self.window_intervals = int(config.adaptation.window_intervals)
        self.states = deque([np.asarray(initial_state, dtype=float).copy()], maxlen=self.window_intervals + 1)
        self.controls = deque(maxlen=self.window_intervals)
        self.anomalies = deque([float(initial_anomaly)], maxlen=self.window_intervals + 1)
        self.smid_records: List[Dict[str, Any]] = []
        self.membership_records: List[Dict[str, Any]] = []

    def dynamics_context(self) -> Dict[str, float]:
        return {"mean_motion": self.truth.mean_motion, "mu": self.truth.mu,
                "semi_major_axis": self.truth.semi_major_axis,
                "time_periapsis": self.truth.time_periapsis}

    def append_transition(self, state: np.ndarray, control: np.ndarray, anomaly: float) -> None:
        self.controls.append(np.asarray(control, dtype=float).copy())
        self.states.append(np.asarray(state, dtype=float).copy())
        self.anomalies.append(float(anomaly))

    def update(self, belief: ParameterBelief, update_index: int, dense_anom_step: float,
               *, eligible: bool = True) -> AdaptationOutcome:
        outcome = AdaptationOutcome()
        if not eligible or len(self.states) != self.window_intervals + 1:
            return outcome
        states, controls = np.asarray(self.states), np.asarray(self.controls)
        context = self.dynamics_context()
        if update_index % self.config.tube.membership_update_period == 0:
            from orbexa.estimation.interval_membership import contract_parameter_box
            ranges, record = contract_parameter_box(
                belief.feasible_sets, states, controls, self.anomalies[0], dense_anom_step, context,
                error_bound=self.config.tube.membership_error_bound,
                max_boxes=self.config.tube.membership_box_budget,
            )
            belief.feasible_sets = ranges
            for name, interval in ranges.items():
                belief.estimates[name] = float(np.clip(belief.estimates[name], *interval))
            self.membership_records.append(record)
            if record.get("status") == "inconsistent":
                outcome.consistent = False
                outcome.messages.append("Recorded dynamics are inconsistent with the parameter box and measurement bound")
        # Copies also prevent an injected point estimator from mutating the box.
        # The qualified order still performs this update after an inconsistent
        # membership record, and the mission then reports the failure evidence.
        _, point, record = self.point_estimator.update(
            feasible_sets=dict(belief.feasible_sets), estimates=dict(belief.estimates),
            states=states, controls=controls, start_anom=self.anomalies[0],
            anom_step=dense_anom_step, dynamics_context=context,
        )
        point_record = record.to_dict()
        try:
            candidate = ParameterBelief(dict(belief.feasible_sets), dict(point))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            outcome.consistent = False
            reason = f"Point estimator returned an invalid belief: {error}"
            outcome.messages.append(reason)
            point_record["mission_point_estimate_accepted"] = False
            point_record["mission_point_rejection_reason"] = reason
        else:
            belief.estimates = candidate.estimates
        self.smid_records.append(point_record)
        return outcome
