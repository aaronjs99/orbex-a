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

"""Mission history and diagnostic recording, separate from solver execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from orbexa.control.tubes.model import input_tightening_from_profile
from orbexa.estimation.adaptation.belief import (
    ParameterBelief,
    serial_estimates,
    serial_ranges,
)

if TYPE_CHECKING:
    from orbexa.estimation.adaptation.belief import ParameterBelief
    from orbexa.simulation.approach.models import ChaserConfig
    from orbexa.simulation.approach.planning import PlanningStep
    from orbexa.simulation.approach.scenario import MissionScenario
    from orbexa.simulation.approach.state import CoupledMissionState


class MissionRecorder:
    """Retain every solve attempt and every committed dense interval endpoint."""

    def __init__(
        self,
        scenario: MissionScenario,
        chasers: Sequence[ChaserConfig],
        state: CoupledMissionState,
        initial_belief: ParameterBelief,
    ):
        self.scenario, self.config, self.chasers = scenario, scenario.config, chasers
        self.initial_belief = initial_belief.copy()
        self.ids = [c.chaser_id for c in chasers]
        (
            self.actual,
            self.nominal,
            self.controls,
            self.stats,
            self.costs,
            self.profiles,
        ) = ({key: [] for key in self.ids} for _ in range(6))
        self.diagnostics = {
            key: {
                "nominal_center": [],
                "phi": [],
                "omega": [],
                "bandwidth": [],
                "position_error": [],
                "sliding_error": [],
                "containment_margin": [],
                "terminal_position_bound": [],
                "terminal_velocity_bound": [],
                "max_dense_input_reserve_ratio": [],
                "nominal_control": [],
                "ancillary_control_rms": [],
                "applied_control_rms": [],
                "applied_control_effort": [],
                "dense_containment_margin": [],
                "dense_physical_clearance": [],
                "nominal_center_defect": [],
            }
            for key in self.ids
        }
        for key in self.ids:
            self.actual[key].append(state.states[key].tolist())
            self.record_tube_sample(
                self.diagnostics[key],
                state.states[key],
                state.centers[key],
                state.phi[key],
                state.omega[key],
                state.bandwidth[key],
            )
        q = state.anomaly
        self.q_history, self.elapsed = [q], [0.0]
        self.phases, self.sample_phases = [], ["rendezvous"]
        self.radius_history = [0.0]
        self.attitudes = [scenario.target.orientation_at(q).tolist()]
        # A proper-rotation target need not expose a nonsingular Euler-rate
        # chart. Its authoritative matrices/spins are recorded at result time.
        self.records_kinematics = callable(
            getattr(scenario.target, "recorded_kinematics", None)
        )
        self.rates = (
            []
            if self.records_kinematics
            else [scenario.target.angular_velocity_at(q).tolist()]
        )
        self.feasible = [serial_ranges(initial_belief.feasible_sets)]
        self.estimates = [serial_estimates(initial_belief.estimates)]
        self.contained = True
        self.dense_min_clearance, self.dense_min_spacing, self.dense_min_containment = (
            np.inf,
            np.inf,
            np.inf,
        )
        self.max_nominal_discretization_residual = 0.0
        self.numerical_restart_history = []
        self.source_uncommitted_planning = {}
        self.source_kinematics = None
        self.source_kinematic_nodes = 0
        self.arrival_runtime = None
        self.policy_transition_history = []

    def tube_metadata(self, anom: float, profile) -> dict[str, Any]:
        return {
            "start_anom": float(anom),
            "time_grid": profile.time_grid.tolist(),
            "phi": profile.phi.tolist(),
            "omega": profile.omega.tolist(),
            "position_radius": profile.position_radius.tolist(),
            "max_position_radius": profile.max_position_radius,
            "max_state_error": profile.max_state_error.tolist(),
            "lambda_gain": None
            if profile.lambda_gain is None
            else np.asarray(profile.lambda_gain).tolist(),
            "bandwidth": None
            if profile.alpha is None
            else np.asarray(profile.alpha).tolist(),
            "input_reserve": None
            if profile.input_reserve is None
            else np.asarray(profile.input_reserve).tolist(),
            "assumptions": list(profile.assumptions),
            "tube_model_version": profile.tube_model_version,
            "hold_width": profile.hold_width,
            "stage_forcing": {}
            if profile.forcing is None
            else {
                name: np.asarray(value).tolist()
                for name, value in profile.forcing.items()
            },
            "input_tightening": input_tightening_from_profile(
                profile, self.config.tube.sliding_gains
            ).tolist(),
        }

    def record_tube_sample(self, data, state, center, phi, omega, bandwidth) -> None:
        error = state - center
        sliding = error[3:] + np.asarray(self.config.tube.sliding_gains) * error[:3]
        for key, value in {
            "nominal_center": center,
            "phi": phi,
            "omega": omega,
            "bandwidth": bandwidth,
            "position_error": error[:3],
            "sliding_error": sliding,
        }.items():
            data[key].append(np.asarray(value).tolist())
        data["containment_margin"].append(
            float(min(np.min(omega - np.abs(error[:3])), np.min(phi - np.abs(sliding))))
        )

    def record_planning(self, planning: PlanningStep, anomaly: float) -> None:
        for key, result in planning.attempts.items():
            self.stats[key].append(
                {
                    "anomaly": float(anomaly),
                    "success": bool(result.success),
                    "solve_time": float(result.solve_time),
                    "message": result.message,
                    "solver_info": {
                        name: value
                        for name, value in result.solver_info.items()
                        if name != "dynamic_tube"
                    },
                }
            )
            if not result.success and "dynamic_tube" in result.solver_info:
                self.stats[key][-1]["failed_dynamic_tube"] = {
                    name: value.tolist() if isinstance(value, np.ndarray) else value
                    for name, value in result.solver_info["dynamic_tube"].items()
                }
            self.costs[key].append(result.cost)
            if key in planning.plans:
                self.nominal[key].append(np.asarray(result.state_trajectory).T.tolist())
                self.profiles[key].append(
                    self.tube_metadata(anomaly, planning.plans[key].profile)
                )

    def record_belief(self, belief: ParameterBelief) -> None:
        self.feasible.append(serial_ranges(belief.feasible_sets))
        self.estimates.append(serial_estimates(belief.estimates))
