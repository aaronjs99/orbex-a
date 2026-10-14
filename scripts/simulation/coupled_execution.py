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

"""Dense sampled ancillary feedback on one shared fleet clock."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING
import numpy as np

from orbexa.control.dynamic_tube_model import advance_tube, boundary_layer_feedback
from orbexa.simulation.integration import rk4_step

if TYPE_CHECKING:
    from orbexa.estimation.mission_adaptation import MissionAdaptation
    from orbexa.simulation.coupled_planning import ChaserPlan, PlanningStep
    from orbexa.simulation.coupled_state import CoupledMissionState
    from orbexa.simulation.mission_models import MissionPhase
    from orbexa.simulation.mission_scenario import MissionScenario


@dataclass
class ExecutionStep:
    start_anomaly: float
    end_anomaly: float
    success: bool = True
    messages: List[str] = field(default_factory=list)
    reserve_ratios: Dict[str, float] = field(default_factory=dict)
    applied: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    ancillary: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    containment: Dict[str, float] = field(default_factory=dict)
    clearance: Dict[str, float] = field(default_factory=dict)
    min_spacing: float = np.inf
    sampled_tube_contained: bool = True
    max_nominal_discretization_residual: float = 0.0

    @property
    def committed(self) -> bool:
        return self.end_anomaly != self.start_anomaly


class DenseFeedbackExecutor:
    """Check all inputs, propose all transitions, then commit the complete fleet."""

    def __init__(self, scenario: MissionScenario, adaptation: MissionAdaptation):
        self.scenario = scenario
        self.config = scenario.config
        self.adaptation = adaptation
        self.substeps = self.config.tube.feedback_substeps
        self.dense_anom_step = self.config.anom_step / self.substeps

    def advance(self, state: CoupledMissionState, planning: PlanningStep,
                phase: MissionPhase, truth_matrices: tuple) -> ExecutionStep:
        scenario, cfg = self.scenario, self.config
        q = state.anomaly
        ids = list(state.states)
        plans, matrices, params = planning.plans, planning.matrices, planning.parameters
        h, lam = self.dense_anom_step, np.asarray(cfg.tube.sliding_gains)
        outcome = ExecutionStep(start_anomaly=q, end_anomaly=q,
            reserve_ratios={key: 0.0 for key in ids}, applied={key: [] for key in ids},
            ancillary={key: [] for key in ids}, containment={key: np.inf for key in ids},
            clearance={key: np.inf for key in ids})
        for inner in range(self.substeps):
            current = q + inner * h
            proposed = {}
            proposed_centers, proposed_phi, proposed_omega, proposed_bandwidth = {}, {}, {}, {}
            inputs, feedbacks = {}, {}
            for key in ids:
                sol = plans[key]
                u_nom = np.asarray(sol.result.control_trajectory)[:, 0]
                alpha = np.asarray(sol.tube['alpha'])[0]
                a = np.asarray(matrices[0](current, params['time_periapsis']), dtype=float)[3:]
                feedback = boundary_layer_feedback(a, sol.spec.delta_A[0], state.states[key],
                    state.centers[key], state.phi[key], alpha, lam)
                u = u_nom + feedback
                reserve = np.asarray(sol.tube['input_reserve'])[0]
                outcome.reserve_ratios[key] = max(outcome.reserve_ratios[key],
                    float(np.max(np.abs(feedback) / np.maximum(reserve, 1e-12))))
                if np.max(np.abs(feedback) - reserve) > 1e-5:
                    outcome.success = False
                    outcome.messages.append(f'{key}: ancillary feedback exceeded the planned input reserve')
                    break
                clipped = scenario.safety.clip_control(u)
                if np.max(np.abs(clipped - u)) > 1e-8:
                    outcome.success = False
                    outcome.messages.append(f'{key}: ancillary input would saturate; tube policy interrupted')
                    break
                inputs[key], feedbacks[key] = u, feedback
            if not outcome.success:
                break
            for key in ids:
                sol = plans[key]
                u_nom = np.asarray(sol.result.control_trajectory)[:, 0]
                alpha = np.asarray(sol.tube['alpha'])[0]
                u = inputs[key]
                proposed[key] = rk4_step(state.states[key], u, current, h, truth_matrices, scenario.truth.time_periapsis)
                proposed_centers[key] = rk4_step(state.centers[key], u_nom, current, h, matrices, params['time_periapsis'])
                proposed_phi[key], proposed_omega[key] = advance_tube(state.phi[key], state.omega[key], alpha, lam,
                    np.asarray(sol.tube['disturbance_bound'])[0], h)
                proposed_bandwidth[key] = alpha
            state.commit(current + h, states=proposed, centers=proposed_centers,
                         phi=proposed_phi, omega=proposed_omega, bandwidth=proposed_bandwidth)
            outcome.end_anomaly = current + h
            self.adaptation.append_transition(state.states[ids[0]], inputs[ids[0]], current + h)
            for key in ids:
                outcome.applied[key].append(inputs[key])
                outcome.ancillary[key].append(feedbacks[key])
                error = state.states[key] - state.centers[key]
                margin = min(float(np.min(state.omega[key] - np.abs(error[:3]))),
                             float(np.min(state.phi[key] - np.abs(error[3:] + lam * error[:3]))))
                clearance = scenario.safety.active_margin_for_state(state.states[key], current + h, phase)
                outcome.containment[key] = min(outcome.containment[key], margin)
                outcome.clearance[key] = min(outcome.clearance[key], clearance)
                outcome.sampled_tube_contained &= margin >= -1e-8
                if margin < -1e-8 or clearance < -scenario.safety.target_safety_tolerance:
                    outcome.success = False
                    outcome.messages.append(f'{key}: dense containment or clearance violated at q={current+h:.9g}')
            for i, key in enumerate(ids):
                for other in ids[i + 1:]:
                    distance = float(np.linalg.norm(state.states[key][:3] - state.states[other][:3]))
                    outcome.min_spacing = min(outcome.min_spacing, distance)
                    if distance < scenario.safety.min_chaser_separation - scenario.safety.target_safety_tolerance:
                        outcome.success = False
                        outcome.messages.append('dense pairwise spacing violated')
            if not outcome.success:
                break
        if outcome.committed and outcome.success:
            for key in ids:
                sol = plans[key]
                residual = np.linalg.norm(state.centers[key] - np.asarray(sol.result.state_trajectory)[:, 1])
                outcome.max_nominal_discretization_residual = max(outcome.max_nominal_discretization_residual, float(residual))
                state.phi[key] = np.maximum(state.phi[key], sol.profile.phi[1])
                state.omega[key] = np.maximum(state.omega[key], sol.profile.omega[1])
        return outcome
